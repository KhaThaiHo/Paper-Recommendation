#!/usr/bin/env python3
"""extractor_paper.py — simple per-record paper extraction (restored old behavior)

Extracts `scientific_domains` and `research_focuses` from papers using the same
style as the journal extractor: per-row LLM calls with retries, checkpointing
and failed logging.
"""

import csv
import json
import time
import logging
import re
import sys
from pathlib import Path

from tqdm import tqdm

from config import (
	INPUT_PAPERS_CSV, OUTPUT_PAPERS_JSONL, CHECKPOINT_PAPERS, FAILED_PAPERS,
	COL_TITLE, COL_ABSTRACT, COL_KEYWORDS, COL_LABEL,
	BATCH_SIZE, MAX_RETRIES, RETRY_DELAY_SEC, SLEEP_BETWEEN_MS, SCHEMA_VERSION,
	PROMPT_BATCH_SIZE,
)
from prompt_builder_paper import build_prompt_paper, build_prompt_paper_no_system

# Reuse model call and loader from extractor.py when available
try:
	from extractor import call_model, call_model_batch, _load_transformers_model, health_check
except Exception:
	call_model = None
	call_model_batch = None
	_load_transformers_model = None
	health_check = None

Path("output").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s [%(levelname)s] %(message)s",
	handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("output/pipeline_papers.log", encoding="utf-8")],
)
log = logging.getLogger(__name__)

REQUIRED_KEYS = {
	"scientific_domains",
	"scientific_domains_evidence",
	"research_focuses",
	"research_focuses_evidence",
}


def repair_json(text: str) -> str:
	text = re.sub(r'"([^\"]*)"\s+[^",\]\}\[\{]+(?=[,\]\}])', r'"\1"', text)
	text = re.sub(r',\s*([\}\]])', r'\1', text)
	open_braces = text.count('{') - text.count('}')
	open_brackets = text.count('[') - text.count(']')
	if open_braces > 0 or open_brackets > 0:
		stripped = text.rstrip()
		n_quotes = len(re.findall(r'(?<!\\)"', stripped))
		if n_quotes % 2 == 1:
			stripped += '"'
		text = stripped + (']' * max(0, open_brackets)) + ('}' * max(0, open_braces))
	return text


def extract_json_from_response(text: str) -> dict | None:
	text = text.strip()
	text = re.sub(r'<think>[\s\S]*?</think>', '', text).strip()
	text = re.sub(r'Thinking Process:[\s\S]*?(?=\{)', '', text).strip()
	fence_match = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', text)
	if fence_match:
		text = fence_match.group(1).strip()
	start = text.find('{')
	if start == -1:
		return None
	depth = 0
	end = -1
	for i, ch in enumerate(text[start:], start):
		if ch == '{': depth += 1
		elif ch == '}':
			depth -= 1
			if depth == 0:
				end = i
				break
	json_str = text[start:end+1] if end != -1 else text[start:]
	try:
		return json.loads(json_str)
	except json.JSONDecodeError:
		pass
	repaired = repair_json(json_str)
	try:
		return json.loads(repaired)
	except json.JSONDecodeError:
		pass
	return None


def validate_output(data: dict) -> tuple[bool, str]:
	for key in REQUIRED_KEYS:
		if key not in data:
			return False, f"Missing key: {key}"
	if not isinstance(data["scientific_domains"], list):
		return False, "scientific_domains must be a list"
	if not isinstance(data["research_focuses"], list):
		return False, "research_focuses must be a list"
	if not isinstance(data["scientific_domains_evidence"], dict):
		return False, "scientific_domains_evidence must be a dict"
	if not isinstance(data["research_focuses_evidence"], dict):
		return False, "research_focuses_evidence must be a dict"
	if len(data["scientific_domains"]) == 0:
		return False, "scientific_domains is empty"
	if len(data["research_focuses"]) == 0:
		return False, "research_focuses is empty"
	return True, "ok"


def load_checkpoint() -> int:
	if CHECKPOINT_PAPERS.exists():
		content = CHECKPOINT_PAPERS.read_text().strip()
		if content.isdigit():
			return int(content)
	return -1


def save_checkpoint(idx: int):
	CHECKPOINT_PAPERS.write_text(str(idx))


def load_processed_ids() -> set[str]:
	processed = set()
	if OUTPUT_PAPERS_JSONL.exists():
		with open(OUTPUT_PAPERS_JSONL, encoding="utf-8") as f:
			for line in f:
				line = line.strip()
				if line:
					try:
						record = json.loads(line)
						processed.add(record.get("title", ""))
					except json.JSONDecodeError:
						pass
	return processed


def log_failed(row: dict, error: str, out_file):
	record = dict(row)
	record["error"] = error
	out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
	out_file.flush()


def process_one(row: dict, use_system_role: bool = True) -> dict | None:
	t0 = time.perf_counter()
	title = row.get(COL_TITLE, "").strip()
	abstract = row.get(COL_ABSTRACT, "").strip()
	keywords = row.get(COL_KEYWORDS, "").strip()

	build_fn = build_prompt_paper if use_system_role else build_prompt_paper_no_system

	for attempt in range(1, MAX_RETRIES + 1):
		messages = build_fn(title, keywords, abstract)

		if call_model is None:
			log.error("call_model not available (ensure extractor.py is importable)")
			return None

		raw = call_model(messages)
		t1 = time.perf_counter()
		log.info(f"[PROFILE] INFERENCE for '{title[:60]}': {t1 - t0:.2f} seconds (attempt {attempt})")
		if raw is None:
			log.warning(f"  Attempt {attempt}/{MAX_RETRIES}: No response for '{title[:60]}'")
			time.sleep(RETRY_DELAY_SEC)
			continue

		parsed = extract_json_from_response(raw)
		if parsed is None:
			log.warning(f"  Attempt {attempt}/{MAX_RETRIES}: Could not parse JSON for '{title[:60]}'")
			log.warning(f"  Raw output (first 600 chars):\n{raw[:600]}")
			time.sleep(RETRY_DELAY_SEC)
			continue

		valid, reason = validate_output(parsed)
		if not valid:
			log.warning(f"  Attempt {attempt}/{MAX_RETRIES}: Validation failed ({reason}) for '{title[:60]}'")
			time.sleep(RETRY_DELAY_SEC)
			continue

		return parsed

	return None


def process_batch(rows: list[dict], use_system_role: bool = True) -> list[dict | None]:
	"""Process multiple rows in one model.generate() call.

	Items that fail parsing/validation are returned as None so the caller
	can retry them individually via process_one().
	"""
	if call_model_batch is None:
		return [process_one(row, use_system_role) for row in rows]

	build_fn = build_prompt_paper if use_system_role else build_prompt_paper_no_system

	messages_list = [
		build_fn(
			row.get(COL_TITLE, "").strip(),
			row.get(COL_KEYWORDS, "").strip(),
			row.get(COL_ABSTRACT, "").strip(),
		)
		for row in rows
	]

	raw_list = call_model_batch(messages_list)

	results: list[dict | None] = []
	for row, raw in zip(rows, raw_list):
		title = row.get(COL_TITLE, "").strip()
		if raw is None:
			results.append(None)
			continue
		parsed = extract_json_from_response(raw)
		if parsed is None:
			log.warning(f"  Batch parse failed for '{title[:60]}' — will retry individually")
			results.append(None)
			continue
		valid, reason = validate_output(parsed)
		if not valid:
			log.warning(f"  Batch validation failed ({reason}) for '{title[:60]}' — will retry individually")
			results.append(None)
			continue
		results.append(parsed)

	return results


def run():
	if not INPUT_PAPERS_CSV.exists():
		log.error(f"Input file not found: {INPUT_PAPERS_CSV}")
		sys.exit(1)

	with open(INPUT_PAPERS_CSV, encoding="utf-8") as f:
		rows = list(csv.DictReader(f))
	total = len(rows)
	log.info(f"Loaded {total} papers from {INPUT_PAPERS_CSV}")

	last_idx = load_checkpoint()
	start_idx = last_idx + 1
	already_processed = load_processed_ids()

	if start_idx > 0:
		log.info(f"Resuming from index {start_idx} (checkpoint: {last_idx})")
	else:
		log.info("Starting fresh extraction for papers")

	stats = {"success": 0, "failed": 0, "skipped": 0}

	if health_check is None:
		try:
			if _load_transformers_model is not None:
				_load_transformers_model()
		except Exception as e:
			log.error(f"Model load failed: {e}")
			sys.exit(1)
	else:
		if not health_check():
			sys.exit(1)

	# Pre-filter: separate rows to skip from rows to process (preserving original indices)
	pending: list[tuple[int, dict]] = []
	for idx in range(start_idx, total):
		row = rows[idx]
		title = row.get(COL_TITLE, "").strip()
		if title in already_processed:
			log.debug(f"  Skipping already-processed: '{title}'")
			stats["skipped"] += 1
			save_checkpoint(idx)
		elif not title or not row.get(COL_ABSTRACT, "").strip():
			log.warning(f"  Skipping row {idx} — empty title or abstract")
			stats["skipped"] += 1
			save_checkpoint(idx)
		else:
			pending.append((idx, row))

	log.info(
		f"Pending: {len(pending)} | Skipped: {stats['skipped']} | "
		f"Batch size: {PROMPT_BATCH_SIZE}"
	)

	with open(OUTPUT_PAPERS_JSONL, "a", encoding="utf-8") as out_f, \
	     open(FAILED_PAPERS, "a", encoding="utf-8") as fail_f:

		pbar = tqdm(total=len(pending), desc="Extracting", unit="paper")

		for batch_start in range(0, len(pending), PROMPT_BATCH_SIZE):
			batch = pending[batch_start:batch_start + PROMPT_BATCH_SIZE]
			batch_rows = [row for _, row in batch]

			batch_results = process_batch(batch_rows)

			for (idx, row), result in zip(batch, batch_results):
				title = row.get(COL_TITLE, "").strip()
				pbar.set_postfix({
					"paper": title[:30] + "…" if len(title) > 30 else title,
					"ok": stats["success"],
					"fail": stats["failed"],
				})

				# Retry individually if batch inference failed for this item
				if result is None:
					result = process_one(row)

				if result is not None:
					record = {
						"_schema": SCHEMA_VERSION,
						"_idx": idx,
						"title": title,
						"label": row.get(COL_LABEL, "").strip(),
						"keywords": row.get(COL_KEYWORDS, "").strip(),
						"abstract": row.get(COL_ABSTRACT, "").strip(),
						**result,
					}
					out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
					out_f.flush()
					already_processed.add(title)
					stats["success"] += 1
					log.debug(f"  ✓ Saved '{title}'")
				else:
					log_failed(row, "All retries exhausted", fail_f)
					stats["failed"] += 1
					log.warning(f"  ✗ Failed '{title}' — logged to failed_papers.jsonl")

				save_checkpoint(idx)
				pbar.update(1)

				if (idx + 1) % BATCH_SIZE == 0:
					log.info(
						f"Progress {idx+1}/{total} | "
						f"✓ {stats['success']} | ✗ {stats['failed']} | skip {stats['skipped']}"
					)

				if SLEEP_BETWEEN_MS > 0:
					time.sleep(SLEEP_BETWEEN_MS / 1000)

		pbar.close()

	log.info("=" * 60)
	log.info(f"DONE. Total: {total} | ✓ Success: {stats['success']} | ✗ Failed: {stats['failed']} | Skipped: {stats['skipped']}")
	log.info(f"Output  → {OUTPUT_PAPERS_JSONL}")
	log.info(f"Failed  → {FAILED_PAPERS}")
	log.info("=" * 60)


def retry_failed():
	if not FAILED_PAPERS.exists():
		log.info("No failed_papers.jsonl found.")
		return

	with open(FAILED_PAPERS, encoding="utf-8") as f:
		failed_rows = [json.loads(l) for l in f if l.strip()]

	if not failed_rows:
		log.info("failed_papers.jsonl is empty.")
		return

	valid_rows = [r for r in failed_rows if r.get(COL_TITLE, r.get("title", "")).strip()]
	invalid_rows = [r for r in failed_rows if not r.get(COL_TITLE, r.get("title", "")).strip()]

	if invalid_rows:
		log.warning(f"Skipping {len(invalid_rows)} rows with empty title")

	log.info(f"Retrying {len(valid_rows)} failed papers...")
	new_failures = []

	with open(OUTPUT_PAPERS_JSONL, "a", encoding="utf-8") as out_f:
		for row in tqdm(valid_rows, desc="Retrying"):
			result = process_one(row)
			if result is not None:
				record = {
					"_schema": SCHEMA_VERSION,
					"_idx": -1,
					"title": row.get(COL_TITLE, row.get("title", "")),
					"label": row.get(COL_LABEL, row.get("label", "")),
					"keywords": row.get(COL_KEYWORDS, row.get("keywords", "")),
					"abstract": row.get(COL_ABSTRACT, row.get("abstract", "")),
					**result,
				}
				out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
				out_f.flush()
				log.info(f"  ✓ Recovered: '{row.get(COL_TITLE, row.get('title', ''))}'")
			else:
				new_failures.append(row)

	with open(FAILED_PAPERS, "w", encoding="utf-8") as f:
		for r in new_failures:
			f.write(json.dumps(r, ensure_ascii=False) + "\n")

	log.info(f"Retry done. Recovered: {len(failed_rows) - len(new_failures)} | Still failed: {len(new_failures)}")


if __name__ == "__main__":
	import argparse
	from pathlib import Path as _Path
	import config as _cfg

	parser = argparse.ArgumentParser(
		description="Paper metadata extraction pipeline",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument(
		"--mode", choices=["run", "retry"], default="run",
		help="'run' = full pipeline; 'retry' = re-process failed rows only",
	)
	# ── Path overrides (essential for Kaggle) ────────────────────
	parser.add_argument("--input", default=None,
		help="Input CSV path (overrides config.INPUT_PAPERS_CSV)")
	parser.add_argument("--output", default=None,
		help="Output JSONL path (overrides config.OUTPUT_PAPERS_JSONL)")
	parser.add_argument("--checkpoint", default=None,
		help="Checkpoint file path (overrides config.CHECKPOINT_PAPERS)")
	parser.add_argument("--failed", default=None,
		help="Failed log path (overrides config.FAILED_PAPERS)")
	# ── Model / inference overrides ───────────────────────────────
	parser.add_argument("--model", default=None,
		help="HuggingFace model id (e.g. Qwen/Qwen3.5-4B-Instruct)")
	parser.add_argument("--prompt-batch-size", type=int, default=None,
		dest="prompt_batch_size",
		help="Papers per LLM batch call (overrides config.PROMPT_BATCH_SIZE)")
	parser.add_argument("--max-new-tokens", type=int, default=None,
		dest="max_new_tokens",
		help="Max output tokens per generation (overrides config.MODEL_MAX_NEW_TOKENS)")
	parser.add_argument("--batch-size", type=int, default=None,
		dest="batch_size",
		help="Checkpoint/log interval in records (overrides config.BATCH_SIZE)")
	args = parser.parse_args()

	# ── Apply path overrides ──────────────────────────────────────
	if args.input:
		_cfg.INPUT_PAPERS_CSV = _Path(args.input)
		globals()["INPUT_PAPERS_CSV"] = _Path(args.input)
		log.info(f"Input overridden to: {args.input}")
	if args.output:
		_cfg.OUTPUT_PAPERS_JSONL = _Path(args.output)
		globals()["OUTPUT_PAPERS_JSONL"] = _Path(args.output)
		log.info(f"Output overridden to: {args.output}")
	if args.checkpoint:
		_cfg.CHECKPOINT_PAPERS = _Path(args.checkpoint)
		globals()["CHECKPOINT_PAPERS"] = _Path(args.checkpoint)
		log.info(f"Checkpoint overridden to: {args.checkpoint}")
	if args.failed:
		_cfg.FAILED_PAPERS = _Path(args.failed)
		globals()["FAILED_PAPERS"] = _Path(args.failed)
		log.info(f"Failed log overridden to: {args.failed}")

	# ── Apply model / inference overrides ─────────────────────────
	if args.model:
		_cfg.MODEL_NAME = args.model
		try:
			import extractor as _ext
			_ext.MODEL_NAME = args.model
		except Exception:
			pass
		log.info(f"Model overridden to: {args.model}")
	if args.prompt_batch_size:
		_cfg.PROMPT_BATCH_SIZE = args.prompt_batch_size
		globals()["PROMPT_BATCH_SIZE"] = args.prompt_batch_size
		log.info(f"PROMPT_BATCH_SIZE overridden to: {args.prompt_batch_size}")
	if args.max_new_tokens:
		_cfg.MODEL_MAX_NEW_TOKENS = args.max_new_tokens
		try:
			import extractor as _ext
			_ext.MODEL_MAX_NEW_TOKENS = args.max_new_tokens
		except Exception:
			pass
		log.info(f"MODEL_MAX_NEW_TOKENS overridden to: {args.max_new_tokens}")
	if args.batch_size:
		_cfg.BATCH_SIZE = args.batch_size
		globals()["BATCH_SIZE"] = args.batch_size
		log.info(f"BATCH_SIZE overridden to: {args.batch_size}")

	if args.mode == "retry":
		retry_failed()
	else:
		run()
