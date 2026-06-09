#!/usr/bin/env python3
# ============================================================
# extractor.py — Main journal extraction pipeline
#
# Features:
#   ✓ Checkpoint/resume  → safe to kill and restart anytime
#   ✓ Per-record retry   → retries failed LLM calls N times
#   ✓ Failed log         → saves broken rows separately
#   ✓ Progress bar       → tqdm with ETA
#   ✓ JSON validation    → ensures output matches schema
#   ✓ Append-only JSONL  → never overwrites past work
#   ✓ Stats summary      → prints success/fail counts at end
# ============================================================

import csv
import json
import time
import logging
import re
import sys
from pathlib import Path

import requests
from tqdm import tqdm

from config import (
    INPUT_CSV, OUTPUT_JSONL, CHECKPOINT_FILE, FAILED_LOG,
    COL_JOURNAL, COL_AIMS, COL_LABEL, COL_CATEGORIES,
    OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT, OLLAMA_OPTIONS,
    BATCH_SIZE, MAX_RETRIES, RETRY_DELAY_SEC, SLEEP_BETWEEN_MS,
    SCHEMA_VERSION,
)
from prompt_builder import build_prompt, build_prompt_no_system

# ── Logging setup ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("output/pipeline.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

REQUIRED_KEYS = {
    "scientific_domains",
    "scientific_domains_evidence",
    "research_focuses",
    "research_focuses_evidence",
}


# ── Helpers ───────────────────────────────────────────────────

def ensure_dirs():
    Path("output").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)


def load_checkpoint() -> int:
    """Return the last successfully processed row index (0-based). -1 if fresh start."""
    if CHECKPOINT_FILE.exists():
        content = CHECKPOINT_FILE.read_text().strip()
        if content.isdigit():
            return int(content)
    return -1


def save_checkpoint(idx: int):
    CHECKPOINT_FILE.write_text(str(idx))


def load_processed_ids() -> set[str]:
    """Load already-processed journal names from output JSONL to allow exact dedup."""
    processed = set()
    if OUTPUT_JSONL.exists():
        with open(OUTPUT_JSONL, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        processed.add(record.get("journal", ""))
                    except json.JSONDecodeError:
                        pass
    return processed


def repair_json(text: str) -> str:
    """
    Attempt to fix common LLM JSON errors before parsing:
    1. Inline comments:  ["value" is implied by something]  →  ["value"]
    2. Trailing commas:  ["a", "b",]  →  ["a", "b"]
    3. Truncated JSON:   close any unclosed braces/brackets
    """
    # Remove inline comments inside arrays: "text" is something → just remove the comment part
    # Pattern: inside [...], after a closing quote, there's unquoted text before ] or ,
    text = re.sub(r'"([^"]*)"\s+[^",\]\}\[\{]+(?=[,\]\}])', r'"\1"', text)

    # Remove trailing commas before } or ]
    text = re.sub(r',\s*([\}\]])', r'\1', text)

    # If JSON is truncated (unbalanced braces), try to close it
    open_braces   = text.count('{') - text.count('}')
    open_brackets = text.count('[') - text.count(']')
    if open_braces > 0 or open_brackets > 0:
        stripped = text.rstrip()
        # Close any unterminated string (count unescaped quotes)
        n_quotes = len(re.findall(r'(?<!\\)"', stripped))
        if n_quotes % 2 == 1:
            stripped += '"'           # close the open string
        # Add closing brackets then braces
        text = stripped + (']' * max(0, open_brackets)) + ('}' * max(0, open_braces))

    return text


def extract_json_from_response(text: str) -> dict | None:
    """
    Robustly extract JSON from LLM output.
    Handles:
      - qwen3 <think>...</think> blocks
      - Markdown code fences
      - Inline comments in arrays (LLM hallucination)
      - Truncated JSON (output cut off mid-generation)
      - Trailing commas
    """
    text = text.strip()

    # Strip qwen3 thinking blocks
    text = re.sub(r'<think>[\s\S]*?</think>', '', text).strip()
    text = re.sub(r'Thinking Process:[\s\S]*?(?=\{)', '', text).strip()

    # Strip markdown code fences
    fence_match = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', text)
    if fence_match:
        text = fence_match.group(1).strip()

    # Extract the outermost { ... } block
    start = text.find('{')
    if start == -1:
        return None
    # Find matching closing brace
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

    # Try direct parse
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Try after repair
    repaired = repair_json(json_str)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    return None


def validate_output(data: dict) -> tuple[bool, str]:
    """Check that the LLM output has all required keys with correct types."""
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


def call_ollama(messages: list[dict]) -> str | None:
    """Call Ollama chat endpoint. Returns raw text or None on failure."""
    payload = {
        "model":    OLLAMA_MODEL,
        "messages": messages,
        "stream":   False,
        "think":    False,        # top-level → actually disables thinking for qwen3
        "options":  OLLAMA_OPTIONS,
    }
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]
    except requests.exceptions.ConnectionError:
        log.error("Cannot connect to Ollama. Is it running? (`ollama serve`)")
        return None
    except requests.exceptions.Timeout:
        log.warning("Ollama request timed out.")
        return None
    except Exception as e:
        log.warning(f"Ollama call failed: {e}")
        return None


def process_one(row: dict, use_system_role: bool = True) -> dict | None:
    """
    Call LLM for a single journal row.
    Returns parsed + validated dict, or None on all-retry failure.
    """
    journal    = row.get(COL_JOURNAL, "").strip()
    aims       = row.get(COL_AIMS, "").strip()
    categories = row.get(COL_CATEGORIES, "").strip()

    build_fn = build_prompt if use_system_role else build_prompt_no_system

    for attempt in range(1, MAX_RETRIES + 1):
        messages = build_fn(journal, categories, aims)
        raw = call_ollama(messages)

        if raw is None:
            log.warning(f"  Attempt {attempt}/{MAX_RETRIES}: No response for '{journal}'")
            time.sleep(RETRY_DELAY_SEC)
            continue

        parsed = extract_json_from_response(raw)
        if parsed is None:
            log.warning(f"  Attempt {attempt}/{MAX_RETRIES}: Could not parse JSON for '{journal}'")
            log.warning(f"  Raw output (first 600 chars):\n{raw[:600]}")  # DEBUG → WARNING tạm thời
            time.sleep(RETRY_DELAY_SEC)
            continue

        valid, reason = validate_output(parsed)
        if not valid:
            log.warning(f"  Attempt {attempt}/{MAX_RETRIES}: Validation failed ({reason}) for '{journal}'")
            time.sleep(RETRY_DELAY_SEC)
            continue

        return parsed

    return None  # All retries exhausted


def log_failed(row: dict, error: str, out_file):
    # Sao chép toàn bộ các trường ban đầu từ CSV để giữ lại cột Aims, Journal, v.v.
    record = dict(row)
    record["error"] = error
    out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    out_file.flush()


# ── Main pipeline ─────────────────────────────────────────────

def run():
    ensure_dirs()

    if not INPUT_CSV.exists():
        log.error(f"Input file not found: {INPUT_CSV}")
        sys.exit(1)

    # Load all rows
    with open(INPUT_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    total = len(rows)
    log.info(f"Loaded {total} journals from {INPUT_CSV}")

    # Determine resume point
    last_idx = load_checkpoint()
    start_idx = last_idx + 1
    already_processed = load_processed_ids()

    if start_idx > 0:
        log.info(f"Resuming from index {start_idx} (checkpoint: {last_idx})")
    else:
        log.info("Starting fresh extraction")

    stats = {"success": 0, "failed": 0, "skipped": 0}

    # Health check — verify Ollama is responding before starting
    log.info(f"Checking Ollama at {OLLAMA_BASE_URL} with model '{OLLAMA_MODEL}'...")
    try:
        ping = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={"model": OLLAMA_MODEL, "messages": [{"role":"user","content":"ping"}],
                  "stream": False, "think": False,
                  "options": {"num_predict": 5}},
            timeout=30,
        )
        ping.raise_for_status()
        log.info("  ✓ Ollama responding OK")
    except Exception as e:
        log.error(f"  ✗ Ollama health check failed: {e}")
        log.error("  Make sure Ollama is running: `ollama serve`")
        sys.exit(1)

    # Open output files in append mode
    with (
        open(OUTPUT_JSONL, "a", encoding="utf-8") as out_f,
        open(FAILED_LOG, "a", encoding="utf-8") as fail_f,
    ):
        pbar = tqdm(
            enumerate(rows),
            total=total,
            initial=start_idx,
            desc="Extracting",
            unit="journal",
        )

        for idx, row in pbar:
            # Skip already-done rows
            if idx < start_idx:
                continue

            journal_name = row.get(COL_JOURNAL, "").strip()

            # Skip if somehow already in output (e.g. re-run after partial batch)
            if journal_name in already_processed:
                log.debug(f"  Skipping already-processed: '{journal_name}'")
                stats["skipped"] += 1
                save_checkpoint(idx)
                continue

            # Skip rows with no journal name or no aims
            if not journal_name or not row.get(COL_AIMS, "").strip():
                log.warning(f"  Skipping row {idx} — empty journal name or aims")
                stats["skipped"] += 1
                save_checkpoint(idx)
                continue

            pbar.set_postfix({
                "journal": journal_name[:35] + "…" if len(journal_name) > 35 else journal_name,
                "ok": stats["success"],
                "fail": stats["failed"],
            })

            result = process_one(row)

            if result is not None:
                # Build full output record
                record = {
                    "_schema":    SCHEMA_VERSION,
                    "_idx":       idx,
                    "journal":    journal_name,
                    "label":      row.get(COL_LABEL, "").strip(),
                    "categories": row.get(COL_CATEGORIES, "").strip(),
                    **result,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                already_processed.add(journal_name)
                stats["success"] += 1
                log.debug(f"  ✓ Saved '{journal_name}'")
            else:
                log_failed(row, "All retries exhausted", fail_f)
                stats["failed"] += 1
                log.warning(f"  ✗ Failed '{journal_name}' — logged to failed.jsonl")

            # Save checkpoint after every record
            save_checkpoint(idx)

            # Batch flush log
            if (idx + 1) % BATCH_SIZE == 0:
                log.info(
                    f"Progress {idx+1}/{total} | "
                    f"✓ {stats['success']} | ✗ {stats['failed']} | skip {stats['skipped']}"
                )

            # Throttle
            if SLEEP_BETWEEN_MS > 0:
                time.sleep(SLEEP_BETWEEN_MS / 1000)

    # Final summary
    log.info("=" * 60)
    log.info(f"DONE. Total: {total} | ✓ Success: {stats['success']} | "
             f"✗ Failed: {stats['failed']} | Skipped: {stats['skipped']}")
    log.info(f"Output  → {OUTPUT_JSONL}")
    log.info(f"Failed  → {FAILED_LOG}")
    log.info("=" * 60)


# ── Retry-failed utility ──────────────────────────────────────

def retry_failed():
    """
    Re-process all rows in failed.jsonl.
    Useful after fixing prompts or when Ollama was temporarily down.
    """
    if not FAILED_LOG.exists():
        log.info("No failed.jsonl found.")
        return

    with open(FAILED_LOG, encoding="utf-8") as f:
        failed_rows = [json.loads(l) for l in f if l.strip()]

    if not failed_rows:
        log.info("failed.jsonl is empty.")
        return

    # Filter out rows with empty journal name or missing aims
    valid_rows   = [r for r in failed_rows if r.get("journal", "").strip()]
    invalid_rows = [r for r in failed_rows if not r.get("journal", "").strip()]
    if invalid_rows:
        log.warning(f"Skipping {len(invalid_rows)} rows with empty journal name")

    log.info(f"Retrying {len(valid_rows)} failed journals...")
    new_failures = []

    with open(OUTPUT_JSONL, "a", encoding="utf-8") as out_f:
        for row in tqdm(valid_rows, desc="Retrying"):
            result = process_one(row)
            if result is not None:
                record = {
                    "_schema":    SCHEMA_VERSION,
                    "_idx":       -1,   # unknown original index
                    "journal":    row.get(COL_JOURNAL, row.get("journal", "")),
                    "label":      row.get(COL_LABEL, row.get("label", "")),
                    "categories": row.get(COL_CATEGORIES, row.get("categories", "")),
                    **result,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                log.info(f"  ✓ Recovered: '{row.get('journal')}'")
            else:
                new_failures.append(row)

    # Overwrite failed log with remaining failures
    with open(FAILED_LOG, "w", encoding="utf-8") as f:
        for r in new_failures:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    log.info(f"Retry done. Recovered: {len(failed_rows) - len(new_failures)} | "
             f"Still failed: {len(new_failures)}")


# ── Entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Journal metadata extraction pipeline")
    parser.add_argument(
        "--mode",
        choices=["run", "retry"],
        default="run",
        help="'run' = full pipeline; 'retry' = re-process failed rows only",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override Ollama model (e.g. --model mistral)",
    )
    args = parser.parse_args()

    if args.model:
        import config
        config.OLLAMA_MODEL = args.model
        log.info(f"Model overridden to: {args.model}")

    if args.mode == "retry":
        retry_failed()
    else:
        run()