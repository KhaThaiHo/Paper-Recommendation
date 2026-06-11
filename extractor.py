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
import importlib.util
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    INPUT_CSV, OUTPUT_JSONL, CHECKPOINT_FILE, FAILED_LOG,
    COL_JOURNAL, COL_AIMS, COL_LABEL, COL_CATEGORIES,
    MODEL_NAME, MODEL_TRUST_REMOTE_CODE, MODEL_MAX_NEW_TOKENS,
    MODEL_TEMPERATURE, MODEL_TOP_P, MODEL_REPETITION_PENALTY,
    MODEL_DO_SAMPLE, MODEL_DEVICE_MAP, MODEL_TORCH_DTYPE,
    BATCH_SIZE, MAX_RETRIES, RETRY_DELAY_SEC, SLEEP_BETWEEN_MS,
    SCHEMA_VERSION, USE_TORCH_COMPILE, PROMPT_BATCH_SIZE,
)
from prompt_builder import build_prompt, build_prompt_no_system

# ── Logging setup ─────────────────────────────────────────────
Path("output").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

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

TOKENIZER = None
MODEL = None


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


def _load_transformers_model() -> tuple[object, object]:
    """Load the tokenizer and model once for the whole process."""
    global TOKENIZER, MODEL
    if TOKENIZER is not None and MODEL is not None:
        return TOKENIZER, MODEL

    log.info(f"Loading Transformers model '{MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=MODEL_TRUST_REMOTE_CODE,
    )

    model_kwargs: dict[str, object] = {
        "trust_remote_code": MODEL_TRUST_REMOTE_CODE,
        "low_cpu_mem_usage": True,
    }

    use_cuda = torch.cuda.is_available()
    has_accelerate = importlib.util.find_spec("accelerate") is not None

    if use_cuda:
        if has_accelerate and MODEL_DEVICE_MAP == "auto":
            model_kwargs["device_map"] = MODEL_DEVICE_MAP
            model_kwargs["dtype"] = torch.float16
            log.info("Using CUDA with device_map='auto' via accelerate")
        else:
            model_kwargs["dtype"] = torch.float16
            log.info("Using CUDA with manual model.to('cuda') placement")
    else:
        model_kwargs["dtype"] = torch.float32
        log.info("CUDA not available; using CPU")

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)

    if use_cuda and "device_map" not in model_kwargs:
        model = model.to("cuda")
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    # Left-padding is required for correct batched generation with causal LMs
    tokenizer.padding_side = "left"

    if USE_TORCH_COMPILE:
        try:
            model = torch.compile(model)
            log.info("torch.compile applied (first call will be slow, subsequent calls faster)")
        except Exception as e:
            log.warning(f"torch.compile failed, continuing without it: {e}")

    TOKENIZER = tokenizer
    MODEL = model
    return tokenizer, model


def _format_messages(messages: list[dict]) -> str:
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role", "user")).upper()
        content = message.get("content", "")
        parts.append(f"{role}: {content}")
    parts.append("ASSISTANT:")
    return "\n\n".join(parts)


def call_model(messages: list[dict]) -> str | None:
    """Call the local Transformers model. Returns raw text or None on failure."""
    try:
        tokenizer, model = _load_transformers_model()

        try:
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        except Exception:
            prompt = _format_messages(messages)
            inputs = tokenizer(prompt, return_tensors="pt")

        if hasattr(model, "device"):
            device = model.device
        else:
            device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        gen_kwargs: dict[str, object] = {
            "max_new_tokens": MODEL_MAX_NEW_TOKENS,
            "do_sample": MODEL_DO_SAMPLE,
            "repetition_penalty": MODEL_REPETITION_PENALTY,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if MODEL_DO_SAMPLE:
            gen_kwargs["temperature"] = MODEL_TEMPERATURE
            gen_kwargs["top_p"] = MODEL_TOP_P

        with torch.inference_mode():
            outputs = model.generate(**inputs, **gen_kwargs)

        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    except Exception as e:
        log.warning(f"Transformers call failed: {e}")
        return None


def call_model_batch(messages_list: list[list[dict]]) -> list[str | None]:
    """Run multiple prompts in a single model.generate() call for higher GPU throughput.

    Falls back to per-item call_model if batching fails (e.g. OOM).
    """
    if not messages_list:
        return []
    if len(messages_list) == 1:
        return [call_model(messages_list[0])]

    try:
        tokenizer, model = _load_transformers_model()

        prompts: list[str] = []
        for messages in messages_list:
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            except Exception:
                text = _format_messages(messages)
            prompts.append(text)

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        )

        if hasattr(model, "device"):
            device = model.device
        else:
            device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        input_length = inputs["input_ids"].shape[1]

        gen_kwargs: dict[str, object] = {
            "max_new_tokens": MODEL_MAX_NEW_TOKENS,
            "do_sample": MODEL_DO_SAMPLE,
            "repetition_penalty": MODEL_REPETITION_PENALTY,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if MODEL_DO_SAMPLE:
            gen_kwargs["temperature"] = MODEL_TEMPERATURE
            gen_kwargs["top_p"] = MODEL_TOP_P

        with torch.inference_mode():
            outputs = model.generate(**inputs, **gen_kwargs)

        return [
            tokenizer.decode(output[input_length:], skip_special_tokens=True).strip()
            for output in outputs
        ]

    except torch.cuda.OutOfMemoryError:
        log.warning("OOM during batch generation — falling back to per-item inference")
        return [call_model(m) for m in messages_list]
    except Exception as e:
        log.warning(f"Batch model call failed ({e}) — falling back to per-item inference")
        return [call_model(m) for m in messages_list]


def health_check() -> bool:
    """Load the model and run a tiny generation as a startup check."""
    log.info(f"Checking Transformers model '{MODEL_NAME}'...")
    ping_messages = [{"role": "user", "content": "Reply with exactly {\"hello\": \"world\"}."}]
    response = call_model(ping_messages)
    if response is None:
        log.error("  ✗ Transformers model health check failed")
        return False

    log.info("  ✓ Transformers responding OK")
    return True


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
        raw = call_model(messages)

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

    # Health check — verify the local Transformers model is loadable before starting
    if not health_check():
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
    Useful after fixing prompts or when the model produced invalid output.
    """
    if not FAILED_LOG.exists():
        log.info("No failed.jsonl found.")
        return

    with open(FAILED_LOG, encoding="utf-8") as f:
        failed_rows = [json.loads(l) for l in f if l.strip()]

    if not failed_rows:
        log.info("failed.jsonl is empty.")
        return

    # SỬA Ở ĐÂY: Dùng COL_JOURNAL thay vì hardcode "journal"
    # Đồng thời dự phòng thêm fallback lấy key "journal" (nếu có)
    valid_rows   = [r for r in failed_rows if r.get(COL_JOURNAL, r.get("journal", "")).strip()]
    invalid_rows = [r for r in failed_rows if not r.get(COL_JOURNAL, r.get("journal", "")).strip()]
    
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
                log.info(f"  ✓ Recovered: '{row.get(COL_JOURNAL, row.get('journal', ''))}'")
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
        help="Override Transformers model id (e.g. --model Qwen/Qwen3.5-4B-Instruct)",
    )
    args = parser.parse_args()

    if args.model:
        import config
        config.MODEL_NAME = args.model
        log.info(f"Model overridden to: {args.model}")

    if args.mode == "retry":
        retry_failed()
    else:
        run()