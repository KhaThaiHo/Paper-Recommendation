# ============================================================
# config.py — Centralized configuration for journal extraction
# ============================================================

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
INPUT_CSV       = BASE_DIR / "data" / "journal_category.csv"        # Your input file
OUTPUT_JSONL    = BASE_DIR / "output" / "extracted.jsonl"   # One JSON per line (append-safe)
CHECKPOINT_FILE = BASE_DIR / "output" / "checkpoint.txt"    # Last processed index
FAILED_LOG      = BASE_DIR / "output" / "failed.jsonl"      # Rows that errored

# ── CSV column names ──────────────────────────────────────────
COL_JOURNAL    = "Journal"
COL_AIMS       = "Aims"
COL_LABEL      = "Label"
COL_CATEGORIES = "Categories"

# ── Transformers settings ────────────────────────────────────
MODEL_NAME             = "Qwen/Qwen3.5-4B"
MODEL_TRUST_REMOTE_CODE = True
MODEL_MAX_NEW_TOKENS    = 1500
MODEL_TEMPERATURE       = 0.1
MODEL_TOP_P             = 0.9
MODEL_REPETITION_PENALTY = 1.05
MODEL_DO_SAMPLE         = True
MODEL_DEVICE_MAP        = "auto"
MODEL_TORCH_DTYPE       = "auto"

# ── Pipeline settings ─────────────────────────────────────────
BATCH_SIZE       = 10     # Save checkpoint every N records
MAX_RETRIES      = 3      # Retry failed LLM calls
RETRY_DELAY_SEC  = 5      # Wait between retries
SLEEP_BETWEEN_MS = 100    # Throttle (ms) between calls — set 0 for max speed

# ── Output schema version (for future migrations) ─────────────
SCHEMA_VERSION = "1.0"