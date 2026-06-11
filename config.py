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
MODEL_NAME             = "Qwen/Qwen3.5-2B"
MODEL_TRUST_REMOTE_CODE = True
MODEL_MAX_NEW_TOKENS    = 700   # JSON output rarely exceeds 600 tokens; saves ~2x gen time
MODEL_TEMPERATURE       = 0.1
MODEL_TOP_P             = 0.9
MODEL_REPETITION_PENALTY = 1.05
MODEL_DO_SAMPLE         = False  # Greedy at temp=0.1 is virtually identical, slightly faster
MODEL_DEVICE_MAP        = "auto"
MODEL_TORCH_DTYPE       = "auto"
USE_TORCH_COMPILE       = True  # Set True for ~20% speedup after one-time compile cost (~60-120s)

# ── Pipeline settings ─────────────────────────────────────────
BATCH_SIZE       = 100    # Save checkpoint every N records
MAX_RETRIES      = 3      # Retry failed LLM calls
RETRY_DELAY_SEC  = 0      # No delay needed for local model
SLEEP_BETWEEN_MS = 0      # Throttle (ms) between calls — set 0 for max speed

# ── Output schema version (for future migrations) ─────────────
SCHEMA_VERSION = "1.0"

# Number of prompts to send in one batched generate call
PROMPT_BATCH_SIZE = 16

# -- Paper extraction (separate from journal pipeline)
INPUT_PAPERS_CSV    = BASE_DIR / "data" / "test_set.csv"
OUTPUT_PAPERS_JSONL = BASE_DIR / "output" / "extracted_papers.jsonl"
CHECKPOINT_PAPERS   = BASE_DIR / "output" / "checkpoint_papers.txt"
FAILED_PAPERS       = BASE_DIR / "output" / "failed_papers.jsonl"

# CSV column names for paper file
COL_TITLE    = "Title"
COL_ABSTRACT = "Abstract"
COL_KEYWORDS = "Keywords"