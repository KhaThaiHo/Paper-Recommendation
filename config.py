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

# ── Ollama settings ───────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL    = "qwen3.5:4b"        # ← updated
OLLAMA_TIMEOUT  = 120                 # seconds per request
OLLAMA_OPTIONS  = {
    "temperature": 0.1,               # Low temp → consistent structured output
    "num_predict": 2048,              # Increased — JSON output can be ~800-1200 tokens
}
# NOTE: think=False is passed at TOP-LEVEL payload in call_ollama(), not here
# Putting it in options silently fails on some Ollama versions

# ── Pipeline settings ─────────────────────────────────────────
BATCH_SIZE       = 10     # Save checkpoint every N records
MAX_RETRIES      = 3      # Retry failed LLM calls
RETRY_DELAY_SEC  = 5      # Wait between retries
SLEEP_BETWEEN_MS = 200    # Throttle (ms) between calls — set 0 for max speed

# ── Output schema version (for future migrations) ─────────────
SCHEMA_VERSION = "1.0"