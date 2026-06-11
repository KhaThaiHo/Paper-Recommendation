# Paper Metadata Extraction Pipeline

Extracts `scientific_domains` and `research_focuses` from paper/journal CSV files using a local Transformer model (default: `Qwen/Qwen3.5-2B`). Designed to scale to 1M+ papers with checkpoint/resume, batched GPU inference, and failed-row retry.

## Project Structure

```text
paper_recommendation/
├── config.py                # Centralized config (model, paths, batch sizes)
├── extractor.py             # Journal extraction pipeline
├── extractor_paper.py       # Paper extraction pipeline  ← main script
├── prompt_builder.py        # Prompt engineering for journals
├── prompt_builder_paper.py  # Prompt engineering for papers
├── inspect_output.py        # Inspect & validate output JSONL
├── score_output.py          # Scoring utilities
├── debug_transformers.py    # Model smoke test
├── check_model.py           # Model check utility
├── requirements.txt
├── data/
│   └── test_set.csv         # Input CSV (Title, Abstract, Keywords, Label)
└── output/
    ├── extracted_papers.jsonl   # Results (append-only, 1 JSON per line)
    ├── checkpoint_papers.txt    # Last processed row index
    └── failed_papers.jsonl      # Rows that failed after all retries
```

## Setup

```bash
pip install -r requirements.txt

# First run will auto-download the model from Hugging Face Hub.
# If the model is gated/private, log in first:
huggingface-cli login
```

## Running Locally

```bash
# Default: reads data/test_set.csv, writes output/extracted_papers.jsonl
python extractor_paper.py

# Use a different model
python extractor_paper.py --model Qwen/Qwen3.5-4B-Instruct

# Increase batch size for better GPU utilisation
python extractor_paper.py --prompt-batch-size 32 --max-new-tokens 500

# Retry failed rows only
python extractor_paper.py --mode retry
```

## Running on Kaggle

Kaggle uses fixed paths (`/kaggle/input/` for datasets, `/kaggle/working/` for output). All paths and settings can be overridden via CLI args — no need to edit `config.py`.

### Step 1 — Upload the scripts as a Kaggle dataset

Upload all `.py` files from this repo as a private Kaggle dataset (e.g. named `paper-extractor`).

### Step 2 — Notebook cells

```python
# ── Cell 1: Install dependencies ──────────────────────────────
!pip install -q transformers torch tqdm accelerate

# ── Cell 2: Copy scripts to working dir (fixes relative imports) ──
import shutil, os
src = "/kaggle/input/paper-extractor/"
for f in os.listdir(src):
    if f.endswith(".py"):
        shutil.copy(os.path.join(src, f), "/kaggle/working/")

# ── Cell 3: Run extraction ─────────────────────────────────────
!cd /kaggle/working && python extractor_paper.py \
    --input      /kaggle/input/<your-data-dataset>/papers.csv \
    --output     /kaggle/working/extracted_papers.jsonl \
    --checkpoint /kaggle/working/checkpoint_papers.txt \
    --failed     /kaggle/working/failed_papers.jsonl \
    --model      Qwen/Qwen3.5-2B \
    --prompt-batch-size 32 \
    --max-new-tokens 500
```

If the session is interrupted (12-hour Kaggle limit), re-run Cell 3 — the pipeline resumes automatically from the checkpoint. Save `extracted_papers.jsonl` and `checkpoint_papers.txt` to a Kaggle dataset between sessions so they persist.

### Kaggle GPU performance

| GPU        | VRAM        | Speed (batch=8) | Recommended batch | 1M papers @ 30h/wk |
| ---------- | ----------- | --------------- | ----------------- | ------------------ |
| Tesla P100 | 16 GB HBM2  | ~1.5 s/paper    | 32–64             | ~14 weeks          |
| Tesla T4   | 16 GB GDDR6 | ~3 s/paper      | 16–32             | ~28 weeks          |

Larger `--prompt-batch-size` improves GPU utilisation. Start at 32 on P100; reduce if you hit OOM (the pipeline falls back to per-item inference automatically).

## All CLI Arguments — `extractor_paper.py`

| Argument             | Default                            | Description                                       |
| -------------------- | ---------------------------------- | ------------------------------------------------- |
| `--mode`             | `run`                              | `run` = full pipeline; `retry` = re-process fails |
| `--input`            | `data/test_set.csv`                | Input CSV path                                    |
| `--output`           | `output/extracted_papers.jsonl`    | Output JSONL path                                 |
| `--checkpoint`       | `output/checkpoint_papers.txt`     | Checkpoint file path                              |
| `--failed`           | `output/failed_papers.jsonl`       | Failed log path                                   |
| `--model`            | `Qwen/Qwen3.5-2B`                  | HuggingFace model ID                              |
| `--prompt-batch-size`| `8`                                | Papers per GPU batch call                         |
| `--max-new-tokens`   | `700`                              | Max output tokens per generation                  |
| `--batch-size`       | `100`                              | Checkpoint/log interval (records)                 |

## Input CSV Format

| Column     | Required | Description                               |
| ---------- | -------- | ----------------------------------------- |
| `Title`    | Yes      | Paper title                               |
| `Abstract` | Yes      | Paper abstract                            |
| `Keywords` | No       | Semicolon-separated keywords              |
| `Label`    | No       | Category label (passed through to output) |

## Output Format

Each line in the output JSONL is one paper:

```json
{
  "_schema": "1.0",
  "_idx": 0,
  "title": "Deep Learning for Medical Image Segmentation",
  "label": "cs.CV",
  "keywords": "Deep Learning; Medical Imaging; Segmentation",
  "abstract": "We propose a convolutional neural network...",
  "scientific_domains": ["Medical Imaging", "Machine Learning"],
  "scientific_domains_evidence": {
    "Medical Imaging": ["medical images", "clinical applications in radiology"],
    "Machine Learning": ["convolutional neural network", "deep learning"]
  },
  "research_focuses": [
    "Automated Medical Image Segmentation",
    "CNN Architectures for Segmentation",
    "Lesion Detection and Segmentation"
  ],
  "research_focuses_evidence": {
    "Automated Medical Image Segmentation": ["automated segmentation of medical images"],
    "CNN Architectures for Segmentation": ["convolutional neural network architecture"],
    "Lesion Detection and Segmentation": ["lesion segmentation"]
  }
}
```

## Checkpoint & Resume

The pipeline saves a checkpoint after **every record**. To resume after any interruption:

```bash
# Locally — just re-run the same command
python extractor_paper.py

# On Kaggle — re-run Cell 3 with the same --checkpoint path
```

Processed titles are also tracked in the output JSONL, so duplicate records are never written even if the checkpoint file is lost.

## Inspecting Results

```bash
python inspect_output.py --stats       # Summary statistics
python inspect_output.py --sample 5    # View 5 random records
python inspect_output.py --validate    # Check schema errors
python inspect_output.py --to-csv      # Export to flat CSV
```

## Key Tuning Parameters

Edit `config.py` or pass as CLI args:

| Parameter              | Default           | Tip                                          |
| ---------------------- | ----------------- | -------------------------------------------- |
| `PROMPT_BATCH_SIZE`    | `8`               | Increase to 32-64 on 16 GB GPU               |
| `MODEL_MAX_NEW_TOKENS` | `700`             | Reduce to 400-500 to cut generation time     |
| `USE_TORCH_COMPILE`    | `False`           | Set `True` for ~15% speedup (60-120s warmup) |
| `MODEL_NAME`           | `Qwen/Qwen3.5-2B` | Swap for a larger model if quality is low    |
