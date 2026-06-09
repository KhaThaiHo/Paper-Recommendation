# Journal Metadata Extraction Pipeline

Trích xuất `scientific_domains` và `research_focuses` từ 1406 journals bằng Transformers chạy local.

## Cấu trúc project

```architecture
journal_extractor/
├── config.py           # Cấu hình tập trung (model, paths, batch size…)
├── prompt_builder.py   # Prompt engineering + few-shot examples
├── extractor.py        # Pipeline chính (checkpoint, retry, JSONL append)
├── inspect_output.py   # Inspect & validate kết quả
├── debug_transformers.py # Smoke test cho model Transformers
├── requirements.txt
├── data/
│   └── journals.csv    # ← đặt file input của bạn vào đây
└── output/
    ├── extracted.jsonl  # Kết quả (append-only, 1 JSON mỗi dòng)
    ├── checkpoint.txt   # Index dòng cuối đã xử lý
    ├── failed.jsonl     # Các dòng bị lỗi
    └── pipeline.log     # Log chi tiết
```

## Setup

```bash
pip install -r requirements.txt

# Lần chạy đầu sẽ tự tải model từ Hugging Face Hub
# Nếu model bị gated/private thì đăng nhập trước:
# huggingface-cli login
```

## Chạy pipeline

```bash
# Đặt file CSV vào data/journals.csv (cột: Journal, Aims, Label, Categories)

# Chạy lần đầu (hoặc resume nếu đã chạy trước)
python extractor.py

# Dùng model khác trên Hugging Face Hub
python extractor.py --model Qwen/Qwen3.5-4B-Instruct

# Retry các dòng failed
python extractor.py --mode retry

# Smoke test riêng cho model
python debug_transformers.py
```

## Inspect kết quả

```bash
python inspect_output.py --stats       # Thống kê tổng quan
python inspect_output.py --sample 5    # Xem 5 records ngẫu nhiên
python inspect_output.py --validate    # Kiểm tra lỗi schema
python inspect_output.py --to-csv      # Export ra CSV phẳng
```

## Output format (mỗi dòng trong extracted.jsonl)

```json
{
  "_schema": "1.0",
  "_idx": 0,
  "journal": "Therapeutic Advances in Neurological Disorders",
  "label": "1340",
  "categories": "Neurology (clinical), Neurology, Pharmacology",
  "scientific_domains": ["Neurology", "Clinical Neurology", "Pharmacology"],
  "scientific_domains_evidence": {
    "Neurology": ["neurological conditions", "clinicians and researchers in neurology"],
    "Clinical Neurology": ["medical treatment of neurological conditions", "strong clinical focus"],
    "Pharmacology": ["pharmacological focus", "innovative studies"]
  },
  "research_focuses": [
    "Pharmacotherapy for Neurological Disorders",
    "Clinical Neurology Treatment",
    "Neurological Disease Management",
    "Translational Neuroscience"
  ],
  "research_focuses_evidence": {
    "Pharmacotherapy for Neurological Disorders": ["pharmacological focus", "medical treatment of neurological conditions"],
    "Clinical Neurology Treatment": ["strong clinical and pharmacological focus"],
    "Neurological Disease Management": ["innovative studies in the medical treatment"],
    "Translational Neuroscience": ["pioneering efforts", "recent research and perspectives"]
  }
}
```

## Checkpoint & Resume

Pipeline lưu checkpoint sau MỖI record. Nếu bị tắt giữa chừng, chỉ cần chạy lại:

```bash
python extractor.py   # Tự động tiếp tục từ chỗ dừng
```

## Tuning cho 1 triệu papers (tương lai)

Trong `config.py`, điều chỉnh:

- `SLEEP_BETWEEN_MS = 0`    — tắt throttle nếu máy chạy model đủ nhanh
- `BATCH_SIZE = 100`        — log ít hơn
- `MODEL_NAME = "..."`      — đổi sang model nhỏ hơn hoặc hợp domain hơn
- `MODEL_MAX_NEW_TOKENS = 512`  — cắt output ngắn hơn nếu papers đơn giản hơn.
