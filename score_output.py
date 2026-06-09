#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean

from config import INPUT_CSV, OUTPUT_JSONL

try:
    from rapidfuzz import fuzz
except ImportError:
    fuzz = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def split_categories(value: str) -> list[str]:
    if not value:
        return []
    parts = re.split(r"\s*[;,|]\s*", value)
    return [part.strip() for part in parts if part and part.strip()]


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_reference_by_index(path: Path) -> dict[int, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Reference CSV not found: {path}")

    reference_rows: dict[int, dict] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            reference_rows[index] = row
    return reference_rows


def collect_labels(record: dict, field_names: tuple[str, ...]) -> list[str]:
    labels: list[str] = []
    for field_name in field_names:
        value = record.get(field_name, [])
        if isinstance(value, list):
            labels.extend(str(item).strip() for item in value if str(item).strip())
        elif isinstance(value, str) and value.strip():
            labels.append(value.strip())
    seen: set[str] = set()
    unique_labels: list[str] = []
    for label in labels:
        normalized = normalize_text(label)
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_labels.append(label)
    return unique_labels


def fuzzy_ratio(left: str, right: str) -> float:
    left_norm = normalize_text(left)
    right_norm = normalize_text(right)
    if not left_norm or not right_norm:
        return 0.0
    if fuzz is not None:
        return fuzz.ratio(left_norm, right_norm) / 100.0
    return SequenceMatcher(None, left_norm, right_norm).ratio()


def score_labels(predicted: list[str], gold: list[str], threshold: float) -> float:
    if not predicted or not gold:
        return 0.0
    hits: list[float] = []
    for predicted_label in predicted:
        best_match = max(fuzzy_ratio(predicted_label, gold_label) for gold_label in gold)
        hits.append(1.0 if best_match > threshold else 0.0)
    return mean(hits) if hits else 0.0


def build_text(labels: list[str]) -> str:
    return ". ".join(labels)


def cosine_similarity(model: SentenceTransformer, left: str, right: str) -> float:
    if not left.strip() or not right.strip():
        return 0.0
    embeddings = model.encode([left, right], normalize_embeddings=True, convert_to_numpy=True)
    return float(embeddings[0] @ embeddings[1])


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "_idx",
        "journal",
        "label",
        "categories",
        "scientific_domains",
        "research_focuses",
        "scientific_domains_eg_score",
        "research_focuses_eg_score",
        "eg_score",
        "scientific_domains_specter2_similarity",
        "research_focuses_specter2_similarity",
        "specter2_similarity",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({
                "_idx": record.get("_idx", ""),
                "journal": record.get("journal", ""),
                "label": record.get("label", ""),
                "categories": record.get("categories", ""),
                "scientific_domains": " | ".join(record.get("scientific_domains", [])),
                "research_focuses": " | ".join(record.get("research_focuses", [])),
                "scientific_domains_eg_score": f'{record.get("scientific_domains_eg_score", 0.0):.4f}',
                "research_focuses_eg_score": f'{record.get("research_focuses_eg_score", 0.0):.4f}',
                "eg_score": f'{record.get("eg_score", 0.0):.4f}',
                "scientific_domains_specter2_similarity": f'{record.get("scientific_domains_specter2_similarity", 0.0):.4f}',
                "research_focuses_specter2_similarity": f'{record.get("research_focuses_specter2_similarity", 0.0):.4f}',
                "specter2_similarity": f'{record.get("specter2_similarity", 0.0):.4f}',
            })


def main() -> int:
    parser = argparse.ArgumentParser(description="Score extraction output with fuzzy EG and SPECTER2 similarity")
    parser.add_argument("--predictions", type=Path, default=OUTPUT_JSONL, help="Path to the output JSONL file")
    parser.add_argument("--reference", type=Path, default=INPUT_CSV, help="Path to the gold CSV file")
    parser.add_argument("--jsonl-output", type=Path, default=Path("output/scored_output.jsonl"), help="Path to write scored JSONL")
    parser.add_argument("--csv-output", type=Path, default=Path("output/scored_output.csv"), help="Path to write scored CSV")
    parser.add_argument("--threshold", type=float, default=0.85, help="Fuzzy match threshold")
    parser.add_argument("--model", type=str, default="allenai/specter2_base", help="SentenceTransformer model name")
    args = parser.parse_args()

    if SentenceTransformer is None:
        raise SystemExit("sentence-transformers is required for SPECTER2 scoring. Install it with: pip install sentence-transformers")

    predictions = load_jsonl(args.predictions)
    reference_rows = load_reference_by_index(args.reference)
    model = SentenceTransformer(args.model)

    scored_records: list[dict] = []
    for record in predictions:
        idx = record.get("_idx")
        reference = reference_rows.get(idx, {}) if isinstance(idx, int) else {}
        categories_text = str(record.get("categories") or reference.get("Categories") or "")
        gold_labels = split_categories(categories_text)

        predicted_domains = collect_labels(record, ("scientific_domains",))
        predicted_focuses = collect_labels(record, ("research_focuses",))

        scientific_domains_eg_score = score_labels(predicted_domains, gold_labels, args.threshold)
        research_focuses_eg_score = score_labels(predicted_focuses, gold_labels, args.threshold)
        eg_values = [score for score in (scientific_domains_eg_score, research_focuses_eg_score)]
        eg_score = mean(eg_values) if eg_values else 0.0

        scientific_domains_specter2_similarity = cosine_similarity(
            model,
            build_text(predicted_domains),
            build_text(gold_labels),
        )
        research_focuses_specter2_similarity = cosine_similarity(
            model,
            build_text(predicted_focuses),
            build_text(gold_labels),
        )
        specter2_similarity = mean([
            scientific_domains_specter2_similarity,
            research_focuses_specter2_similarity,
        ])

        scored_record = dict(record)
        scored_record.update({
            "categories": categories_text,
            "gold_categories": gold_labels,
            "scientific_domains_eg_score": scientific_domains_eg_score,
            "research_focuses_eg_score": research_focuses_eg_score,
            "eg_score": eg_score,
            "scientific_domains_specter2_similarity": scientific_domains_specter2_similarity,
            "research_focuses_specter2_similarity": research_focuses_specter2_similarity,
            "specter2_similarity": specter2_similarity,
        })
        scored_records.append(scored_record)

    write_jsonl(args.jsonl_output, scored_records)
    write_csv(args.csv_output, scored_records)

    print(f"Scored records: {len(scored_records)}")
    print(f"Average EG score: {mean(record['eg_score'] for record in scored_records):.4f}")
    print(f"Average SPECTER2 similarity: {mean(record['specter2_similarity'] for record in scored_records):.4f}")
    print(f"JSONL output: {args.jsonl_output}")
    print(f"CSV output:   {args.csv_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())