#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from typing import Any

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


def collect_strings(record: dict, field_names: tuple[str, ...]) -> list[str]:
    values: list[str] = []
    for field_name in field_names:
        value = record.get(field_name, [])
        if isinstance(value, list):
            values.extend(str(item).strip() for item in value if str(item).strip())
        elif isinstance(value, str) and value.strip():
            values.append(value.strip())
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        normalized = normalize_text(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_values.append(value)
    return unique_values


def collect_evidence(record: dict) -> list[str]:
    return collect_strings(record, ("scientific_domains_evidence", "research_focuses_evidence"))


def flatten_evidence(evidence_data: dict | list | str | None) -> list[str]:
    if not evidence_data:
        return []

    evidence_items: list[str] = []
    if isinstance(evidence_data, dict):
        for value in evidence_data.values():
            evidence_items.extend(flatten_evidence(value))
    elif isinstance(evidence_data, list):
        for item in evidence_data:
            evidence_items.extend(flatten_evidence(item))
    elif isinstance(evidence_data, str) and evidence_data.strip():
        evidence_items.append(evidence_data.strip())
    return evidence_items


def fuzzy_ratio(left: str, right: str) -> float:
    left_norm = normalize_text(left)
    right_norm = normalize_text(right)
    if not left_norm or not right_norm:
        return 0.0
    if fuzz is not None:
        return fuzz.ratio(left_norm, right_norm) / 100.0
    return SequenceMatcher(None, left_norm, right_norm).ratio()


def partial_fuzzy_ratio(needle: str, haystack: str) -> float:
    needle_norm = normalize_text(needle)
    haystack_norm = normalize_text(haystack)
    if not needle_norm or not haystack_norm:
        return 0.0
    if len(needle_norm) > len(haystack_norm):
        needle_norm, haystack_norm = haystack_norm, needle_norm
    if fuzz is not None:
        return fuzz.partial_ratio(needle_norm, haystack_norm) / 100.0

    window = len(needle_norm)
    if window == 0:
        return 0.0
    if window >= len(haystack_norm):
        return SequenceMatcher(None, needle_norm, haystack_norm).ratio()

    best = 0.0
    for start in range(0, len(haystack_norm) - window + 1):
        candidate = haystack_norm[start:start + window]
        score = SequenceMatcher(None, needle_norm, candidate).ratio()
        if score > best:
            best = score
            if best >= 1.0:
                break
    return best


def score_evidence_against_source(evidence_items: list[str], source_text: str, threshold: float) -> float:
    if not evidence_items or not source_text.strip():
        return 0.0

    hits: list[float] = []
    for evidence in evidence_items:
        match_score = partial_fuzzy_ratio(evidence, source_text)
        hits.append(1.0 if match_score >= threshold else 0.0)
    return mean(hits) if hits else 0.0


def build_text(labels: list[str]) -> str:
    return ". ".join(labels)


def cosine_similarity(model: Any, left: str, right: str) -> float:
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
    parser = argparse.ArgumentParser(description="Score extraction output with evidence-grounded EG and SPECTER2 similarity")
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
        aims_text = str(reference.get("Aims") or "")

        scientific_domains = collect_strings(record, ("scientific_domains",))
        research_focuses = collect_strings(record, ("research_focuses",))

        scientific_domains_evidence = flatten_evidence(record.get("scientific_domains_evidence"))
        research_focuses_evidence = flatten_evidence(record.get("research_focuses_evidence"))
        all_evidence = scientific_domains_evidence + research_focuses_evidence

        scientific_domains_eg_score = score_evidence_against_source(scientific_domains_evidence, aims_text, args.threshold)
        research_focuses_eg_score = score_evidence_against_source(research_focuses_evidence, aims_text, args.threshold)
        eg_score = score_evidence_against_source(all_evidence, aims_text, args.threshold)

        scientific_domains_specter2_similarity = cosine_similarity(
            model,
            build_text(scientific_domains_evidence),
            aims_text,
        )
        research_focuses_specter2_similarity = cosine_similarity(
            model,
            build_text(research_focuses_evidence),
            aims_text,
        )
        specter2_similarity = mean([
            scientific_domains_specter2_similarity,
            research_focuses_specter2_similarity,
        ])

        scored_record = dict(record)
        scored_record.update({
            "categories": categories_text,
            "aims": aims_text,
            "scientific_domains_eg_score": scientific_domains_eg_score,
            "research_focuses_eg_score": research_focuses_eg_score,
            "eg_score": eg_score,
            "scientific_domains_specter2_similarity": scientific_domains_specter2_similarity,
            "research_focuses_specter2_similarity": research_focuses_specter2_similarity,
            "specter2_similarity": specter2_similarity,
            "scientific_domains": scientific_domains,
            "research_focuses": research_focuses,
            "scientific_domains_evidence": scientific_domains_evidence,
            "research_focuses_evidence": research_focuses_evidence,
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