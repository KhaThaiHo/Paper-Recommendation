#!/usr/bin/env python3
# ============================================================
# inspect_output.py — Utilities to validate & explore results
# ============================================================
#
# Usage:
#   python inspect_output.py --stats          # summary statistics
#   python inspect_output.py --sample 5       # show 5 random records
#   python inspect_output.py --to-csv         # export flat CSV for review
#   python inspect_output.py --validate       # check all records for issues
# ============================================================

import json
import csv
import random
import argparse
from pathlib import Path
from collections import Counter

from config import OUTPUT_JSONL, FAILED_LOG


def load_records() -> list[dict]:
    records = []
    if not OUTPUT_JSONL.exists():
        print("No output file found yet.")
        return records
    with open(OUTPUT_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"  Malformed line: {e}")
    return records


def cmd_stats(records: list[dict]):
    print(f"\n{'='*50}")
    print(f"  EXTRACTION STATISTICS")
    print(f"{'='*50}")
    print(f"  Total records:     {len(records)}")

    all_domains = []
    all_focuses = []
    domain_counts = []
    focus_counts = []

    for r in records:
        sd = r.get("scientific_domains", [])
        rf = r.get("research_focuses", [])
        all_domains.extend(sd)
        all_focuses.extend(rf)
        domain_counts.append(len(sd))
        focus_counts.append(len(rf))

    print(f"\n  Scientific Domains:")
    print(f"    Total unique labels:  {len(set(all_domains))}")
    print(f"    Avg per journal:      {sum(domain_counts)/len(domain_counts):.1f}")
    print(f"    Min / Max per journal:{min(domain_counts)} / {max(domain_counts)}")

    print(f"\n  Research Focuses:")
    print(f"    Total unique labels:  {len(set(all_focuses))}")
    print(f"    Avg per journal:      {sum(focus_counts)/len(focus_counts):.1f}")
    print(f"    Min / Max per journal:{min(focus_counts)} / {max(focus_counts)}")

    print(f"\n  Top 15 Scientific Domains:")
    for domain, cnt in Counter(all_domains).most_common(15):
        print(f"    {cnt:4d}x  {domain}")

    print(f"\n  Top 15 Research Focuses:")
    for focus, cnt in Counter(all_focuses).most_common(15):
        print(f"    {cnt:4d}x  {focus}")

    if FAILED_LOG.exists():
        failed = sum(1 for l in open(FAILED_LOG) if l.strip())
        print(f"\n  Failed (in failed.jsonl): {failed}")
    print()


def cmd_sample(records: list[dict], n: int):
    sample = random.sample(records, min(n, len(records)))
    for r in sample:
        print(f"\n{'─'*60}")
        print(f"  Journal:    {r['journal']}")
        print(f"  Categories: {r.get('categories','')}")
        print(f"  Domains:    {', '.join(r.get('scientific_domains',[]))}")
        print(f"  Focuses:    {', '.join(r.get('research_focuses',[]))}")
        print(f"\n  Domain evidence:")
        for domain, evs in r.get("scientific_domains_evidence", {}).items():
            print(f"    [{domain}] → {evs}")
        print(f"\n  Focus evidence:")
        for focus, evs in r.get("research_focuses_evidence", {}).items():
            print(f"    [{focus}] → {evs}")
    print()


def cmd_validate(records: list[dict]):
    issues = []
    for i, r in enumerate(records):
        journal = r.get("journal", f"row_{i}")

        sd  = r.get("scientific_domains", [])
        rf  = r.get("research_focuses", [])
        sde = r.get("scientific_domains_evidence", {})
        rfe = r.get("research_focuses_evidence", {})

        # Check evidence keys match domain/focus labels
        for d in sd:
            if d not in sde:
                issues.append(f"[{journal}] Domain '{d}' has no evidence entry")
        for f_ in rf:
            if f_ not in rfe:
                issues.append(f"[{journal}] Focus '{f_}' has no evidence entry")

        # Check evidence phrases are strings
        for d, evs in sde.items():
            if not isinstance(evs, list):
                issues.append(f"[{journal}] Domain evidence for '{d}' is not a list")
            else:
                for ev in evs:
                    if not isinstance(ev, str):
                        issues.append(f"[{journal}] Evidence item not a string: {ev}")

        if len(sd) == 0:
            issues.append(f"[{journal}] Empty scientific_domains")
        if len(rf) == 0:
            issues.append(f"[{journal}] Empty research_focuses")

    if issues:
        print(f"\n  Found {len(issues)} validation issues:\n")
        for iss in issues[:50]:  # cap at 50
            print(f"  ⚠  {iss}")
        if len(issues) > 50:
            print(f"  ... and {len(issues)-50} more")
    else:
        print(f"\n  ✓ All {len(records)} records passed validation!\n")


def cmd_to_csv(records: list[dict]):
    out_path = Path("output/extracted_flat.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "journal", "label", "categories",
            "scientific_domains", "research_focuses",
            "scientific_domains_evidence", "research_focuses_evidence",
        ])
        for r in records:
            writer.writerow([
                r.get("journal", ""),
                r.get("label", ""),
                r.get("categories", ""),
                " | ".join(r.get("scientific_domains", [])),
                " | ".join(r.get("research_focuses", [])),
                json.dumps(r.get("scientific_domains_evidence", {}), ensure_ascii=False),
                json.dumps(r.get("research_focuses_evidence", {}), ensure_ascii=False),
            ])
    print(f"\n  Exported {len(records)} records → {out_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect extraction output")
    parser.add_argument("--stats",    action="store_true", help="Print summary statistics")
    parser.add_argument("--sample",   type=int, metavar="N", help="Show N random records")
    parser.add_argument("--validate", action="store_true", help="Validate all records")
    parser.add_argument("--to-csv",   action="store_true", help="Export flat CSV")
    args = parser.parse_args()

    records = load_records()
    if not records:
        print("No records loaded. Run extractor.py first.")
        exit(0)

    if args.stats:
        cmd_stats(records)
    if args.sample:
        cmd_sample(records, args.sample)
    if args.validate:
        cmd_validate(records)
    if args.to_csv:
        cmd_to_csv(records)
    if not any([args.stats, args.sample, args.validate, args.to_csv]):
        cmd_stats(records)
        cmd_validate(records)