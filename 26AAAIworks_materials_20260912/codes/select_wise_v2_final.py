#!/usr/bin/env python3
"""Select the final 631 source-verified WISE additions and merge retained rows."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

from prepare_wise_fact_shortlist import BASE, build_shortlist, read
from wise_fact_progress import fingerprint


ROOT = Path(__file__).resolve().parents[1]
FACTS = BASE / "wise_facts"
OUTPUT = ROOT / "codes/wise/wise_generate_v2.json"
MANIFEST = BASE / "final/wise_v2_manifest.json"
RANK = {"pending": 0, "fact_verified": 1, "needs_revision": 2, "reject": 3}


def effective_reviews() -> dict:
    result = {}
    for name in ("auto_reviews_epmc.json", "auto_reviews_openalex.json", "auto_reviews_wikipedia.json", "auto_reviews_wikipedia_claims.json", "auto_reviews_web.json", "auto_reviews_web_claims.json", "auto_reviews_web_claims_all.json", "auto_reviews_web_pages.json", "auto_reviews_web_reader.json"):
        path = FACTS / name
        if not path.exists():
            continue
        for identifier, review in read(path).items():
            old = result.get(identifier)
            if old is None or RANK[review["status"]] > RANK[old["status"]]:
                result[identifier] = review
    manual = FACTS / "reviews.json"
    if manual.exists():
        result.update(read(manual))
    return result


def atomic_write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def build() -> tuple[list, dict]:
    quota_doc = read(BASE / "wise_quota.json")
    retained = read(BASE / "wise_retained_369.json")
    shortlist, excluded, shortlist_summary = build_shortlist()
    reviews = effective_reviews()

    retained_counts = collections.Counter((row["Category"], row["Subcategory"]) for row in retained)
    if len(retained) != quota_doc["kept_total"] or len({row["prompt_id"] for row in retained}) != len(retained):
        raise ValueError("retained WISE pool count or prompt_id uniqueness is invalid")

    selected = []
    deficits = []
    provenance = []
    for group in quota_doc["quota"]:
        key = (group["category"], group["subcategory"])
        if retained_counts[key] != group["kept"]:
            raise ValueError(f"retained count mismatch for {key}: {retained_counts[key]} != {group['kept']}")
        eligible = [
            row for row in shortlist
            if (row["Category"], row["Subcategory"]) == key
            and reviews.get(row["draft_id"], {}).get("status") == "fact_verified"
            and reviews[row["draft_id"]].get("content_sha256") == fingerprint(row)
        ]
        needed = group["new_required"]
        if len(eligible) < needed:
            deficits.append({
                "category": key[0], "subcategory": key[1],
                "required": needed, "fact_verified": len(eligible), "missing": needed - len(eligible),
            })
            continue
        for offset, row in enumerate(eligible[:needed]):
            prompt_id = group["new_id_start"] + offset
            if prompt_id > group["new_id_end"]:
                raise ValueError(f"new prompt_id exceeds range for {key}")
            final_row = {field: row[field] for field in ("Prompt", "Explanation", "Category", "Subcategory")}
            final_row["prompt_id"] = prompt_id
            selected.append(final_row)
            provenance.append({
                "prompt_id": prompt_id,
                "draft_id": row["draft_id"],
                "source_wave": row["source_wave"],
                "author": "gemini3flash" if "GEMINI3FLASH" in row["draft_id"] else "gpt55",
                "content_sha256": fingerprint(row),
                "fact_review": reviews[row["draft_id"]],
            })

    readiness = {
        "ready": not deficits,
        "screened_candidates": shortlist_summary["screened_candidates"],
        "shortlisted": shortlist_summary["queued"],
        "fact_verified_total": sum(review["status"] == "fact_verified" for review in reviews.values()),
        "selected_if_ready": len(selected) if not deficits else 0,
        "deficits": deficits,
    }
    if deficits:
        return [], readiness

    output = sorted(retained + selected, key=lambda row: row["prompt_id"])
    if len(output) != quota_doc["target_total"] or len({row["prompt_id"] for row in output}) != len(output):
        raise ValueError("final WISE count or prompt_id uniqueness is invalid")
    final_counts = collections.Counter((row["Category"], row["Subcategory"]) for row in output)
    for group in quota_doc["quota"]:
        key = (group["category"], group["subcategory"])
        if final_counts[key] != group["target"]:
            raise ValueError(f"final count mismatch for {key}")
    manifest = {
        "benchmark": quota_doc["benchmark"],
        "summary": {
            "total": len(output), "retained": len(retained), "new": len(selected),
            "new_by_author": dict(collections.Counter(row["author"] for row in provenance)),
        },
        "inputs": {
            "retained": str(BASE / "wise_retained_369.json"),
            "quota": str(BASE / "wise_quota.json"),
            "retained_sha256": hashlib.sha256((BASE / "wise_retained_369.json").read_bytes()).hexdigest(),
            "quota_sha256": hashlib.sha256((BASE / "wise_quota.json").read_bytes()).hexdigest(),
        },
        "selection_policy": "Exact subgroup quotas; only content-hash-matched fact_verified rows from the semantic-screened shortlist.",
        "new_rows": provenance,
    }
    return output, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status-only", action="store_true")
    args = parser.parse_args()
    output, report = build()
    if not output:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not args.status_only:
            raise SystemExit("WISE final selection blocked by per-subcategory fact-verification deficits")
        return
    if args.status_only:
        print(json.dumps({"ready": True, "total": len(output), "new": len(report["new_rows"])}, ensure_ascii=False))
        return
    atomic_write(OUTPUT, output)
    atomic_write(MANIFEST, report)
    print(json.dumps(report["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
