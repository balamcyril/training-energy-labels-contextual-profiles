"""Audit the archived ESEM 2023 label calculation without executing ML models.

Source: https://doi.org/10.5281/zenodo.8115187, notebook
code/Analysis/HFEmissionsAnalysis.ipynb, cells 41--42.
This is an independent standard-library reconstruction, not the original runtime.
It preserves the archived calculation, including its carbon-boundary transform.
"""

import argparse
import collections
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import tempfile
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", nargs="?", type=Path,
                        help="Optional complete Zenodo v3 archive; otherwise use the verified bundled table.")
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1] / "data/sources/gaissa_2023"
    provenance = json.loads((source / "provenance.json").read_text(encoding="utf-8"))
    digest = provenance["archive_md5"]
    if args.archive:
        if hashlib.md5(args.archive.read_bytes()).hexdigest() != digest:
            raise ValueError("Archive does not match the published v3 checksum")
        with zipfile.ZipFile(args.archive) as archive:
            content = archive.read("datasets/HFCO2.csv")
    else:
        content = (source / "HFCO2.csv").read_bytes()
    if hashlib.sha256(content).hexdigest() != provenance["members"]["HFCO2.csv"]["sha256"]:
        raise ValueError("Historical source table checksum mismatch")
    rows = list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))
    weights = {"co2_eq_emissions": .35, "size_efficency": .1,
               "datasets_size_efficency": .1, "downloads": .25, "performance_score": .2}

    def number(value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return math.nan

    reference = next(row for row in rows if row["modelId"] == "distilgpt2")
    refs = {metric: number(reference[metric]) for metric in weights}
    for metric in refs:
        if math.isnan(refs[metric]):
            refs[metric] = statistics.median(number(r[metric]) for r in rows
                                              if not math.isnan(number(r[metric])))

    def index(row, metric):
        value = number(row[metric])
        if math.isnan(value):
            return math.nan
        numerator, denominator = (refs[metric], value) if metric == "co2_eq_emissions" else (value, refs[metric])
        if denominator == 0:
            return math.inf if numerator > 0 else math.nan
        return numerator / denominator

    def percentile(values, percentage):
        ordered = sorted(v for v in values if not math.isnan(v))
        position = (len(ordered) - 1) * percentage / 100
        low, high = math.floor(position), math.ceil(position)
        return ordered[low] if low == high else ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    boundaries = {}
    for metric in weights:
        values = [index(row, metric) for row in rows]
        if metric == "co2_eq_emissions":
            values = [math.sqrt(value) for value in values]
        percentages = [96, 94, 85, 60] if metric == "downloads" else [80, 60, 40, 20]
        boundaries[metric] = [percentile(values, p) for p in percentages]

    def rating(value, metric):
        if math.isnan(value):
            return None
        edges = [math.inf] + boundaries[metric] + [0]
        for grade, (upper, lower) in enumerate(zip(edges, edges[1:])):
            if upper >= value > lower:
                return grade
        return 4

    counts = collections.Counter()
    examples = []
    for row in rows:
        grades = {m: rating(index(row, m), m) for m in weights}
        available = [m for m in weights if grades[m] is not None]
        total_weight = sum(weights[m] for m in available)
        # Normalize weights before multiplication, as in the archived function.
        score = sum(grades[m] * (weights[m] / total_weight) for m in available)
        label = "ABCDE"[round(score)]
        counts[label] += 1
        if row["modelId"] in {"distilgpt2", "bloom", "AI-image-detector", "BERT-Banking77"}:
            examples.append({"modelId": row["modelId"], "label": label})
    identifiers = collections.Counter(row["modelId"] for row in rows)
    expected = {"A": 37, "B": 486, "C": 826, "D": 55, "E": 15}
    print(json.dumps({"archive_doi": "10.5281/zenodo.8115187", "archive_md5": digest,
                      "input_mode": "complete_archive" if args.archive else "verified_extracted_table",
                      "source_table": "datasets/HFCO2.csv", "rows": len(rows),
                      "unique_modelId_strings": len(identifiers),
                      "duplicate_modelId_strings": {k: v for k, v in identifiers.items() if v > 1},
                      "reconstructed_counts": dict(sorted(counts.items())),
                      "archived_notebook_counts": expected, "counts_match": dict(counts) == expected,
                      "examples": examples}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
