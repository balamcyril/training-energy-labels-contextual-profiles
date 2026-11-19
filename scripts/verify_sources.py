"""Verify the archived Zeus and Ember inputs without downloading or executing them."""
from pathlib import Path
import csv
import gzip
import hashlib
import io
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    zeus = ROOT / "data/sources/zeus"
    original = json.loads((ROOT / "provenance/original_manifest.json").read_text(encoding="utf-8"))
    for name, expected in original["source_sha256"].items():
        actual = hashlib.sha256((zeus / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Zeus source checksum mismatch: {name}")

    carbon = ROOT / "data/sources/carbon"
    metadata = json.loads((carbon / "metadata.json").read_text(encoding="utf-8"))
    for record in metadata["files"]:
        actual = hashlib.sha256((carbon / record["file"]).read_bytes()).hexdigest()
        if actual != record["sha256"]:
            raise ValueError(f"Ember source checksum mismatch: {record['file']}")
    content = gzip.decompress((carbon / "ember_generation_yearly_global.csv.gz").read_bytes())
    if hashlib.sha256(content).hexdigest() != metadata["source_sha256"]:
        raise ValueError("Decompressed Ember archive checksum mismatch")
    records = list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))
    with (carbon / "candidate_factors.csv").open(encoding="utf-8-sig", newline="") as f:
        factors = list(csv.DictReader(f))
    for factor in factors:
        matches = [r for r in records if r["Area"] == factor["Area"]
                   and r["Year"] == factor["Year"]
                   and r["Electricity source"] == factor["Electricity source"]]
        if len(matches) != 1 or matches[0] != factor:
            raise ValueError(f"Archived factor is not an exact source row: {factor['Area']}")
    historical = ROOT / "data/sources/gaissa_2023"
    provenance = json.loads((historical / "provenance.json").read_text(encoding="utf-8"))
    for name, metadata in provenance["members"].items():
        if hashlib.sha256((historical / name).read_bytes()).hexdigest() != metadata["sha256"]:
            raise ValueError(f"Historical source checksum mismatch: {name}")
    print("Verified Zeus, Ember and historical source hashes, including the three country factors.")


if __name__ == "__main__":
    main()
