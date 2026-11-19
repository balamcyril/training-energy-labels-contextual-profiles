"""Inspect pinned public QA training logs without executing any model.

Requires TensorBoard (not TensorFlow). Pass --dependencies for an isolated
installation. A log's wall-clock span is NOT treated as training runtime.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import sys
import urllib.request


SOURCES = [
    ("it5-small-question-answering", "6acbb88d78c77ebeceb0fbfc0453ddcff1814296",
     "events.out.tfevents.1642301432.t1v-n-9e38d79e-w-0.1339296.0.v2"),
    ("it5-base-question-answering", "42f9cc104612a173c5ffb335c2c77b9a94c32dc7",
     "events.out.tfevents.1642298516.t1v-n-9e38d79e-w-0.1334796.0.v2"),
    ("it5-large-question-answering", "1a9102e557a469141fa9fee356b99e76553564de",
     "events.out.tfevents.1642302618.t1v-n-9e38d79e-w-0.1341939.0.v2"),
    ("mt5-small-question-answering", "2db028bf1d12026e25024cf42e777ce509534e4b",
     "events.out.tfevents.1642648268.t1v-n-9e38d79e-w-0.1764917.0.v2"),
    ("mt5-base-question-answering", "2220b747c65156a3569f940cf6249441b93ca4ad",
     "events.out.tfevents.1642650166.t1v-n-9e38d79e-w-0.1768311.0.v2"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dependencies", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "logs")
    args = parser.parse_args()
    if args.dependencies:
        sys.path.insert(0, str(args.dependencies.resolve()))
    from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
    from tensorboard.util import tensor_util

    args.output.mkdir(parents=True, exist_ok=True)
    findings = []
    for model, revision, filename in SOURCES:
        url = f"https://huggingface.co/gsarti/{model}/resolve/{revision}/{filename}"
        local = args.output / model / filename
        local.parent.mkdir(parents=True, exist_ok=True)
        if not local.exists():
            request = urllib.request.Request(url, headers={"User-Agent": "GAISSA-research-audit"})
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read(20_000_001)
            if len(raw) > 20_000_000:
                raise ValueError(f"Unexpectedly large event file: {model}")
            local.write_bytes(raw)
        raw = local.read_bytes()
        tags = {}
        event_count = 0
        earliest = latest = None
        for event in EventFileLoader(str(local)).Load():
            event_count += 1
            wall = float(event.wall_time)
            if not math.isfinite(wall):
                raise ValueError(f"Non-finite timestamp in {model}")
            earliest = wall if earliest is None else min(earliest, wall)
            latest = wall if latest is None else max(latest, wall)
            for value in event.summary.value:
                scalar = None
                if value.HasField("simple_value"):
                    scalar = float(value.simple_value)
                elif value.HasField("tensor"):
                    array = tensor_util.make_ndarray(value.tensor)
                    if array.size == 1 and array.dtype.kind in "biuf":
                        scalar = float(array.reshape(-1)[0])
                if scalar is not None and not math.isfinite(scalar):
                    scalar = None
                item = {"step": int(event.step), "wall_time": wall, "scalar": scalar}
                if value.tag not in tags:
                    tags[value.tag] = {"count": 0, "first": item}
                tags[value.tag]["count"] += 1
                tags[value.tag]["last"] = item
        if event_count == 0:
            raise ValueError(f"No events decoded in {model}")
        energy_tags = [tag for tag in tags if re.search(r"energy|power|carbon|emission|kwh|watt|joule", tag, re.I)]
        time_tags = [tag for tag in tags if re.search(r"runtime|duration|time", tag, re.I)]
        finding = {
            "model_id": "gsarti/" + model, "revision": revision,
            "source_url": url, "local_file": str(local.relative_to(args.output)),
            "file_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            "decoded_events": event_count,
            "observed_log_span_seconds": latest - earliest,
            "log_span_is_training_runtime": False,
            "energy_related_tags": energy_tags, "time_related_tags": time_tags,
            "tags": tags,
        }
        findings.append(finding)
        print(json.dumps({k: finding[k] for k in ["model_id", "file_bytes", "decoded_events",
                         "observed_log_span_seconds", "energy_related_tags", "time_related_tags"]},
                         allow_nan=False))
        print(json.dumps({"all_tags": list(tags), "time_values": {tag: tags[tag] for tag in time_tags}},
                         allow_nan=False))
    report = {
        "scope": "Pinned historical SQuAD-IT QA logs; no model execution or scenario simulation",
        "reader": {"tensorboard": importlib.metadata.version("tensorboard"),
                   "python": sys.version.split()[0]},
        "interpretation": "Wall-clock spans describe recorded events, not validated training time. "
                          "A tag-name search does not establish the completeness of physical measurements.",
        "models": findings,
    }
    (args.output / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
