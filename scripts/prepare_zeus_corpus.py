"""Prepare a secondary-analysis corpus from pinned Zeus NSDI 2023 traces.

No model execution, remote code execution, or third-party dependency.
Energy and total duration are reconstructed, not new measurements.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/sources/zeus"
OUT = ROOT / "data"
REVISION = "9604d1346215f820fb04983cb3df2468f9c873e4"
TARGET = 84.0


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (SOURCE / "summary_train.csv").open(encoding="utf-8") as f:
        train = [r for r in csv.DictReader(f) if r["dataset"] == "squad"
                 and r["network"] == "bert_base_uncased"
                 and float(r["target_metric"]) == TARGET]
    with (SOURCE / "summary_power_v100.csv").open(encoding="utf-8") as f:
        power = [r for r in csv.DictReader(f) if r["dataset"] == "squad"
                 and r["network"] == "bert_base_uncased"]
    key = lambda r: (r["dataset"], r["network"], r["batch_size"], r["optimizer"])
    assert len({(key(r), r["run"]) for r in train}) == len(train)
    assert len({(key(r), r["power_limit"]) for r in power}) == len(power)
    batches = sorted({int(r["batch_size"]) for r in power})
    limits = sorted({int(r["power_limit"]) for r in power})
    main_batches = {min(batches), 32, max(batches)}  # endpoints + paper baseline
    main_limits = {min(limits), max(limits)}
    rows, configurations = [], []
    for p in sorted(power, key=lambda r: (int(r["batch_size"]), int(r["power_limit"]))):
        group = sorted([r for r in train if key(r) == key(p)], key=lambda r: int(r["run"]))
        assert group and len({r["learning_rate"] for r in group}) == 1
        config = f"bert_b{p['batch_size']}_p{p['power_limit']}"
        valid = []
        for r in group:
            epochs = float(r["target_epoch"])
            if not math.isfinite(epochs):
                raise ValueError(f"Target not reached: {config}, run={r['run']}")
            t = epochs * float(p["time_per_epoch"])
            e = t * float(p["average_power"]) / 3_600_000
            assert t > 0 and e > 0
            item = dict(configuration=config, batch_size=int(p["batch_size"]),
                        power_limit_w=int(p["power_limit"]), run=int(r["run"]),
                        optimizer=r["optimizer"], learning_rate=float(r["learning_rate"]),
                        target_f1=TARGET, target_epoch=epochs, time_s=t,
                        energy_gpu_kwh=e, average_power_w=float(p["average_power"]),
                        provenance="reconstructed_from_published_traces")
            rows.append(item)
            valid.append(item)
        configurations.append(dict(
            configuration=config, batch_size=int(p["batch_size"]),
            power_limit_w=int(p["power_limit"]), optimizer=group[0]["optimizer"],
            learning_rate=float(group[0]["learning_rate"]), target_f1=TARGET,
            training_paths=len(group), power_profiles=1,
            mean_target_epoch=statistics.mean(x["target_epoch"] for x in valid),
            time_s=statistics.mean(x["time_s"] for x in valid),
            energy_gpu_kwh=statistics.mean(x["energy_gpu_kwh"] for x in valid),
            min_time_s=min(x["time_s"] for x in valid),
            max_time_s=max(x["time_s"] for x in valid),
            min_energy_gpu_kwh=min(x["energy_gpu_kwh"] for x in valid),
            max_energy_gpu_kwh=max(x["energy_gpu_kwh"] for x in valid),
            average_power_w=float(p["average_power"]),
            main_demo=int(int(p["batch_size"]) in main_batches
                          and int(p["power_limit"]) in main_limits),
            provenance="reconstructed_from_published_traces"))
    unmatched = sorted({int(r["batch_size"]) for r in train} - set(batches))
    assert len(configurations) == 49 and len(rows) == 490
    assert sum(r["main_demo"] for r in configurations) == 6
    write_csv(OUT / "training_paths.csv", rows)
    write_csv(OUT / "configurations.csv", configurations)
    manifest = dict(
        version="0.1", date="2026-11-19", source_revision=REVISION,
        paper="https://www.usenix.org/conference/nsdi23/presentation/you",
        trace_url=f"https://github.com/ml-energy/zeus/tree/{REVISION}/examples/research_reproducibility/zeus_nsdi23/trace",
        target_f1=TARGET, configurations=len(configurations), reconstructed_rows=len(rows),
        training_paths_with_power_profiles=70,
        excluded_training_batches_without_v100_power_profile=unmatched,
        demo_selection="Minimum batch 8, paper baseline 32, maximum batch 56; power limits 100 and 250 W. Selected by settings, not label outcomes.",
        units=dict(time="s", energy="kWh, GPU only", power="W", quality="validation F1 target, 0-100 scale"),
        reconstruction="time_s=target_epoch*time_per_epoch; energy_gpu_kwh=time_s*average_power_w/3600000",
        assumptions=["Profile reuse across training paths; epoch count independent of power limit, following Zeus.",
                     "Static configurations: no online optimizer exploration or JIT profiling cost added.",
                     "Ten training paths share one power profile per batch/power pair; they are not ten independent power measurements.",
                     "Target 84 is a reached threshold, not an exact final F1 score.",
                     "CPU, RAM, storage, cooling and hardware manufacture excluded."],
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(SOURCE.glob("*")) if p.is_file() and p.name != "LICENSE"})
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"configurations":49,"reconstructed_rows":490,"main_demo":6,
                      "excluded_batches":unmatched}))
    for r in configurations:
        if r["main_demo"]:
            print(f"{r['configuration']}: {r['time_s']/60:.2f} min, {r['energy_gpu_kwh']:.6f} kWh")


if __name__ == "__main__":
    main()
