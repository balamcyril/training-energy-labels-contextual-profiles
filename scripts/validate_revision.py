"""Protocol 0.2: decision comparators, empirical admissibility and second-task control.

Preserves the protocol-0.1 outputs. No training or network access. Costs are
reconstructed or simulated, never reported as new electrical measurements.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics as st
from analyse_zeus_profiles import (CRITERIA, LETTERS, read_csv, write_csv,
    quantile, grade_record, available_label, recommendations, simulate)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/validation_v0_2"


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


def prepare_secondary(spec):
    source = ROOT / "data/sources/zeus"
    train = [r for r in read_csv(source / "summary_train.csv")
             if all(r[k] == spec[k] for k in ("dataset", "network", "optimizer"))
             and float(r["target_metric"]) == spec["target_metric"]]
    power = [r for r in read_csv(source / "summary_power_v100.csv")
             if all(r[k] == spec[k] for k in ("dataset", "network", "optimizer"))]
    key = lambda r: (r["dataset"], r["network"], r["optimizer"], int(r["batch_size"]))
    if len({(key(r), r["run"]) for r in train}) != len(train):
        raise ValueError("Duplicate learning trajectory")
    if len({(key(r), r["power_limit"]) for r in power}) != len(power):
        raise ValueError("Duplicate power profile")
    configurations, paths = [], []
    matched_keys = set()
    for p in sorted(power, key=lambda r: (int(r["batch_size"]), int(r["power_limit"]))):
        group = sorted([r for r in train if key(r) == key(p)], key=lambda r: int(r["run"]))
        if not group:
            continue
        matched_keys.add(key(p))
        if len({r["learning_rate"] for r in group}) != 1:
            raise ValueError("Inconsistent optimizer settings within a configuration")
        identity = f"{spec['configuration_prefix']}_b{p['batch_size']}_p{p['power_limit']}"
        runs = []
        for r in group:
            epoch = float(r["target_epoch"])
            reached = math.isfinite(epoch) and epoch > 0
            duration = epoch*float(p["time_per_epoch"]) if reached else None
            energy = duration*float(p["average_power"])/3_600_000 if reached else None
            row = dict(configuration=identity, batch_size=int(p["batch_size"]),
                       power_limit_w=int(p["power_limit"]), run=int(r["run"]),
                       optimizer=r["optimizer"], learning_rate=float(r["learning_rate"]),
                       target_metric=spec["target_metric"], quality_reached=int(reached),
                       target_epoch=epoch if reached else None, time_s=duration,
                       energy_gpu_kwh=energy, average_power_w=float(p["average_power"]))
            runs.append(row)
            paths.append(row)
        complete = all(r["quality_reached"] for r in runs)
        configurations.append(dict(configuration=identity, batch_size=int(p["batch_size"]),
            power_limit_w=int(p["power_limit"]), optimizer=group[0]["optimizer"],
            learning_rate=float(group[0]["learning_rate"]), target_metric=spec["target_metric"],
            training_paths=len(runs), successful_paths=sum(r["quality_reached"] for r in runs),
            quality_eligible=int(complete), power_profiles=1,
            time_s=st.mean(r["time_s"] for r in runs) if complete else None,
            energy_gpu_kwh=st.mean(r["energy_gpu_kwh"] for r in runs) if complete else None,
            average_power_w=float(p["average_power"])))
    matched = [r for r in train if key(r) in matched_keys]
    manifest = dict(task=spec, candidate_configurations=len(configurations),
        quality_eligible_configurations=sum(r["quality_eligible"] for r in configurations),
        unique_learning_trajectories=len(matched), reconstructed_rows=len(paths),
        successful_learning_trajectories=sum(math.isfinite(float(r["target_epoch"])) for r in matched),
        repeats_per_batch={str(b):sum(int(r["batch_size"]) == b for r in matched)
                          for b in sorted({int(r["batch_size"]) for r in matched})},
        failed_trajectories=[r for r in matched if not math.isfinite(float(r["target_epoch"]))],
        unmatched_training_batches=sorted({int(r["batch_size"]) for r in train if key(r) not in matched_keys}),
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [source/"summary_train.csv", source/"summary_power_v100.csv"]})
    folder = ROOT / "data/cifar100"
    folder.mkdir(exist_ok=True)
    write_csv(folder / "configurations.csv", configurations)
    write_csv(folder / "training_paths.csv", paths)
    dump(folder / "manifest.json", manifest)
    return configurations, paths, manifest


def load_primary():
    configs, paths = (read_csv(ROOT / "data" / name) for name in ("configurations.csv", "training_paths.csv"))
    for r in configs + paths:
        for k in ("time_s", "energy_gpu_kwh", "average_power_w", "learning_rate"):
            r[k] = float(r[k])
        for k in ("batch_size", "power_limit_w"):
            r[k] = int(r[k])
    for r in configs:
        r["quality_eligible"] = 1
        r["training_paths"] = int(r["training_paths"])
    for r in paths:
        r["quality_reached"] = 1
    return configs, paths


def dominates(a, b):
    return all(a[k] <= b[k] for k in CRITERIA) and any(a[k] < b[k] for k in CRITERIA)


def frontier(rows):
    return [r for r in rows if not any(dominates(s, r) for s in rows)]


def strict_admissible(runs, horizon, budget=None):
    return bool(runs) and all(r["completed"] and r["time_s"] <= horizon and
                             (budget is None or r["energy_gpu_kwh"] <= budget) for r in runs)


def method_sets(reference, contextual):
    """M1 ignores the added constraint; M2 and M3 share contextual eligibility."""
    original = {r["configuration"]:r for r in reference}
    filtered = [dict(r, label=original[r["configuration"]]["label"]) for r in contextual]
    return {"M1":recommendations(reference), "M2":recommendations(filtered),
            "M3":recommendations(contextual)}


def donor_estimate(reference, identity, hidden, estimator):
    donors = [r[hidden] for r in reference if r["configuration"] != identity]
    if not donors:
        raise ValueError("No donors")
    function = {"mean":st.mean, "median":st.median}[estimator]
    return function(donors), len(donors)


def evaluate(task, configs, paths, protocol, revision, factors):
    folder = OUT / task
    folder.mkdir(parents=True, exist_ok=True)
    valid = [r for r in configs if r["quality_eligible"]]
    groups = {r["configuration"]:[p for p in paths if p["configuration"] == r["configuration"]]
              for r in valid}
    intensity = factors[protocol["reference_country"]]
    weights = protocol["weights_T_E_C"]
    for r in valid:
        r["carbon_g"] = r["energy_gpu_kwh"]*intensity
    thresholds = {k:[quantile([r[k] for r in valid], p) for p in protocol["quantile_probabilities"]]
                  for k in CRITERIA}
    median_energy = st.median(r["energy_gpu_kwh"] for r in valid)
    horizon = protocol["outage"]["horizon_s"]

    def cost_record(r, name, runs, ci=intensity, budget=None):
        complete = all(x["completed"] for x in runs)
        t, e = st.mean(x["time_s"] for x in runs), st.mean(x["energy_gpu_kwh"] for x in runs)
        strict = strict_admissible(runs, horizon, budget)
        item = dict(configuration=r["configuration"], profile=name, time_s=t,
                    energy_gpu_kwh=e, carbon_g=e*ci, intensity_g_per_kwh=ci,
                    eligible=int(complete and (budget is None or e <= budget)),
                    completed=int(complete), budget_kwh=budget,
                    strict_eligible=int(strict), total_paths=len(runs),
                    completed_paths=sum(x["completed"] for x in runs),
                    max_time_s=max(x["time_s"] for x in runs),
                    max_energy_gpu_kwh=max(x["energy_gpu_kwh"] for x in runs))
        item.update(grade_record(item, thresholds, weights) if complete else
                    dict(grade_T="", grade_E="", grade_C="", score="", label=""))
        return item

    cases, outage_paths = {}, []
    def add(name, ci=intensity, budget=None, outage=None, checkpoint=False):
        rows = []
        for r in valid:
            group = groups[r["configuration"]]
            if outage is None:
                runs = [dict(completed=1, time_s=p["time_s"], energy_gpu_kwh=p["energy_gpu_kwh"]) for p in group]
            else:
                runs = [simulate(p["time_s"], p["average_power_w"], checkpoint, **outage) for p in group]
                for p, x in zip(group, runs):
                    outage_paths.append(dict(configuration=r["configuration"], profile=name, run=p["run"], **x))
            rows.append(cost_record(r, name, runs, ci, budget))
        cases[name] = rows

    add("reference")
    for b in protocol["budget_multipliers"]:
        add(f"budget_{b:g}", budget=b*median_energy)
    for country, ci in factors.items():
        add(f"carbon_{country}", ci=ci)
    for kind, ci in protocol["supply_type_variant"]["factors_g_per_kwh"].items():
        add(f"illustrative_supply_{kind}", ci=ci)
    for change in [{"name":"central"}, *protocol["outage_sensitivity"], revision["shifted_start"]]:
        params = dict(protocol["outage"], **{k:v for k,v in change.items() if k != "name"})
        for checkpoint in (False, True):
            suffix = "checkpoint" if checkpoint else "no_recovery"
            name = f"outage_{suffix}" if change["name"] == "central" else f"{change['name']}_{suffix}"
            add(name, outage=params, checkpoint=checkpoint)
    add("combined_discussion_only", ci=factors["South Africa"], budget=median_energy,
        outage=protocol["outage"], checkpoint=True)
    reference = cases["reference"]
    base_by_id = {r["configuration"]:r for r in reference}
    method_rows, strict_rows, weight_rows, profile_summary = [], [], [], {}
    for name, rows in cases.items():
        eligible = [r for r in rows if r["eligible"]]
        by_id = {r["configuration"]:r for r in rows}
        sets = method_sets(reference, rows)
        for method, selected in sets.items():
            infeasible = [i for i in selected if not by_id[i]["eligible"]]
            dominated = [i for i in selected if by_id[i]["eligible"]
                         and any(dominates(s, by_id[i]) for s in eligible)]
            method_rows.append(dict(profile=name, method=method, recommended=";".join(selected),
                count=len(selected), infeasible_count=len(infeasible), infeasible=";".join(infeasible),
                dominated_count=len(dominated), dominated=";".join(dominated),
                changed_from_M1=int(selected != sets["M1"])))
        strict_feasible = [r for r in rows if r["strict_eligible"]]
        strict_frontier = frontier(strict_feasible)
        strict_ids = {r["configuration"] for r in strict_frontier}
        strict_best = recommendations(strict_frontier)
        best_grid = []
        for w in protocol["weight_sensitivity"]:
            graded = [dict(r, **grade_record(r, thresholds, w)) for r in rows if r["completed"]]
            chosen = recommendations(graded)
            strict_chosen = recommendations([r for r in graded if r["configuration"] in strict_ids])
            best_grid.append(set(strict_chosen))
            weight_rows.append(dict(profile=name, weights=":".join(map(str,w)),
                                    recommended=";".join(chosen), strict_recommended=";".join(strict_chosen)))
        stable_weights = sorted(set.intersection(*best_grid))
        for r in rows:
            strict_rows.append(dict(profile=name, configuration=r["configuration"],
                strict_eligible=r["strict_eligible"], nondominated=int(r["configuration"] in strict_ids),
                recommended=int(r["configuration"] in strict_best),
                recommended_in_all_weight_cases=int(r["configuration"] in stable_weights),
                total_paths=r["total_paths"], completed_paths=r["completed_paths"],
                budget_kwh=r["budget_kwh"], max_time_s=r["max_time_s"],
                max_energy_gpu_kwh=r["max_energy_gpu_kwh"]))
        profile_summary[name] = dict(eligible=len(eligible), strict_eligible=len(strict_feasible),
            completed_paths=sum(r["completed_paths"] for r in rows),
            total_paths=sum(r["total_paths"] for r in rows),
            changed_labels=sum(r["label"] != base_by_id[r["configuration"]]["label"] for r in rows if r["label"]),
            labels={letter:sum(r["label"] == letter for r in rows) for letter in LETTERS},
            methods=sets, strict_recommended=strict_best,
            strict_recommended_all_weights=stable_weights,
            mean_time_s=st.mean(r["time_s"] for r in rows) if all(r["completed"] for r in rows) else None)

    missing, missing_summary = [], {}
    reference_set = recommendations(reference)
    for hidden in ("time_s", "energy_gpu_kwh"):
        missing_summary[hidden] = {}
        for estimator in revision["missing_estimators"]:
            subset = []
            for r in reference:
                if estimator == "omit":
                    value, n = None, len(reference)-1
                    label = available_label(r, thresholds, hidden, weights)
                else:
                    value, n = donor_estimate(reference, r["configuration"], hidden, estimator)
                    filled = dict(r, **{hidden:value})
                    filled["carbon_g"] = filled["energy_gpu_kwh"]*intensity
                    label = grade_record(filled, thresholds, weights)["label"]
                altered = [dict(x, label=label) if x["configuration"] == r["configuration"] else x for x in reference]
                chosen = recommendations(altered)
                subset.append(dict(configuration=r["configuration"], hidden=hidden, estimator=estimator,
                    donors=n if estimator != "omit" else 0, true_value=r[hidden], estimated_value=value,
                    absolute_error=abs(value-r[hidden]) if value is not None else None,
                    reference_label=r["label"], assessed_label=label+("*" if estimator != "omit" else ""),
                    label_changed=int(label != r["label"]), recommendations_changed=int(chosen != reference_set),
                    recommended=";".join(chosen)))
            missing.extend(subset)
            missing_summary[hidden][estimator] = dict(masks=len(subset),
                changed_labels=sum(r["label_changed"] for r in subset),
                changed_recommendations=sum(r["recommendations_changed"] for r in subset),
                mean_absolute_error=st.mean(r["absolute_error"] for r in subset) if estimator != "omit" else None)

    calibration = []
    for batch in sorted({r["batch_size"] for r in valid}):
        donors = [r for r in valid if r["batch_size"] != batch]
        altered_thresholds = {k:[quantile([r[k] for r in donors], p) for p in protocol["quantile_probabilities"]] for k in CRITERIA}
        altered = [dict(r, **grade_record(r, altered_thresholds, weights)) for r in reference]
        calibration.append(dict(excluded_batch=batch, donors=len(donors),
            changed_labels=sum(x["label"] != y["label"] for x,y in zip(reference, altered)),
            recommended=";".join(recommendations(altered)),
            recommendations_changed=int(recommendations(altered) != reference_set)))
    for filename, rows in (("profiles", [r for rows in cases.values() for r in rows]),
                           ("methods", method_rows), ("strict", strict_rows), ("weights", weight_rows),
                           ("missing", missing), ("outage_paths", outage_paths), ("calibration", calibration)):
        write_csv(folder/f"{filename}.csv", rows)
    summary = dict(task=task, candidate_configurations=len(configs), calibrated_configurations=len(valid),
                   thresholds=thresholds, energy_reference_kwh=median_energy,
                   profiles=profile_summary, missing=missing_summary,
                   calibration=calibration)
    physical_names = [name for name in cases if name != "combined_discussion_only"]
    summary["cross_context"] = dict(
        scope=physical_names,
        interpretation="Intersection across separately evaluated complete-information cases; not simultaneous operation, unseen runs, calibration changes or missing-information masks.",
        strict_recommended=sorted(set.intersection(*[set(profile_summary[n]["strict_recommended"]) for n in physical_names])),
        strict_recommended_all_weights=sorted(set.intersection(*[set(profile_summary[n]["strict_recommended_all_weights"]) for n in physical_names])),
        empty_strict_scenarios=[n for n in physical_names if not profile_summary[n]["strict_recommended"]])
    dump(folder/"summary.json", summary)
    return summary


def check_primary_agreement(summary):
    original = json.loads((ROOT/"results/summary.json").read_text())
    assert summary["thresholds"] == original["thresholds"]
    for name, expected in original["profiles"].items():
        actual = summary["profiles"][name]
        for k in ("eligible", "changed_labels"):
            assert actual[k] == expected[k], (name, k)
        assert actual["methods"]["M3"] == expected["recommended"], name
    for hidden, expected in original["missing"].items():
        actual = summary["missing"][hidden]["mean"]
        for k in ("changed_labels", "changed_recommendations", "mean_absolute_error"):
            assert actual[k] == expected[k], (hidden,k)


def main():
    protocol = json.loads((ROOT/"config/protocol.json").read_text())
    revision = json.loads((ROOT/"config/validation_v0_2.json").read_text())
    factors = {r["Area"]:float(r["Emissions intensity (gCO2e/kWh)"])
               for r in read_csv(ROOT/"data/sources/carbon/candidate_factors.csv")}
    primary_configs, primary_paths = load_primary()
    secondary_configs, secondary_paths, manifest = prepare_secondary(revision["secondary_task"])
    summaries = {}
    for task, configs, paths in (("bert_squad",primary_configs,primary_paths),
                                ("shufflenet_cifar100",secondary_configs,secondary_paths)):
        summaries[task] = evaluate(task, configs, paths, protocol, revision, factors)
    check_primary_agreement(summaries["bert_squad"])
    sources = [ROOT/"config/protocol.json", ROOT/"config/validation_v0_2.json",
               ROOT/"scripts/analyse_zeus_profiles.py", Path(__file__),
               ROOT/"data/configurations.csv", ROOT/"data/training_paths.csv",
               ROOT/"data/sources/zeus/summary_train.csv", ROOT/"data/sources/zeus/summary_power_v100.csv",
               ROOT/"data/sources/carbon/candidate_factors.csv"]
    dump(OUT/"manifest.json", dict(version="0.2", date=revision["date"],
        source_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        agreement_with_protocol_0_1="all original primary scenario and mean-imputation summaries agree",
        secondary_corpus=manifest))
    print(json.dumps({task:dict(calibrated=s["calibrated_configurations"],
        budget_kwh=s["energy_reference_kwh"], missing=s["missing"],
        profiles={k:v for k,v in s["profiles"].items() if k in ("reference","budget_1",
            "outage_no_recovery","outage_checkpoint","half_window_start_checkpoint")})
        for task,s in summaries.items()}, indent=2))


if __name__ == "__main__":
    main()
