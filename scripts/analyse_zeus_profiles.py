"""Secondary analysis v0.1: fixed grades, explicit constraints and outage simulation.

Standard library only. Does not train models or treat simulated values as measurements.
Run prepare_zeus_corpus.py first. Numerical assumptions live in protocol.json.
"""
from pathlib import Path
from bisect import bisect_left
import csv
import hashlib
import json
import math
import statistics as st

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULT = ROOT / "results"
LETTERS = "ABCDE"
CRITERIA = ("time_s", "energy_gpu_kwh", "carbon_g")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def quantile(values, p):
    x = sorted(values)
    index = (len(x)-1)*p
    lo, hi = math.floor(index), math.ceil(index)
    return x[lo] + (index-lo)*(x[hi]-x[lo])


def grade_record(record, thresholds, weights):
    grades = [bisect_left(thresholds[k], record[k]) for k in CRITERIA]
    score = sum(w*g for w, g in zip(weights, grades))/sum(weights)
    return dict(grade_T=grades[0], grade_E=grades[1], grade_C=grades[2],
                score=score, label=LETTERS[math.floor(score+0.5)])


def simulate(work_s, power_w, checkpoint, *, on_s, off_s, horizon_s,
             checkpoint_interval_work_s, checkpoint_cost_s, restart_cost_s):
    """Deterministic windows; progress and GPU active time are distinct.

    Incremental saves retain complete training state. A save interrupted before
    completion is invalid. Completion exactly at a cutoff takes precedence.
    GPU consumes no energy during outages. Work time is a continuous surrogate.
    """
    if (work_s <= 0 or power_w <= 0 or on_s <= 0 or off_s < 0
            or horizon_s <= 0 or checkpoint_interval_work_s <= 0
            or checkpoint_cost_s < 0 or restart_cost_s < 0):
        raise ValueError("Invalid physical parameters")
    now = active = progress = saved = lost = 0.0
    interruptions = saves = 0
    eps = 1e-8
    while now < horizon_s-eps:
        boundary = now + on_s
        end = min(boundary, horizon_s)
        if interruptions:
            spent = min(restart_cost_s, end-now)
            active += spent
            now += spent
        while now < end-eps:
            remaining = work_s-progress
            next_save = (saved+checkpoint_interval_work_s-progress
                         if checkpoint else math.inf)
            available = end-now
            if remaining <= min(next_save, available)+eps:
                now += remaining
                active += remaining
                return dict(completed=1, time_s=now, active_s=active,
                            energy_gpu_kwh=active*power_w/3_600_000,
                            interruptions=interruptions, saves=saves,
                            lost_work_s=lost)
            step = min(next_save, available)
            progress += step
            active += step
            now += step
            if checkpoint and next_save <= available+eps:
                spent = min(checkpoint_cost_s, end-now)
                active += spent
                now += spent
                if spent+eps >= checkpoint_cost_s:
                    saved = progress
                    saves += 1
        if end >= horizon_s-eps:
            break
        # An actual power cutoff, not the observation horizon.
        lost += progress-saved
        progress = saved if checkpoint else 0.0
        interruptions += 1
        now = min(boundary+off_s, horizon_s)
    return dict(completed=0, time_s=horizon_s, active_s=active,
                energy_gpu_kwh=active*power_w/3_600_000,
                interruptions=interruptions, saves=saves, lost_work_s=lost)


def imputed_value(records, hidden_id, criterion):
    # No cached whole-corpus mean, hidden record, or recursively imputed donor.
    donors = [r[criterion] for r in records if r["configuration"] != hidden_id]
    if not donors:
        raise ValueError("No eligible donor")
    return st.mean(donors), len(donors)


def available_label(record, thresholds, hidden, weights):
    available = [i for i, k in enumerate(CRITERIA)
                 if k != hidden and not (hidden == "energy_gpu_kwh" and k == "carbon_g")]
    score = sum(weights[i]*bisect_left(thresholds[CRITERIA[i]], record[CRITERIA[i]])
                for i in available)/sum(weights[i] for i in available)
    return LETTERS[math.floor(score+0.5)]


def recommendations(rows):
    feasible = [r for r in rows if r["eligible"] and r["label"]]
    if not feasible:
        return []
    best = min(r["label"] for r in feasible)
    return sorted(r["configuration"] for r in feasible if r["label"] == best)


def main():
    protocol = json.loads((ROOT / "config/protocol.json").read_text(encoding="utf-8"))
    weights = protocol["weights_T_E_C"]
    configs = read_csv(DATA / "configurations.csv")
    paths = read_csv(DATA / "training_paths.csv")
    for r in configs + paths:
        for key in ("time_s", "energy_gpu_kwh", "average_power_w"):
            r[key] = float(r[key])
    factors = {r["Area"]: float(r["Emissions intensity (gCO2e/kWh)"])
               for r in read_csv(ROOT / "data/sources/carbon/candidate_factors.csv")}
    intensity = factors[protocol["reference_country"]]
    for r in configs:
        r["carbon_g"] = r["energy_gpu_kwh"]*intensity
    thresholds = {key: [quantile([r[key] for r in configs], p)
                        for p in protocol["quantile_probabilities"]] for key in CRITERIA}
    median_e = st.median(r["energy_gpu_kwh"] for r in configs)
    RESULT.mkdir(exist_ok=True)
    reference, profiles, missing, outage_runs, supply = [], [], [], [], []

    def record(r, profile, t, e, ci=intensity, eligible=1, status="completed"):
        item = dict(configuration=r["configuration"], main_demo=int(r["main_demo"]),
                    profile=profile, time_s=t, energy_gpu_kwh=e,
                    carbon_g=e*ci, intensity_g_per_kwh=ci,
                    eligible=int(eligible), status=status)
        item.update(grade_record(item, thresholds, weights) if status == "completed"
                    else dict(grade_T="", grade_E="", grade_C="", score="", label=""))
        return item

    for r in configs:
        base = record(r, "reference", r["time_s"], r["energy_gpu_kwh"])
        reference.append(base)
        group = [p for p in paths if p["configuration"] == r["configuration"]]
        for b in protocol["budget_multipliers"]:
            profiles.append(record(r, f"budget_{b:g}", r["time_s"], r["energy_gpu_kwh"],
                                   eligible=r["energy_gpu_kwh"] <= b*median_e))
        for country, ci in factors.items():
            profiles.append(record(r, f"carbon_{country}", r["time_s"], r["energy_gpu_kwh"], ci))
        for kind, ci in protocol["supply_type_variant"]["factors_g_per_kwh"].items():
            supply.append(record(r, f"illustrative_supply_{kind}", r["time_s"], r["energy_gpu_kwh"], ci))
        for checkpoint in (False, True):
            name = "outage_checkpoint" if checkpoint else "outage_no_recovery"
            runs = [simulate(p["time_s"], p["average_power_w"], checkpoint,
                             **protocol["outage"]) for p in group]
            for p, outcome in zip(group, runs):
                outage_runs.append(dict(configuration=r["configuration"], run=p["run"],
                                        policy=name, **outcome))
            completed = all(x["completed"] for x in runs)
            # No survivor-only mean or task-completion label when some paths fail.
            profiles.append(record(r, name, st.mean(x["time_s"] for x in runs),
                                   st.mean(x["energy_gpu_kwh"] for x in runs),
                                   eligible=completed,
                                   status="completed" if completed else "not_all_10_paths_completed"))
        for hidden in ("time_s", "energy_gpu_kwh"):
            value, n = imputed_value(configs, r["configuration"], hidden)
            filled = dict(r)
            filled[hidden] = value
            filled["carbon_g"] = filled["energy_gpu_kwh"]*intensity
            graded = grade_record(filled, thresholds, weights)
            control = available_label(r, thresholds, hidden, weights)
            # Candidate set with exactly one masked record at a time.
            missing.append(dict(configuration=r["configuration"], main_demo=int(r["main_demo"]),
                                hidden=hidden, donors=n, true_value=r[hidden],
                                imputed_value=value, absolute_error=abs(value-r[hidden]),
                                reference_label=base["label"], imputed_label=graded["label"]+"*",
                                available_only_label=control,
                                label_changed=int(graded["label"] != base["label"])))
    # Set-level checks use all 49 complete reference records, not a partial loop.
    ref_recommendations = recommendations(reference)
    for case in missing:
        imputed_set, available_set = [dict(x) for x in reference], [dict(x) for x in reference]
        for collection, label in ((imputed_set, case["imputed_label"][0]),
                                  (available_set, case["available_only_label"])):
            for x in collection:
                if x["configuration"] == case["configuration"]:
                    x["label"] = label
        case["recommendations_changed"] = int(recommendations(imputed_set) != ref_recommendations)
        case["available_recommendations_changed"] = int(recommendations(available_set) != ref_recommendations)

    sensitivity = []
    for name in sorted({r["profile"] for r in reference+profiles}):
        rows = [r for r in reference+profiles if r["profile"] == name]
        for w in protocol["weight_sensitivity"]:
            altered = [dict(r, **grade_record(r, thresholds, w))
                       for r in rows if r["eligible"] and r["status"] == "completed"]
            sensitivity.append(dict(profile=name, weights=":".join(map(str, w)),
                                    recommended=";".join(recommendations(altered)),
                                    eligible=len(altered)))
    physical_sensitivity = []
    for change in protocol["outage_sensitivity"]:
        params = dict(protocol["outage"], **{k:v for k,v in change.items() if k != "name"})
        for r in configs:
            group = [p for p in paths if p["configuration"] == r["configuration"]]
            for checkpoint in (False, True):
                runs = [simulate(p["time_s"], p["average_power_w"], checkpoint, **params) for p in group]
                complete = all(x["completed"] for x in runs)
                item = record(r, change["name"], st.mean(x["time_s"] for x in runs),
                              st.mean(x["energy_gpu_kwh"] for x in runs), eligible=complete,
                              status="completed" if complete else "not_all_10_paths_completed")
                item["checkpoint"] = int(checkpoint)
                physical_sensitivity.append(item)
    combined = []
    for r in profiles:
        if r["profile"] == "outage_checkpoint":
            combined.append(record(r, "combined_discussion_only", r["time_s"], r["energy_gpu_kwh"],
                                   factors["South Africa"], r["eligible"] and r["energy_gpu_kwh"] <= median_e,
                                   r["status"]))
    budget_detail = [dict(configuration=r["configuration"], multiplier=b, budget_kwh=b*median_e,
                         mean_within_budget=int(r["energy_gpu_kwh"] <= b*median_e),
                         paths_within_budget=sum(p["energy_gpu_kwh"] <= b*median_e for p in paths
                                                 if p["configuration"] == r["configuration"]),
                         total_paths=10)
                     for r in configs for b in protocol["budget_multipliers"]]
    calibration = []
    for batch in sorted({r["batch_size"] for r in configs}, key=int):
        donors = [r for r in configs if r["batch_size"] != batch]
        th = {key: [quantile([r[key] for r in donors], p)
                    for p in protocol["quantile_probabilities"]] for key in CRITERIA}
        altered = [dict(r, **grade_record(r, th, weights)) for r in reference]
        calibration.append(dict(excluded_batch=batch, calibration_n=len(donors), evaluated_n=len(altered),
                                changed_labels=sum(x["label"] != y["label"] for x,y in zip(reference,altered)),
                                recommended=";".join(recommendations(altered))))
    for name, rows in (("reference", reference), ("profiles", profiles), ("missing", missing),
                       ("outage_paths", outage_runs), ("weights", sensitivity),
                       ("outage_sensitivity", physical_sensitivity), ("combined", combined),
                       ("budgets", budget_detail), ("illustrative_supply", supply),
                       ("calibration_sensitivity", calibration)):
        write_csv(RESULT / f"{name}.csv", rows)

    summary = dict(version=protocol["version"], thresholds=thresholds,
                   energy_reference_kwh=median_e, factors_g_per_kwh=factors,
                   profiles={}, missing={}, source_hashes={})
    for name in sorted({r["profile"] for r in reference+profiles+combined}):
        rows = [r for r in reference+profiles+combined if r["profile"] == name]
        best = recommendations(rows)
        admissible = [r for r in rows if r["eligible"] and r["status"] == "completed"]
        dominated = [r["configuration"] for r in admissible if r["configuration"] in best
                     and any(all(s[k] <= r[k] for k in CRITERIA)
                             and any(s[k] < r[k] for k in CRITERIA) for s in admissible)]
        summary["profiles"][name] = dict(eligible=sum(r["eligible"] for r in rows),
            recommended=best, recommended_but_pareto_dominated=dominated,
            changed_labels=sum(a["label"] != b["label"] for a,b in zip(reference, rows) if b["label"]),
            evaluable_labels=sum(bool(r["label"]) for r in rows))
    for hidden in ("time_s", "energy_gpu_kwh"):
        subset = [r for r in missing if r["hidden"] == hidden]
        summary["missing"][hidden] = dict(n=len(subset),
            changed_labels=sum(r["label_changed"] for r in subset),
            changed_recommendations=sum(r["recommendations_changed"] for r in subset),
            available_changed_recommendations=sum(r["available_recommendations_changed"] for r in subset),
            mean_absolute_error=st.mean(r["absolute_error"] for r in subset))
    for path in [ROOT / "config/protocol.json", DATA / "configurations.csv", DATA / "training_paths.csv",
                 Path(__file__), ROOT / "scripts/prepare_zeus_corpus.py"]:
        summary["source_hashes"][path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    (RESULT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
