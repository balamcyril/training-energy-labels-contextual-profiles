"""Generate publication figures and tables exclusively from analysis CSV/JSON files."""
from pathlib import Path
from collections import Counter
import csv
import json
import statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results"
FIG = ROOT / "figures"
TABLE = ROOT / "tables.tex"
LETTERS = "ABCDE"
COLORS = ["#1b7837", "#80b88a", "#f1d16a", "#e58e51", "#b54747", "#dadde1"]


def read(name):
    with (DATA / f"{name}.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def code(r):
    return r["configuration"].replace("bert_", "").replace("_", "/")


def save(fig, stem):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIG / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def tex_table(stem, headings, rows, alignment):
    text = "\\begin{tabular}{"+alignment+"}\n\\toprule\n"
    text += " & ".join(headings)+" \\\\\n\\midrule\n"
    text += "\n".join(" & ".join(map(str,row))+" \\\\" for row in rows)
    text += "\n\\bottomrule\n\\end{tabular}\n"
    # Update only the generated tabular data; keep captions, literature tables
    # and manually edited material elsewhere in the shared file unchanged.
    source = TABLE.read_bytes()
    newline = b"\r\n" if b"\r\n" in source else b"\n"
    begin = f"% BEGIN GENERATED TABULAR: {stem}".encode("ascii") + newline
    end = f"% END GENERATED TABULAR: {stem}".encode("ascii")
    if source.count(begin) != 1 or source.count(end) != 1:
        raise ValueError(f"Missing or duplicated table markers for {stem} in {TABLE}")
    prefix, _, remaining = source.partition(begin)
    _, _, suffix = remaining.partition(end)
    generated = text.encode("utf-8").replace(b"\n", newline)
    updated = prefix + begin + generated + end + suffix
    if updated != source:
        TABLE.write_bytes(updated)


def grade_grid(ax, rows, title):
    """Display the existing 49 labels; no thresholds or scores are recomputed."""
    coordinates = {}
    for row in rows:
        batch, power = row["configuration"].removeprefix("bert_b").split("_p")
        key = (int(batch), int(power))
        if key in coordinates:
            raise ValueError(f"Duplicated configuration in {title}: {key}")
        coordinates[key] = row
    batches = sorted({key[0] for key in coordinates})
    powers = sorted({key[1] for key in coordinates})
    if len(coordinates) != 49 or len(batches) != 7 or len(powers) != 7:
        raise ValueError(f"Incomplete 7 by 7 configuration grid in {title}")
    matrix = np.array([
        [LETTERS.index(coordinates[b, p]["label"])
         if coordinates[b, p]["eligible"] == "1" else 5 for p in powers]
        for b in batches
    ])
    ax.imshow(matrix, cmap=ListedColormap(COLORS),
              norm=BoundaryNorm(np.arange(-.5, 6.5), 6), aspect="equal")
    ax.set_xticks(range(7), powers, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(7), batches, fontsize=10)
    ax.set(xlabel="GPU power limit (W)", ylabel="Batch size", title=title)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-.5, 7, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 7, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=.8)
    ax.tick_params(which="minor", length=0)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(j, i, "--" if value == 5 else LETTERS[value],
                    ha="center", va="center", fontsize=10.5,
                    color="white" if value in (0, 4) else "#20252a")
    return matrix.tolist()


def grade_legend(fig, budget=False):
    handles = [Patch(facecolor=COLORS[i], label=letter) for i, letter in enumerate(LETTERS)]
    if budget:
        handles.append(Patch(facecolor=COLORS[5], label="-- Outside budget"))
    fig.legend(handles=handles, title="Fixed A-E label (A best)",
               loc="outside lower center", ncol=len(handles), frameon=False,
               fontsize=10, title_fontsize=10)


def render_profile_figures():
    """Four profiles from saved analysis outputs, including existing sensitivities."""
    FIG.mkdir(parents=True, exist_ok=True)
    ref, profiles = read("reference"), read("profiles")
    missing, phys, supply = read("missing"), read("outage_sensitivity"), read("illustrative_supply")
    budgets = read("budgets")
    summary = json.loads((DATA / "summary.json").read_text(encoding="utf-8"))
    by_profile = {name: [r for r in profiles if r["profile"] == name]
                  for name in {r["profile"] for r in profiles}}
    audit = {"source": "Existing CSV/JSON analysis outputs; no new simulations",
             "grids": {}, "counts": {}}
    style = {"font.family": "DejaVu Sans", "font.size": 11,
             "axes.titlesize": 11, "axes.labelsize": 10,
             "axes.spines.top": False, "axes.spines.right": False,
             "pdf.fonttype": 42, "svg.fonttype": "none"}
    with plt.rc_context(style):
        # Profile 1: preserve every cell of the former reference/budget columns.
        fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.6), layout="constrained",
                                 gridspec_kw={"width_ratios": [1, 1, .92]})
        audit["grids"]["reference"] = grade_grid(axes[0], ref, "(a) Reference")
        audit["grids"]["budget_1"] = grade_grid(
            axes[1], by_profile["budget_1"], "(b) Median energy budget")
        multipliers = [.75, 1., 1.25]
        counts = [sum(r["mean_within_budget"] == "1" for r in budgets
                      if float(r["multiplier"]) == value) for value in multipliers]
        for value, count in zip(multipliers, counts):
            key = "budget_" + format(value, "g")
            assert count == summary["profiles"][key]["eligible"]
        axes[2].bar(range(3), counts, color="#426580", width=.6)
        for x, count in enumerate(counts):
            axes[2].text(x, count + 1, str(count), ha="center", va="bottom", fontsize=11)
        axes[2].set_xticks(range(3), ["0.75", "1.00", "1.25"])
        axes[2].set(title="(c) Budget admissibility", xlabel="Budget / reference median",
                    ylabel="Admissible configuration means", ylim=(0, 56), yticks=[0, 10, 20, 30, 40, 49])
        axes[2].grid(axis="y", alpha=.18)
        axes[2].set_axisbelow(True)
        grade_legend(fig, budget=True)
        audit["counts"]["budgets"] = counts
        save(fig, "fig2_profile_limited_energy")

        # Profile 2: the central label comparison plus both former sensitivity panels.
        fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.1), layout="constrained")
        grade_grid(axes[0, 0], ref, "(a) Reference")
        audit["grids"]["outage_checkpoint"] = grade_grid(
            axes[0, 1], by_profile["outage_checkpoint"], "(b) 2 h supply, with checkpoints")
        windows = [1, 2, 4]
        no_cp = [sum(r["eligible"] == "1" for r in phys
                     if r["profile"] == f"on_{h}h" and r["checkpoint"] == "0")
                 if h != 2 else summary["profiles"]["outage_no_recovery"]["eligible"] for h in windows]
        cp = [sum(r["eligible"] == "1" for r in phys
                  if r["profile"] == f"on_{h}h" and r["checkpoint"] == "1")
              if h != 2 else summary["profiles"]["outage_checkpoint"]["eligible"] for h in windows]
        ax = axes[1, 0]
        ax.plot(windows, no_cp, "o-", color="#bb593d", label="No usable recovery state")
        ax.plot(windows, cp, "s-", color="#197a72", label="Periodic checkpoints")
        for x, a, b in zip(windows, no_cp, cp):
            ax.annotate(str(a), (x, a), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=10)
            ax.annotate(str(b), (x, b), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=10)
        ax.set(title="(c) Completion under interruptions", xlabel="Uninterrupted supply window (h)",
               ylabel="Configurations completing all 10 paths", ylim=(-3, 59), xticks=windows,
               yticks=[0, 10, 20, 30, 40, 49])
        ax.legend(fontsize=9, loc="center left", frameon=False)
        ax.grid(alpha=.18)
        distributions = {}
        for x, h in enumerate(windows):
            rows = by_profile["outage_checkpoint"] if h == 2 else [
                r for r in phys if r["profile"] == f"on_{h}h" and r["checkpoint"] == "1"]
            counts = Counter(r["label"] for r in rows if r["eligible"] == "1")
            distributions[str(h)] = [counts[letter] for letter in LETTERS]
            bottom = 0
            for k, letter in enumerate(LETTERS):
                count = counts[letter]
                axes[1, 1].bar(x, count, bottom=bottom, color=COLORS[k], width=.6)
                if count:
                    axes[1, 1].text(x + (.36 if count < 4 else 0), bottom + count / 2,
                                    str(count), ha="left" if count < 4 else "center", va="center",
                                    fontsize=10, color="white" if k in (0, 4) else "#20252a")
                bottom += count
        axes[1, 1].set_xticks(range(3), windows)
        axes[1, 1].set(title="(d) Labels with checkpoints", xlabel="Uninterrupted supply window (h)",
                       ylabel="Configurations by label", ylim=(0, 53), yticks=[0, 10, 20, 30, 40, 49])
        grade_legend(fig)
        audit["counts"]["interruptions"] = {"windows_h": windows, "without_recovery": no_cp,
                                              "with_checkpoints": cp, "labels": distributions}
        save(fig, "fig3_profile_intermittent_supply")

        # Profile 3: retain all country-specific labels, plus the existing supply sensitivity.
        fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.1), layout="constrained")
        for ax, country, panel in [(axes[0, 0], "Cameroon", "a"),
                                   (axes[0, 1], "Kenya", "b"),
                                   (axes[1, 0], "South Africa", "c")]:
            rows = by_profile[f"carbon_{country}"]
            intensity = float(rows[0]["intensity_g_per_kwh"])
            title = f"({panel}) {country}" + (" (reference)" if country == "Cameroon" else "")
            title += f"\n{intensity:.3f} g CO2e/kWh"
            audit["grids"][f"carbon_{country}"] = grade_grid(ax, rows, title)
        supply_cases = [("PV_low", "PV 100"), ("PV_midpoint_assumption", "PV 130"),
                        ("PV_high", "PV 160"), ("Diesel", "Diesel 1140")]
        distributions = {}
        for y, (key, _) in enumerate(supply_cases):
            rows = [r for r in supply if r["profile"] == "illustrative_supply_" + key]
            assert len(rows) == 49 and all(r["eligible"] == "1" for r in rows)
            counts = Counter(r["label"] for r in rows)
            distributions[key] = [counts[letter] for letter in LETTERS]
            left = 0
            for k, letter in enumerate(LETTERS):
                count = counts[letter]
                axes[1, 1].barh(y, count, left=left, color=COLORS[k], height=.6)
                if count:
                    axes[1, 1].text(left + count / 2, y, str(count), ha="center", va="center",
                                    fontsize=10, color="white" if k in (0, 4) else "#20252a")
                left += count
        axes[1, 1].set_yticks(range(4), [name for _, name in supply_cases], fontsize=10)
        axes[1, 1].invert_yaxis()
        axes[1, 1].set(title="(d) Illustrative supply coefficients\n(g CO2e/kWh)",
                       xlabel="Configurations by label", xlim=(0, 49), xticks=[0, 10, 20, 30, 40, 49])
        grade_legend(fig)
        audit["counts"]["supply_labels"] = distributions
        save(fig, "fig4_profile_carbon_intensity")

        # Profile 4: preserve the two original transition matrices and their scale.
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), layout="constrained")
        transitions = {}
        for ax, hidden, title in zip(axes, ["time_s", "energy_gpu_kwh"],
                                    ["(a) Time hidden", "(b) Energy / derived carbon hidden"]):
            counts = np.zeros((5, 5), dtype=int)
            for row in missing:
                if row["hidden"] == hidden:
                    assert row["imputed_label"].endswith("*")
                    counts[LETTERS.index(row["reference_label"]), LETTERS.index(row["imputed_label"][0])] += 1
            assert counts.sum() == 49 and counts.max() <= 25
            assert int(counts.sum() - np.trace(counts)) == summary["missing"][hidden]["changed_labels"]
            im = ax.imshow(counts, cmap="Blues", vmin=0, vmax=25)
            ax.set(xticks=range(5), xticklabels=list(LETTERS), yticks=range(5), yticklabels=list(LETTERS),
                   xlabel="Label after mean imputation (*)", ylabel="Complete-reference label", title=title)
            for i in range(5):
                for j in range(5):
                    ax.text(j, i, str(counts[i, j]), ha="center", va="center", fontsize=11,
                            color="white" if counts[i, j] > 12 else "#253443")
            transitions[hidden] = counts.tolist()
        fig.colorbar(im, ax=list(axes), label="Number of single-configuration masks", ticks=range(0, 26, 5), shrink=.85)
        audit["counts"]["imputation_transitions"] = transitions
        save(fig, "fig5_profile_missing_information")
    (FIG / "profile_plot_values.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    if not TABLE.is_file():
        raise FileNotFoundError(f"Shared manuscript table file not found: {TABLE}")
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10,
                         "axes.spines.top":False, "axes.spines.right":False,
                         "pdf.fonttype":42, "svg.fonttype":"none"})
    ref, profiles, missing = read("reference"), read("profiles"), read("missing")
    summary = json.loads((DATA / "summary.json").read_text(encoding="utf-8"))
    t = np.array([float(x["time_s"])/60 for x in ref])
    e = np.array([float(x["energy_gpu_kwh"]) for x in ref])
    power = [int(x["configuration"].split("_p")[1]) for x in ref]
    frontier = [i for i in range(len(ref)) if not any(
        t[j] <= t[i] and e[j] <= e[i] and (t[j] < t[i] or e[j] < e[i])
        for j in range(len(ref)))]
    frontier.sort(key=lambda i:t[i])
    fig, ax = plt.subplots(figsize=(7.2,4.7), layout="constrained")
    sc = ax.scatter(t,e,c=power,cmap="viridis",s=48,edgecolor="white",linewidth=.5,zorder=3)
    ax.plot(t[frontier],e[frontier],color="#303b4a",linestyle="--",linewidth=1,label="Pareto frontier")
    for i,r in enumerate(ref):
        if r["main_demo"] == "1":
            ax.scatter(t[i],e[i],facecolors="none",edgecolors="black",s=100,linewidth=1,zorder=4)
            ax.annotate(code(r),(t[i],e[i]),xytext=(5,5),textcoords="offset points",fontsize=8)
    ax.axhline(summary["energy_reference_kwh"],color="#9b4a1b",linestyle=":",label="Median energy budget")
    ax.set(xlabel="Time to target F1 (min)",ylabel="GPU energy to target F1 (kWh)")
    ax.grid(alpha=.15)
    ax.legend(loc="lower left",bbox_to_anchor=(0,1.02),ncol=2,fontsize=8,frameon=False)
    fig.colorbar(sc,ax=ax,label="GPU power limit (W)")
    save(fig,"fig1_time_energy")

    render_profile_figures()

    mainrows=[]
    for r in ref:
        if r["main_demo"]!="1":continue
        c=next(x for x in profiles if x["configuration"]==r["configuration"] and x["profile"]=="outage_checkpoint")
        mainrows.append([code(r),f"{float(r['time_s'])/60:.2f}",f"{float(r['energy_gpu_kwh']):.4f}",
                         r["label"],f"{float(c['time_s'])/60:.2f}",f"{float(c['energy_gpu_kwh']):.4f}",c["label"]])
    tex_table("six_configurations",["Configuration","$T_0$ (min)","$E_0$ (kWh)","$L_0$","$T_P$ (min)","$E_P$ (kWh)","$L_P$"],mainrows,"lrrrrrr")
    thresholdrows=[]
    for k,name,factor in [("time_s","Time (min)",1/60),("energy_gpu_kwh","GPU energy (kWh)",1),("carbon_g","Carbon (g CO$_2$e)",1)]:
        thresholdrows.append([name]+[f"{x*factor:.4f}" for x in summary["thresholds"][k]])
    tex_table("thresholds",["Criterion","A/B","B/C","C/D","D/E"],thresholdrows,"lrrrr")
    selected=[("Reference","reference"),("Budget 0.75 median","budget_0.75"),("Budget 1.00 median","budget_1"),
              ("Budget 1.25 median","budget_1.25"),("Interruption, no recovery","outage_no_recovery"),
              ("Interruption, checkpoints","outage_checkpoint"),("Kenya factor","carbon_Kenya"),("South Africa factor","carbon_South Africa")]
    tex_table("profile_summary",["Profile","Admissible / 49","Labels changed","Best-class set size"],
              [[name,summary["profiles"][key]["eligible"],
                str(summary["profiles"][key]["changed_labels"])+" / "+str(summary["profiles"][key]["evaluable_labels"])
                if summary["profiles"][key]["evaluable_labels"] else "Not assigned",
                len(summary["profiles"][key]["recommended"])] for name,key in selected],"lrrr")
    tex_table("missing_summary",["Hidden field","Letters changed","Imputation: set changes","Omission: set changes"],
              [[name,f"{summary['missing'][key]['changed_labels']}/49",f"{summary['missing'][key]['changed_recommendations']}/49",
                f"{summary['missing'][key]['available_changed_recommendations']}/49"]
               for name,key in [("Time","time_s"),("Energy + carbon","energy_gpu_kwh")]],"lrrr")

    report=["# Article 2 — Résultats calculés, version 0.1", "", "19 novembre 2026. Analyse secondaire et simulations ; aucune nouvelle exécution de modèle.", "",
            "Corpus : 49 configurations BERT/SQuAD sur V100, cible F1 84, 70 parcours d'apprentissage réutilisés sous sept plafonds de puissance. Les 490 lignes reconstruites ne sont pas 490 expériences électriques indépendantes.", "",
            "## Six configurations illustratives", "", "| Configuration | Temps de référence (min) | Énergie GPU (kWh) | Label | Temps avec coupures et sauvegardes (min) | Énergie simulée (kWh) | Label |", "|---|---:|---:|:---:|---:|---:|:---:|"]
    report += ["| "+" | ".join(row)+" |" for row in mainrows]
    report += ["", "## Conclusions de la version calculée", "",
               "- Budget médian : 25/49 configurations respectent le budget en moyenne ; les lettres ne changent pas. Les consommations individuelles restent dans budgets.csv, sans garantie d'autonomie réelle.",
               "- Coupures centrales (2 h alimentées, 30 min hors tension, horizon 24 h) : sans état de reprise, 124/490 parcours reconstruits terminent, mais aucune configuration ne termine ses dix parcours. Avec sauvegardes additionnelles, les 49 configurations terminent leurs dix parcours ; 37 lettres changent.",
               "- Avec seulement 1 h d'alimentation continue, toutes les configurations avec sauvegardes reçoivent E : l'échelle sature et ne discrimine plus. Les valeurs brutes deviennent indispensables.",
               "- Facteurs nationaux : 31/49 lettres changent pour le Kenya, 18/49 pour l'Afrique du Sud. Le carbone reste proportionnel à l'énergie dans chaque contexte. Ces changements résultent de la calibration fixe et de l'agrégation, pas d'une inversion du classement énergétique physique.",
               "- Imputation : masquer la durée modifie 32/49 lettres ; masquer l'énergie modifie 31/49 lettres. Les ensembles recommandés changent dans 12/49 et 10/49 masquages, contre 20/49 dans chacun des deux contrôles par omission. Cela ne prouve pas une supériorité générale de la moyenne.",
               "- Type d'alimentation : sensibilité illustrative à des facteurs PV/diesel publiés, avec hypothèse de transfert simplifiée. Ce volet n'est pas une comparaison d'analyses de cycle de vie harmonisées.", "",
               "## Figures", ""]
    for n,title in [("fig1_time_energy","Compromis temps–énergie"),
                    ("fig2_profile_limited_energy","Profil : énergie limitée"),
                    ("fig3_profile_intermittent_supply","Profil : alimentation intermittente"),
                    ("fig4_profile_carbon_intensity","Profil : intensité carbone"),
                    ("fig5_profile_missing_information","Profil : informations manquantes")]:
        report += [f"![{title}](figures/{n}.png)", ""]
    (ROOT / "RESULTATS_ARTICLE_2.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print("Generated five figures in PDF/SVG/PNG, four result tables, and RESULTATS_ARTICLE_2.md")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles-only", action="store_true",
                        help="Render only the four profile figures from existing results.")
    args = parser.parse_args()
    if args.profiles_only:
        render_profile_figures()
        print("Generated four profile figures from existing analysis outputs.")
    else:
        main()

