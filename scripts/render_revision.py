"""Compact protocol-0.2 figure and tabular data, generated only from saved results."""
from pathlib import Path
import json
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/validation_v0_2"
TASKS = ("bert_squad", "shufflenet_cifar100")
NAMES = {"bert_squad":"BERT/SQuAD", "shufflenet_cifar100":"ShuffleNet/CIFAR-100"}


def main():
    summaries = {t:json.loads((OUT/t/"summary.json").read_text()) for t in TASKS}
    table = []
    table += ["% BEGIN GENERATED TABULAR: method_comparison", r"\begin{tabular}{@{}llrrr@{}}",
        r"\toprule", r"Task & Context & M1 & M2 & M3 \\", r"\midrule"]
    contexts = [("budget_0.75","Tight budget"),("budget_1","Median budget"),
                ("outage_no_recovery","Outages, no recovery"),("outage_checkpoint","Outages, checkpoints"),
                ("carbon_Kenya","Kenyan factor"),("carbon_South Africa","South African factor")]
    for task in TASKS:
        with (OUT/task/"methods.csv").open() as f:
            methods = {(r["profile"],r["method"]):r for r in csv.DictReader(f)}
        for j,(profile,name) in enumerate(contexts):
            cells = []
            for method in ("M1","M2","M3"):
                r = methods[profile,method]
                cells.append(f"{r['count']} ({r['infeasible_count']}, {r['dominated_count']})")
            table.append(f"{NAMES[task] if j == 0 else ''} & {name} & "+" & ".join(cells)+r" \\")
        table.append(r"\addlinespace")
    table += [r"\bottomrule", r"\end{tabular}", "% END GENERATED TABULAR: method_comparison", "",
              "% BEGIN GENERATED TABULAR: imputation_comparison", r"\begin{tabular}{@{}llrrrrr@{}}",
              r"\toprule", r"Task & Hidden cost & MAE mean & MAE median & Mean & Median & Omit \\", r"\midrule"]
    for task in TASKS:
        for j,hidden in enumerate(("time_s","energy_gpu_kwh")):
            m = summaries[task]["missing"][hidden]
            convert = 60 if hidden == "time_s" else 1
            fmt = ".2f" if hidden == "time_s" else ".5f"
            unit = "Time (min)" if hidden == "time_s" else "Energy (kWh)"
            errors = [format(m[k]["mean_absolute_error"]/convert,fmt) for k in ("mean","median")]
            changed = [f"{m[k]['changed_recommendations']}/{m[k]['masks']}" for k in ("mean","median","omit")]
            table.append(f"{NAMES[task] if j == 0 else ''} & {unit} & "+" & ".join(errors+changed)+r" \\")
    table += [r"\bottomrule", r"\end{tabular}", "% END GENERATED TABULAR: imputation_comparison", ""]
    (OUT/"tables.tex").write_text("\n".join(table), encoding="utf-8")

    s = summaries["shufflenet_cifar100"]
    p = s["profiles"]
    plt.rcParams.update({"font.size":10,"axes.titlesize":11,"axes.labelsize":10,
                         "font.family":"DejaVu Sans","pdf.fonttype":42})
    fig,axes = plt.subplots(2,2,figsize=(10,6.8),layout="constrained")
    colors = ["#2166ac","#ef8a62","#6a51a3"]
    x=np.arange(3); width=.34
    budget_names=["budget_0.75","budget_1","budget_1.25"]
    mean_counts=[p[n]["eligible"] for n in budget_names]
    strict_counts=[p[n]["strict_eligible"] for n in budget_names]
    ax=axes[0,0]
    for offset,values,color,label in ((-width/2,mean_counts,colors[0],"Mean within budget"),
                                      (width/2,strict_counts,colors[1],"Every path within budget")):
        bars=ax.bar(x+offset,values,width,color=color,label=label); ax.bar_label(bars,padding=2)
    ax.set(xticks=x,xticklabels=["0.75","1.00","1.25"],xlabel="Budget / task reference median",
           ylabel="Eligible configurations",ylim=(0,68),title="(a) Energy budget")
    ax.legend(loc="upper left",fontsize=8)
    ax=axes[0,1]
    names=["reference","outage_no_recovery","outage_checkpoint"]
    counts=[p[n]["eligible"] for n in names]
    bars=ax.bar(x,counts,color=["#777777",colors[1],colors[0]]); ax.bar_label(bars,padding=2)
    ax.set(xticks=x,xticklabels=["Reference","No recovery","Checkpoints"],ylim=(0,68),
           ylabel="All paths complete",title="(b) Central interruption schedule")
    ax=axes[1,0]
    names=["carbon_Kenya","carbon_Cameroon","carbon_South Africa"]
    bottoms=np.zeros(3); grade_colors=["#1b7837","#80b88a","#f1d16a","#e58e51","#b54747"]
    distributions={}
    for letter,color in zip("ABCDE",grade_colors):
        values=[p[n]["labels"][letter] for n in names]; distributions[letter]=values
        bars=ax.bar(x,values,bottom=bottoms,label=letter,color=color)
        ax.bar_label(bars,labels=[str(v) if v else "" for v in values],label_type="center",fontsize=9)
        bottoms+=values
    ax.set(xticks=x,xticklabels=["Kenya","Cameroon","South Africa"],ylim=(0,73),
           ylabel="Configurations",title="(c) Labels under national factors")
    ax.legend(ncol=5,loc="upper center",fontsize=8,columnspacing=.9,handlelength=1)
    ax=axes[1,1]
    x2=np.arange(2); w=.24; missing_values={}
    for j,(method,label) in enumerate((("mean","Mean"),("median","Median"),("omit","Omission"))):
        counts=[s["missing"][h][method]["changed_recommendations"] for h in ("time_s","energy_gpu_kwh")]
        missing_values[method]=counts
        bars=ax.bar(x2+(j-1)*w,counts,w,label=label,color=colors[j]); ax.bar_label(bars,padding=2)
    ax.set(xticks=x2,xticklabels=["Time hidden","Energy + carbon hidden"],ylim=(0,14),
           ylabel="Changed sets / 63 masks",title="(d) Recommendation sensitivity")
    ax.legend(loc="upper center",ncol=3,fontsize=8)
    for ax in axes.flat:
        ax.spines[["top","right"]].set_visible(False)
    target=ROOT/"figures/fig6_cifar100_control"
    for suffix in ("pdf","svg","png"):
        fig.savefig(target.with_suffix("."+suffix),dpi=180)
    plt.close(fig)
    values=dict(budget_mean=mean_counts,budget_all_paths=strict_counts,
                interruption_completion=[p[n]["eligible"] for n in ["reference","outage_no_recovery","outage_checkpoint"]],
                carbon_labels=distributions,missing_recommendation_changes=missing_values)
    (OUT/"plot_values.json").write_text(json.dumps(values,indent=2)+"\n")
    print("Generated two comparison tabulars and the four-panel CIFAR-100 control figure.")


if __name__ == "__main__":
    main()
