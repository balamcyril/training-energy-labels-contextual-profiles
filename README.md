# Training energy labels under electricity constraints

Reproducibility resources for the working article **Reliability of AI Energy Labels under Electricity Constraints: The Training Case**. Numerical protocol **0.2**, resource release **v0.2.0**, 24 November 2026. This release retains the original protocol-0.1 BERT results and adds explicit selection comparators, stricter sample-level admissibility, a shifted-start sensitivity, median imputation, and a complete compact ShuffleNet-v2/CIFAR-100 control. It is a versioned secondary analysis, not new training or field validation.

This repository reconstructs 49 BERT/SQuAD configurations from published Zeus traces and evaluates four separate contextual profiles. The second-task archive retains 70 CIFAR-100 candidates, of which 63 meet the declared quality eligibility rule. It contains the archived inputs, numerical configuration, preparation and analysis scripts, tests, computed results, and plotting code. No GPU, model weights, training execution, credentials or external service is needed to reproduce the results.

## Reproduce the results

Python **3.13** was used for the verified reproduction. From this repository's root, create an environment and install the plotting dependencies:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead:
# .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/reproduce.py --check
```

`--check` works in a temporary copy and leaves the checkout unchanged. It verifies source checksums, reconstructs both corpora, runs both analyses and 17 unit tests, checks the separate historical class counts, regenerates six figures and six numerical tabulars, and compares the numerical outputs with the frozen files for both versions. It also checks that the archived national factors are exact rows of the archived Ember data. Figure metadata such as PDF creation times can differ; numerical values, labels and table contents are checked independently of that metadata. The original BERT summaries must agree exactly with protocol 0.1.

For numerical reproduction without third-party Python packages:

```bash
python scripts/reproduce.py --check --no-figures
```

To deliberately regenerate the outputs in the checkout, omit `--check`:

```bash
python scripts/reproduce.py
```

After dependency installation, these commands work offline. No archived third-party notebook or simulator is executed. The independent historical reconstruction uses the bundled verified source table.

## Organization

```text
config/protocol.json             Preserved protocol-0.1 reference parameters
config/validation_v0_2.json       Accepted revision choices and second-task specification
data/sources/zeus/               Pinned traces, upstream documentation and license
data/sources/carbon/             Frozen Ember archive, factors and provenance
data/sources/gaissa_2023/         Extracts for the separate historical check
data/configurations.csv         49 reconstructed configuration means
data/training_paths.csv         490 reconstructed configuration/trajectory rows
data/manifest.json              Corpus scope, source revision and checksums
data/cifar100/                  All second-task candidates, trajectories and provenance
results/                        All profiles, sensitivities and summaries
results/validation_v0_2/         Separate task outputs, comparisons and strict shortlists
figures/                        Six figures in PDF, SVG and PNG; plotted values
tables.tex                      Four generated tabular commands (requires booktabs)
scripts/                        Preparation, analysis, plotting and verification
provenance/                     Original metadata and frozen numerical checksums
supporting_audit/                Separate corpus-suitability audit and instructions
requirements.txt                Verified plotting dependency versions
CITATION.cff                    Citation metadata for these resources
THIRD_PARTY_NOTICES.md           Attribution and source-specific terms
```

This folder is also the standalone `contextual_profiles/` directory in the manuscript project. It does not depend on parent folders, the Wiley template or a local temporary download. The manuscript itself, author correspondence, working notes, old drafts and credentials are not part of this repository.

## Profiles and parameters

The primary physical comparisons use the same BERT-base-uncased/SQuAD/V100 corpus and a reached F1 target of 84. Baseline labels use frozen 20/40/60/80% quantile boundaries, half-up rounding and time:energy:carbon weights of 2:1:1. The baseline assumes no additional interruption, budget or information mask; this is an analytical reference, not a field observation of electricity reliability.

| Profile | Inputs and interpretation | Main output |
|---|---|---|
| Limited energy | Budgets 0.75, 1 and 1.25 times median mean GPU energy; eligibility is checked on the configuration mean | `results/budgets.csv`, `results/profiles.csv` |
| Intermittent supply | Central schedule: 7,200 s on, 1,800 s off, 86,400 s horizon; full-state saves every 900 s useful work, 5 s save cost, 30 s restart cost; with/without usable recovery | `results/outage_paths.csv`, `results/profiles.csv` |
| Carbon intensity | Country factors: Kenya 83.516, Cameroon 219.859 (baseline), South Africa 717.431 g CO2e/kWh; separate illustrative PV 100/130/160 and diesel 1,140 coefficients | `results/profiles.csv`, `results/illustrative_supply.csv` |
| Missing evaluation information | One time value or energy/derived-carbon block masked at a time; mean of the other 48 configurations; estimation marker and omission comparator | `results/missing.csv` |

`config/protocol.json` also specifies all five weight ratios, interruption sensitivities, budget multipliers and the separate combined example. The missing-information procedure is implemented in `scripts/analyse_zeus_profiles.py`; its 48 donors follow from the fixed 49-configuration corpus. The reached quality target and pinned Zeus revision are retained in `scripts/prepare_zeus_corpus.py` and `data/manifest.json`.

Change parameters only in a new, explicitly versioned study. The default verifier intentionally fails if changed outputs no longer match this article's frozen results. It does not optimize weights, thresholds or simulation assumptions to reproduce a desired ranking.

## Protocol 0.2 validation extension

Run `python scripts/validate_revision.py` and `python scripts/render_revision.py` to regenerate only the extension; the full `reproduce.py --check` command verifies both versions together. The extension does not overwrite `results/reference.csv`, the original scenario files, or the original frozen thresholds. The simulator's optional `first_on_s` argument preserves its original behavior when omitted.

- **M1/M2/M3:** original best-class recommendations; contextual feasibility then original label; identical feasibility then contextual label. `methods.csv` records all identities, infeasible selections and selections dominated by another context-eligible configuration. Ties are retained.
- **Stricter practical recommendations:** every available trajectory must complete within 24 hours and meet any imposed budget individually. Remove Pareto-dominated configurations within this strict feasible set, then select its best contextual letter. Dominance concerns mean contextual costs, not per-trajectory dominance. `strict.csv` and `weights.csv` keep this separate from the original mean-based comparisons.
- **Start phase:** the extra case uses one hour until the first cutoff, followed by the unchanged two-hour supply windows and 30-minute outages. All other parameters and the 24-hour horizon remain fixed.
- **Missing information:** mean, median and omission use the same single-record masks. There are 48 BERT donors or 62 CIFAR donors; a hidden cost and its direct reconstruction routes are unavailable to the estimators. Calibration is retained from the complete reference. Omission has no numerical prediction or absolute error.
- **CIFAR-100:** match dataset, ShuffleNet-v2, Adadelta, batch size and target accuracy 0.6 in the archived Zeus/V100 inputs. Of 54 distinct learning trajectories, batch 4096/run 4 has no recorded target epoch. All 70 candidates and 378 reconstructed records remain in `data/cifar100/`; the seven configurations sharing that failed target outcome have no completed-task mean or label. The comparison calibrates on 63 configurations and 343 reused trajectory records. Repeats per batch vary from four to ten. Neither their reuse nor unequal sample sizes support a probabilistic reliability guarantee.
- **Task-specific reference:** CIFAR thresholds and budgets use its own quality-eligible configuration means. All candidates have equal calibration weight; available trajectories have equal weight within a configuration mean. The task's median mean GPU energy is 0.0155084104 kWh. Letters cannot be ranked across the two tasks. The same recovery costs remain simulation assumptions, not measured ShuffleNet costs.

`results/validation_v0_2/bert_squad/` and `shufflenet_cifar100/` each contain `profiles.csv`, `methods.csv`, `strict.csv`, `weights.csv`, `missing.csv`, `outage_paths.csv`, `calibration.csv` and `summary.json`. The enclosing folder contains the source manifest, two generated tabulars, and the new figure's plotted values. `provenance/reference_validation_v0_2_sha256.json` freezes the reviewed extension separately from the original results.

Key checks include BERT median-budget eligibility 25 on means versus 4 on all trajectories; 43 changed BERT labels with a shifted first window and checkpoints; CIFAR budget counts 10/32/35 on means versus 2/14/33 on all trajectories; and central no-recovery/checkpoint completion of 56/63 CIFAR configurations. The CIFAR best-class set survives central interruptions, while omission changes 3/63 recommendation sets per hidden field versus 10/63 for either imputer. These are conditional outcomes, including counterexamples to universal relabeling or imputation benefits.

The cross-context strict-shortlist intersection is empty for BERT. For CIFAR it contains batch 128 at 100 or 125 W, Adadelta learning rate 0.1, across the listed separately evaluated physical scenarios and five weight choices. These short trajectories finish before all tested cutoffs. This finding excludes arbitrary start times, unseen runs, missing costs, recalibration and simultaneous constraints; it is not a universal recommendation.

## Files, units and expected checks

Times are seconds unless a rendered table or axis explicitly converts them to minutes. `energy_gpu_kwh` is GPU energy in kWh; `carbon_g` is the product of that energy and `intensity_g_per_kwh`, in grams CO2e. Configurations identify both batch size and configured GPU power limit; the limit is not the measured average power.

- `reference.csv`: baseline costs, grades and labels for all 49 configurations.
- `profiles.csv`: budget, country-factor and central interruption results. Check `eligible` and `status` before interpreting time as completed-task duration. Failed configurations have no completed-task letter.
- `outage_paths.csv`: 980 outcomes, one per reconstructed trajectory and recovery policy; includes completion, active time, waiting implications, saves and lost work.
- `outage_sensitivity.csv`, `weights.csv`, `calibration_sensitivity.csv`: the declared sensitivity cases.
- `missing.csv`: 98 separate masks; no experiment hides all rows simultaneously. Energy imputation also estimates carbon. The asterisk denotes an estimate, not uncertainty quantified by a confidence interval.
- `combined.csv`: the one combined example reported separately in the discussion.
- `summary.json`: thresholds, recommendation identities, dominated recommendations and summary counts. `source_hashes` records the current package's input and script paths; the original metadata remains in `provenance/`.
- `historical_check.json`: the independent historical reconstruction, separate from Zeus comparisons.

Expected main checks include budget eligibility 0/25/49; central no-recovery completion 124/490 trajectories but zero configurations completing all ten; 49 eligible configurations and 37 changed letters with checkpoints; 31 and 18 changed letters for the Kenyan and South African factors; and 32 and 31 changed letters for the two masking experiments. The historical check recovers A=37, B=486, C=826, D=55 and E=15 across 1,419 archived rows.

## Scientific scope

These are secondary-analysis and deterministic simulation results, not new training or field measurements. Seventy learning trajectories are reused under seven power limits; the 490 reconstructed rows are not independent electrical repetitions. Energy covers the GPU only. Budget eligibility on a mean does not certify battery autonomy. The simulation assumes successful full-state recovery and constant profiled GPU power during active operations. National factors are counterfactual contexts; PV/diesel coefficients are an illustrative sensitivity with incompletely harmonized life-cycle boundaries. The missing-information experiment retains the already-published calibration and does not validate predictions on unseen models. Recommendations retain ties and do not establish universal model rankings.

## Provenance and reuse

- Zeus: [NSDI 2023 paper](https://www.usenix.org/conference/nsdi23/presentation/you), [pinned source revision](https://github.com/ml-energy/zeus/tree/9604d1346215f820fb04983cb3df2468f9c873e4/examples/research_reproducibility/zeus_nsdi23). Archived text line endings are retained; local SHA-256 values describe those archived bytes rather than upstream Git blob IDs.
- Ember: [Yearly Electricity Data](https://ember-energy.org/data/yearly-electricity-data/), archived on 18 November 2026 from the 17 November revision; country comparisons use 2024, the latest common year in that snapshot. Original extraction metadata and its historical `candidate_*` names are preserved.
- Historical classification: [Castaño et al., replication package v3](https://doi.org/10.5281/zenodo.8115187). Only the unchanged required table, notebook and README are mirrored. The complete 76.4 MB archive was checksum-verified before extraction.
- The PV/diesel coefficients are documented in the protocol with [their source DOI](https://doi.org/10.1016/j.seta.2021.101471). No publisher PDF is redistributed for that study.

The original analysis scripts and their associated original documentation are available under the [MIT License](LICENSE). Retain the copyright and permission notices when redistributing them. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the precise scope and the unchanged third-party source terms. Cite this repository's release and commit together with the upstream publications; see `CITATION.cff`. This working resource release is not an assertion that the associated manuscript has been accepted or published.
