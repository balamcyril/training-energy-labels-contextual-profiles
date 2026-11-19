# Attribution and source terms

## Original analysis code

The original preparation, profile analysis, tests, plotting and verification code in `scripts/`, the original optional reader `supporting_audit/audit_squad_it_training_logs.py`, and their associated original documentation are licensed under the [MIT License](LICENSE). This grant also covers the same original code and documentation distributed in resource release v0.1.0. The license does not relicense archived third-party code, notebooks, documentation or datasets, and does not assign a new license to research datasets or results. The source-specific terms below remain unchanged.

## Zeus

The five archived source files in `data/sources/zeus/` originate from **Zeus: Understanding and Optimizing GPU Energy Consumption of DNN Training**, Jie You, Jae-Won Chung and Mosharaf Chowdhury, NSDI 2023. The upstream repository is [ml-energy/zeus](https://github.com/ml-energy/zeus), pinned at `9604d1346215f820fb04983cb3df2468f9c873e4`.

The upstream [Apache License 2.0](https://github.com/ml-energy/zeus/blob/9604d1346215f820fb04983cb3df2468f9c873e4/LICENSE) is preserved in `data/sources/zeus/LICENSE`. The source simulator and documentation are archived for provenance and are not executed by the reproduction command. Text files retain the line endings normalized during the original local archival step; no numerical trace content was changed. Their checksums are in `provenance/original_manifest.json`.

## Ember

The electricity archive, extracted country rows and methodology document in `data/sources/carbon/` are attributed to **Ember**, [Yearly Electricity Data](https://ember-energy.org/data/yearly-electricity-data/). Ember states that its content is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The dataset revision is 17 November 2026, retrieved 18 November 2026. The complete CSV is compressed without changing its decompressed content. `candidate_factors.csv` and `candidate_generation_rows.csv` are extracts for the three countries and 2024; they are not original complete Ember datasets. `metadata.json` retains source URLs, dates, extraction status and checksums. Country-factor application to the Zeus workload is this study's counterfactual calculation, not an Ember measurement of these training runs.

## Historical class-count check

`data/sources/gaissa_2023/` contains unchanged extracts from **Replication Package for 'Exploring the Carbon Footprint of Hugging Face's ML Models: A Repository Mining Study'**, Joel Castaño Fernández, Silverio Martínez-Fernández, Xavier Franch and Justus Bogner, version v3, [DOI 10.5281/zenodo.8115187](https://doi.org/10.5281/zenodo.8115187).

The [record metadata](https://zenodo.org/api/records/8115187) identifies **CC BY 4.0**. The source archive MD5 is `0f1ad54ea13f45e750edde4bf401fd2a`; extraction details and member SHA-256 values are in `data/sources/gaissa_2023/provenance.json`. The original notebook is not executed. The independent reconstruction preserves the archived calculation and rounding; it is not a claim to reproduce every function of GAISSALabel.

## Supporting audit

The optional audit addresses five published SQuAD-IT training logs from the pinned Hugging Face repositories identified in `supporting_audit/audit.json`. Log binaries and model weights are not mirrored here. The optional reader retrieves the declared public files; their upstream terms remain applicable. Its observations were used to assess corpus suitability, not as physical inputs to the Zeus profiles.
