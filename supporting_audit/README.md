# Separate corpus-suitability audit

This optional audit records the inspection of published IT5/mT5 SQuAD-IT question-answering training logs. It supports the manuscript's explanation for selecting Zeus. It does not provide energy measurements or training-time inputs to the four contextual profiles.

`audit.json` retains the original source URLs, pinned revisions, byte counts, checksums and observations. Only the machine-specific `local_file` paths were replaced by relative paths. No log binary or model weight is included in this public repository. The complete local audit remains in the manuscript project's local archive.

To repeat the inspection in a separate environment:

```bash
python -m venv .venv-audit
# Activate that environment, then:
python -m pip install -r supporting_audit/requirements.txt
python supporting_audit/audit_squad_it_training_logs.py
```

This optional command requires network access to download the five pinned public log files if they are absent. It does not execute a model. Its output goes into `supporting_audit/logs/`, which is ignored by Git. Upstream terms continue to apply to downloaded files.

Recorded event spans are not automatically complete training durations. The absence of an energy tag in these examined files does not establish that suitable measurements cannot exist elsewhere. See the source study by [Sarti and Nissim](https://arxiv.org/abs/2203.03759).
