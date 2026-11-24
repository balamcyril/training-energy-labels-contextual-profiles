"""Reproduce protocol 0.1 and its versioned 0.2 validation extension.

--check runs in a temporary copy and leaves this checkout unchanged.
--no-figures needs only the Python standard library; the complete run needs requirements.txt.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def run(root, args):
    result = subprocess.run([sys.executable, *args], cwd=root, text=True,
                            encoding="utf-8", capture_output=True)
    if result.returncode:
        raise RuntimeError(f"Failed: {' '.join(args)}\n{result.stdout}\n{result.stderr}")
    return result


def reproduce(root, figures=True):
    print(run(root, ["scripts/verify_sources.py"]).stdout.strip())
    expected = json.loads((root / "provenance/reference_numeric_sha256.json").read_text())
    expected_summary = json.loads((root / "provenance/original_summary.json").read_text())
    expected_summary.pop("source_hashes")
    expected_tables = (root / "tables.tex").read_bytes()
    expected_plot_values = json.loads((root / "figures/profile_plot_values.json").read_text())

    run(root, ["scripts/prepare_zeus_corpus.py"])
    analysis = run(root, ["scripts/analyse_zeus_profiles.py"])
    (root / "results/run_summary.txt").write_text(analysis.stdout, encoding="utf-8")
    historical = json.loads(run(root, ["scripts/reproduce_gaissa_2023_labels.py"]).stdout)
    expected_historical = json.loads((root / "provenance/original_gaissa_counts.json").read_text())
    for key, expected_value in expected_historical.items():
        if historical[key] != expected_value:
            raise ValueError(f"Historical consistency check differs: {key}")
    (root / "results/historical_check.json").write_text(json.dumps(historical, indent=2) + "\n", encoding="utf-8")
    print("Historical check: A=37, B=486, C=826, D=55, E=15 over 1,419 rows (separate from Zeus profiles).")
    tests = run(root, ["-m", "unittest", "discover", "-s", "scripts", "-p", "test_zeus_profiles.py"])
    print(tests.stderr.strip())

    for name, expected_hash in expected.items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != expected_hash:
            raise ValueError(f"Numerical regression or changed protocol: {name}")
    actual_summary = json.loads((root / "results/summary.json").read_text())
    actual_summary.pop("source_hashes")
    if actual_summary != expected_summary:
        raise ValueError("Scientific summary differs from the frozen article results")
    if figures:
        run(root, ["scripts/render_zeus_results.py"])
        if (root / "tables.tex").read_bytes() != expected_tables:
            raise ValueError("Regenerated numerical tables differ")
        if json.loads((root / "figures/profile_plot_values.json").read_text()) != expected_plot_values:
            raise ValueError("Regenerated plotted values differ")
    print(f"Verified {len(expected)} frozen numerical/configuration files and all scientific summary values.")
    if figures:
        print("Verified four numerical tables and five figures from unchanged result data.")
    run(root, ["scripts/validate_revision.py"])
    tests = run(root, ["-m", "unittest", "discover", "-s", "scripts", "-p", "test_revision.py"])
    print(tests.stderr.strip())
    if figures:
        run(root, ["scripts/render_revision.py"])
    extension = json.loads((root / "provenance/reference_validation_v0_2_sha256.json").read_text())
    for name, expected_hash in extension.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Protocol-0.2 validation regression: {name}")
    print(f"Verified {len(extension)} version-0.2 files, both tasks and all decision comparators.")
    if figures:
        print("Verified two new comparison tables and the CIFAR-100 control figure data.")
    print("No training, GPU execution or network access was required.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify in a temporary copy; do not overwrite outputs.")
    parser.add_argument("--no-figures", action="store_true", help="Verify numerical outputs without plotting dependencies.")
    args = parser.parse_args()
    if args.check:
        with tempfile.TemporaryDirectory(prefix="training-energy-labels-") as temporary:
            path = Path(temporary).resolve()
            if path.parent != Path(tempfile.gettempdir()).resolve():
                raise RuntimeError("Unexpected temporary output location")
            target = path / "resources"
            shutil.copytree(ROOT, target, ignore=shutil.ignore_patterns(".git", "__pycache__", ".venv", "build"))
            reproduce(target, figures=not args.no_figures)
    else:
        reproduce(ROOT, figures=not args.no_figures)


if __name__ == "__main__":
    main()
