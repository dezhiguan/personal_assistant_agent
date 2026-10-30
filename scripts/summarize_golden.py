"""Display dimension scores and individual regressions from stored golden reports."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from personal_assistant_agent.evaluation.golden import failure_layer


def rates(rows, field="passed"):
    grouped = defaultdict(lambda: [0, 0])
    for row in rows:
        grouped[row["dimension"]][0] += int(row.get(field, row["passed"]))
        grouped[row["dimension"]][1] += 1
    return grouped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("artifacts/golden/latest.json"))
    parser.add_argument("--baseline", type=Path, default=Path("evaluations/golden/baseline.json"))
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    baseline = json.loads(args.baseline.read_text()) if args.baseline.exists() else None
    compatible = baseline is not None and all(
        report.get(k) == baseline.get(k)
        for k in (
            "suite_hash",
            "repeats",
            "judge_model",
            "judge_prompt_hash",
            "oracle_hash",
            "offline",
        )
    )
    current = rates(report["rows"])
    old = rates(baseline["rows"]) if compatible else {}
    layers = {
        layer: rates(report["rows"], layer + "_passed")
        for layer in ("semantic", "contract", "execution", "safety")
    }
    print("Dimension             Overall   Semantic Contract Execution Safety   Delta")
    for dimension, (passed, total) in sorted(current.items()):
        old_passed, old_total = old.get(dimension, (0, 0))
        delta = f"{100 * (passed / total - old_passed / old_total):+.1f}pp" if old_total else "n/a"
        scores = [f"{values[dimension][0]}/{values[dimension][1]}" for values in layers.values()]
        print(
            f"{dimension:21} {passed:3}/{total:<5} "
            + " ".join(f"{score:8}" for score in scores)
            + f" {delta}"
        )
    critical = [r for r in report["rows"] if r["critical"]]
    critical_passed = sum(r.get("safety_passed", r["passed"]) for r in critical)
    print(f"Critical safety: {critical_passed}/{len(critical)}")
    print("Gate:", "PASS" if report["gate"]["passed"] else "FAIL", report["gate"]["reasons"])
    if baseline and not compatible:
        print("Baseline comparison unavailable: suite/repeats/judge/scope differs")
    for row in report["rows"]:
        if not row["passed"]:
            details = ",".join(f"{failure_layer(f, row['critical'])}:{f}" for f in row["failures"])
            print(f"FAIL {row['id']} #{row['repeat']}: {details}")


if __name__ == "__main__":
    main()
