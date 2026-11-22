"""One-command deterministic end-to-end reproduction of the delivery ML system.

Runs real HTTP benchmarks (not simulated latency), tests and a fresh gated update.
The managed update registry is isolated in retraining/full_run and can be regenerated.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from data.generate import ROOT


def _run(module: str, *args: str) -> tuple[float, str]:
    command = [sys.executable, "-B", "-m", module, *args]
    print("\n> " + " ".join(command), flush=True)
    start = time.perf_counter()
    process = subprocess.run(command, cwd=ROOT, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                             capture_output=True, text=True, check=False)
    if process.stdout:
        print(process.stdout[-9000:], flush=True)
    if process.stderr:
        print(process.stderr[-9000:], file=sys.stderr, flush=True)
    if process.returncode:
        raise RuntimeError(f"Step {module} exited with code {process.returncode}")
    return round(time.perf_counter() - start, 3), process.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate the complete delivery ML project")
    parser.add_argument("--quick", action="store_true", help="Shorter HTTP benchmark, all other checks remain full")
    args = parser.parse_args()
    if not (ROOT / "configs/data.json").exists():
        raise RuntimeError("Run this project from its repository root")
    durations = {}
    results = {}
    for key, module in (
        ("data", "data.generate"),
        ("training", "src.train"),
        ("offline_evaluation", "src.evaluate"),
        ("serving_bundle", "src.prepare_service"),
        ("release_replay", "src.replay"),
        ("monitoring", "src.monitor"),
    ):
        duration, _ = _run(module)
        durations[key] = duration
        results[key] = "passed"

    # Keep the source stage-8 registry untouched; wipe only our own reproduction registry.
    managed = ROOT / "retraining" / "full_run"
    if managed.exists():
        shutil.rmtree(managed)
    duration, _ = _run("src.retrain", "--registry", str(managed), "--exercise-rejection")
    durations["retraining"] = duration
    results["retraining"] = "passed"

    benchmark_args = ("--steady-count", "80", "--burst-count", "40", "--warmup-count", "10") if args.quick else ()
    duration, _ = _run("src.benchmark", "--out", "reports/performance.json",
                       "--raw", "reports/latency_raw.csv", *benchmark_args)
    durations["http_benchmark"] = duration
    results["http_benchmark"] = "passed"

    # Generate preliminary evidence before pytest: final integration tests inspect
    # the complete report and reproduction registry. Rebuild after pytest with
    # the actual confirmed test count.
    _run("src.final_report", "--test-count", "0",
         "--profile", "quick" if args.quick else "full")
    duration, test_stdout = _run("pytest", "-q", "tests")
    durations["tests"] = duration
    results["tests"] = "passed"
    match = re.search(r"(\d+) passed", test_stdout)
    if match is None:
        raise RuntimeError("Cannot confirm that all tests passed")
    tests_passed = int(match.group(1))

    duration, _ = _run("src.final_report", "--test-count", str(tests_passed),
                       "--profile", "quick" if args.quick else "full")
    durations["final_report"] = duration
    summary = {
        "status": "passed", "profile": "quick" if args.quick else "full",
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "test_count": tests_passed, "steps": results,
        "durations_seconds": durations,
        "reports": ["reports/report.md", "reports/rubric.csv",
                    "reports/reproducibility.json", "reports/performance.json",
                    "reports/latency_raw.csv"],
    }
    (ROOT / "reports/pipeline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\nFINAL RESULT " + json.dumps({"status": summary["status"],
             "profile": summary["profile"], "tests_passed": tests_passed,
             "report": "reports/report.md"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
