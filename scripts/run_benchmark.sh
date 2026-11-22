#!/usr/bin/env bash
set -euo pipefail
# Each test server is configured to use one computational thread.
# For exact platform-specific results, run on the target machine.
python -m src.benchmark --out reports/performance.json --raw reports/latency_raw.csv "$@"
