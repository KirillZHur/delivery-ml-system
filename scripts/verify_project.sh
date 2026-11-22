#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
python -B -m pytest -q tests
python -B -m src.final_report --verify-only
