#!/usr/bin/env bash
set -euo pipefail
python -m data.generate >/dev/null
python -m src.train
python -m pytest -q tests
