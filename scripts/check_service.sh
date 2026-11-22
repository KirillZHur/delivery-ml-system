#!/usr/bin/env bash
set -euo pipefail
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
curl -fsS "${BASE_URL}/health/live"
printf '\n'
curl -fsS "${BASE_URL}/health/ready"
printf '\n'
curl -fsS -X POST "${BASE_URL}/predict" \
  -H 'Content-Type: application/json' \
  -d '{"order_id":"o00000001","ordered_at":"2026-01-01T00:09:05Z","distance_km":568.9763618716252}'
printf '\n'
