#!/bin/bash

# Start script for Google Cloud Run deployment
# This script sets up the environment and starts the illustrator web application

set -euo pipefail

echo "Starting Manuscript Illustrator..."

# Allow Cloud Run to inject HOST/PORT while providing sensible defaults for local runs
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
APP_MODULE="${APP_MODULE:-illustrator.web.app:app}"
UVICORN_WORKERS="${UVICORN_WORKERS:-1}"
FORWARDED_ALLOW_IPS="${FORWARDED_ALLOW_IPS:-*}"
UVICORN_EXTRA_ARGS="${UVICORN_EXTRA_ARGS:-}"

# Ensure we're in the correct directory
cd /app

echo "Starting uvicorn ${APP_MODULE} on ${HOST}:${PORT}..."

# Assemble uvicorn command line safely
declare -a uvicorn_cmd=(
  uvicorn
  "${APP_MODULE}"
  --host "${HOST}"
  --port "${PORT}"
  --proxy-headers
  --forwarded-allow-ips "${FORWARDED_ALLOW_IPS}"
  --log-level info
)

if [[ "${UVICORN_WORKERS}" != "1" ]]; then
  uvicorn_cmd+=(--workers "${UVICORN_WORKERS}")
fi

if [[ -n "${UVICORN_EXTRA_ARGS}" ]]; then
  read -r -a extra_args <<<"${UVICORN_EXTRA_ARGS}"
  uvicorn_cmd+=("${extra_args[@]}")
fi

exec "${uvicorn_cmd[@]}"
