#!/usr/bin/env bash
# Echo Studio launcher. Creates a venv on first run, then starts the server.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
VENV=".venv"

if [ ! -d "$VENV" ]; then
  echo "→ creating virtualenv ($VENV)…"
  "$PY" -m venv "$VENV"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  echo "→ upgrading pip"
  pip install --upgrade pip
  echo "→ NOTE: install the torch build for YOUR hardware first (see README)."
  echo "  (skipping torch here; installing the rest)"
  pip install -r requirements.txt
else
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi

# Fetch the Chroma Q4 model if it isn't there yet.
if [ ! -f "models/chroma-q4.gguf" ] && [ -z "${ECHO_CHROMA_GGUF:-}" ]; then
  echo "→ Chroma Q4 model not found — downloading…"
  "$PY" scripts/download_models.py
fi

HOST="${ECHO_HOST:-0.0.0.0}"
PORT="${ECHO_PORT:-8000}"
echo "→ starting Echo Studio on http://${HOST}:${PORT}"
exec uvicorn backend.main:app --host "$HOST" --port "$PORT"
