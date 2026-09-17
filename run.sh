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

  # Install the right torch build automatically. Advanced override via
  # ECHO_CUDA=cu118|cu121|cu124|cpu (optional, no prompts).
  case "${ECHO_CUDA:-}" in
    cu118|cu121|cu124)
      echo "→ installing torch ($ECHO_CUDA)"
      pip install torch --index-url "https://download.pytorch.org/whl/$ECHO_CUDA" ;;
    cpu)
      echo "→ installing torch (CPU)"
      pip install torch --index-url https://download.pytorch.org/whl/cpu ;;
    *)
      if command -v nvidia-smi >/dev/null 2>&1; then
        echo "→ NVIDIA GPU detected: installing torch (CUDA 12.1)"
        pip install torch --index-url https://download.pytorch.org/whl/cu121
      else
        echo "→ no NVIDIA GPU: installing default torch (CPU / Apple MPS)"
        pip install torch
      fi ;;
  esac

  echo "→ installing dependencies"
  pip install -r requirements.txt
else
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi

# Fetch the Chroma Q4 model if it isn't there yet.
if [ ! -f "models/chroma-q4.gguf" ] && [ -z "${ECHO_CHROMA_GGUF:-}" ]; then
  echo "→ Chroma Q4 model not found — downloading…"
  python scripts/download_models.py
fi

HOST="${ECHO_HOST:-0.0.0.0}"
PORT="${ECHO_PORT:-8000}"
echo "→ starting Echo Studio on http://${HOST}:${PORT}"
exec uvicorn backend.main:app --host "$HOST" --port "$PORT"
