#!/usr/bin/env bash
# Start the Footprint API from the repo root.
# Usage: ./run_server.sh [port]           default port 8000 (or $PORT)
#        HOST=0.0.0.0 ./run_server.sh     expose on the LAN
#        RELOAD=1 ./run_server.sh         dev only: restart on backend/*.py edits
# Demo default is NO reload: an edit to any backend file mid-demo would restart the
# server and kill the request in flight.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/uvicorn ]; then
  echo "missing .venv — run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi
source .venv/bin/activate
PORT="${1:-${PORT:-8000}}"
ARGS=(--app-dir backend server:app --host "${HOST:-127.0.0.1}" --port "$PORT")
if [ "${RELOAD:-0}" = "1" ]; then ARGS+=(--reload --reload-dir backend); fi
echo "Footprint API on http://${HOST:-127.0.0.1}:${PORT}  (GET /health)  reload=${RELOAD:-0}"
exec uvicorn "${ARGS[@]}"
