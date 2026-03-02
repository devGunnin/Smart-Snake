#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_HOST="${FRONTEND_HOST:-127.0.0.1}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

cd "$ROOT_DIR"

if ! command -v uvicorn >/dev/null 2>&1; then
  echo "uvicorn not found. Install backend deps with: pip install -e '.[server,ai,dev]'"
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "npm not found. Install Node.js 20+ and npm."
  exit 1
fi

if [ ! -d frontend/node_modules ]; then
  echo "Installing frontend dependencies..."
  (cd frontend && npm install)
fi

export VITE_API_URL="${VITE_API_URL:-http://${BACKEND_HOST}:${BACKEND_PORT}}"

echo "Starting backend API on http://${BACKEND_HOST}:${BACKEND_PORT}"
uvicorn smart_snake.server.app:create_app \
  --factory \
  --host "$BACKEND_HOST" \
  --port "$BACKEND_PORT" &
BACKEND_PID=$!

cleanup() {
  if kill -0 "$BACKEND_PID" >/dev/null 2>&1; then
    kill "$BACKEND_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

echo "Starting frontend dev server on http://${FRONTEND_HOST}:${FRONTEND_PORT}"
cd frontend
npm run dev -- --host "$FRONTEND_HOST" --port "$FRONTEND_PORT"
