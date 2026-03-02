#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v smart-snake-train >/dev/null 2>&1; then
  echo "smart-snake-train not found. Install deps with: pip install -e '.[ai,dev]'"
  exit 1
fi

smart-snake-train train "$@"
