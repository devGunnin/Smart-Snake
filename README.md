# Smart-Snake

Smart-Snake is a multiplayer snake project with:
- A FastAPI backend for lobby/game lifecycle and WebSocket gameplay
- A React/TypeScript frontend lobby + canvas client
- Optional AI training and AI opponents

## Prerequisites

- Python 3.10+
- Node.js 20+ and npm

## Install

```bash
pip install -e '.[dev,server,ai]'
cd frontend && npm install
```

## Run With Frontend

Use the helper script to start backend + frontend together:

```bash
./scripts/play.sh
```

Then open `http://127.0.0.1:5173`.

Environment overrides (optional):
- `BACKEND_HOST`, `BACKEND_PORT`
- `FRONTEND_HOST`, `FRONTEND_PORT`
- `VITE_API_URL`

## Train AI

Use the helper script to launch training:

```bash
./scripts/train.sh --num-games 200 --grid-width 20 --grid-height 20
```

Any arguments passed to `scripts/train.sh` are forwarded to:
- `smart-snake-train train ...`

After training, checkpoint files are written under `checkpoints/`.
These can be used by AI opponents in the lobby.

## Validation

```bash
make check
cd frontend && npm run lint && npm run build
```
