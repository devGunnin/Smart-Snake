# Smart Snake

A multiplayer snake game (2–4 players) with a DQN reinforcement-learning AI
opponent, Python/FastAPI backend, and React/TypeScript frontend.

## Architecture

```
┌──────────────────────┐      WebSocket / REST       ┌──────────────────────┐
│   React Frontend     │ ◄──────────────────────────► │   FastAPI Backend    │
│  (Vite, Canvas, TS)  │                              │  (async, uvicorn)    │
└──────────────────────┘                              └────────┬─────────────┘
                                                               │
                                                   ┌───────────┴───────────┐
                                                   │   Game Engine         │
                                                   │  (MultiplayerEngine)  │
                                                   └───────────┬───────────┘
                                                               │
                                                   ┌───────────┴───────────┐
                                                   │   DQN AI Agent        │
                                                   │  (PyTorch, CNN)       │
                                                   └───────────────────────┘
```

**Data flow**: Frontend → REST API (lobby/game lifecycle) → WebSocket
(real-time state) → Game Engine (step-based tick loop) → AI Agent
(per-tick inference for AI-controlled snakes).

Key components:
- **Game engine** (`src/smart_snake/`): NumPy-backed grid, step-based tick,
  simultaneous collision resolution, 2–4 player support.
- **Server** (`src/smart_snake/server/`): FastAPI app with in-memory
  `GameManager`, async tick loops, WebSocket broadcast, rate limiting.
- **AI** (`src/smart_snake/ai/`): DQN with CNN backbone, self-play training,
  prioritized replay, difficulty tiers (beginner → impossible).
- **Frontend** (`frontend/`): React 18 + Vite, canvas renderer at 60 fps,
  lobby UI, WebSocket auto-reconnect.

## Prerequisites

- Python 3.10+
- Node.js 20+ and npm

## Install

```bash
# Python (all extras for development)
pip install -e ".[dev,server,ai]"

# Frontend
cd frontend && npm install
```

Or use the Makefile shortcut for the Python side:

```bash
make install
```

## Quick Start

Start both backend and frontend dev servers:

```bash
./scripts/play.sh
```

Then open http://127.0.0.1:5173 in your browser.

Environment overrides:
- `BACKEND_HOST` / `BACKEND_PORT` — backend bind address (default `127.0.0.1:8000`)
- `FRONTEND_HOST` / `FRONTEND_PORT` — Vite dev server (default `127.0.0.1:5173`)
- `VITE_API_URL` — API base URL for the frontend

## Docker Deployment

Build and run the entire stack in a single container:

```bash
# Build
docker compose build

# Run
docker compose up -d

# Check health
curl http://localhost:8000/health
```

The Docker image uses a multi-stage build: Node.js builds the frontend, then
Python serves both the API and the static frontend assets. The container
exposes port 8000.

Environment variables for the container:
- `BACKEND_PORT` — port mapping (default `8000`)
- `SMART_SNAKE_CHECKPOINT_DIR` — path to AI model checkpoints (default `/app/checkpoints`)
- `SMART_SNAKE_SERVE_FRONTEND` — serve built frontend from `/` (default `1` in Docker)

To use pre-trained AI checkpoints, mount them into the container:

```bash
docker run -v ./checkpoints:/app/checkpoints -p 8000:8000 smart-snake
```

## API Reference

The backend auto-generates interactive API docs via FastAPI:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Key Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check with game stats |
| `POST` | `/games` | Create a game lobby |
| `GET` | `/games` | List active/waiting games |
| `GET` | `/games/{id}` | Get full game state |
| `POST` | `/games/{id}/join` | Join a waiting lobby |
| `POST` | `/games/{id}/start` | Start game (host only) |
| `POST` | `/games/{id}/leave` | Leave a waiting lobby |
| `WS` | `/games/{id}/play?token=` | Player WebSocket |
| `WS` | `/games/{id}/spectate` | Spectator WebSocket |

### Game Configuration

When creating a game (`POST /games`), you can configure:

| Field | Default | Range | Description |
|-------|---------|-------|-------------|
| `player_count` | 2 | 2–4 | Number of player slots |
| `grid_width` | auto | ≥ 4 | Grid width (auto-scales: 20/25/30) |
| `grid_height` | auto | ≥ 4 | Grid height (auto-scales: 20/25/30) |
| `tick_rate_ms` | 150 | 50–2000 | Milliseconds per game tick |
| `max_apples` | 3 | ≥ 1 | Maximum apples on the grid |
| `wall_mode` | `"death"` | `death`/`wrap` | Wall collision behavior |
| `dead_body_mode` | `"remove"` | `remove`/`obstacle` | Dead snake behavior |
| `ai_opponents` | `[]` | — | List of `{"difficulty": "..."}` |

AI difficulty tiers: `beginner`, `easy`, `medium`, `hard`, `impossible`.

## AI Training

### Train a Model

```bash
# Quick training run
./scripts/train.sh --num-games 200 --grid-width 20 --grid-height 20

# Full training with custom hyperparameters
smart-snake-train train \
    --num-games 5000 \
    --grid-width 20 \
    --grid-height 20 \
    --num-envs 4 \
    --batch-size 64 \
    --learning-rate 0.0001
```

Checkpoints are saved to `checkpoints/`. The best model (by win rate) is
automatically saved as `checkpoints/best_model.pt` and used by AI opponents
in the game server.

### Monitor Training

```bash
tensorboard --logdir runs/
```

### Export for Serving

```bash
smart-snake-train export --checkpoint checkpoints/best_model.pt --output model.pt
```

### Benchmark Throughput

```bash
smart-snake-train benchmark --num-games 100 --num-envs 4
```

### Difficulty Tiers

All difficulty tiers use the same trained checkpoint with varying random-action
injection rates:

| Tier | Random Action % | Description |
|------|----------------|-------------|
| Beginner | 80% | Mostly random moves |
| Easy | 50% | Half random, half learned |
| Medium | 20% | Occasional random moves |
| Hard | 5% | Nearly optimal play |
| Impossible | 0% | Pure model output |

## Development

### Validation

```bash
# Python lint + tests
make check

# Frontend lint + build
cd frontend && npm run lint && npm run build
```

### Test Suite

- 319+ tests with 93% coverage
- Covers: game engine, multiplayer, API, WebSocket, AI agent, training, integration
- Stress tests (deselected by default): `pytest -m stress`

### Code Style

- Python: ruff (line length 100, py310 target)
- Frontend: ESLint with react-hooks plugin
- Absolute imports only, no star imports

### Project Structure

```
├── src/smart_snake/          # Python package
│   ├── engine.py             # Single-player game engine
│   ├── multiplayer.py        # 2-4 player engine
│   ├── grid.py               # NumPy grid + cell types
│   ├── snake.py              # Snake + Direction
│   ├── apple.py              # Apple spawner
│   ├── server/               # FastAPI server
│   │   ├── app.py            # App factory
│   │   ├── routes.py         # REST endpoints
│   │   ├── websocket.py      # WebSocket handlers
│   │   ├── game_manager.py   # In-memory game registry
│   │   └── models.py         # Pydantic schemas
│   └── ai/                   # DQN reinforcement learning
│       ├── agent.py          # DQN agent (Double DQN + PER)
│       ├── networks.py       # CNN architectures
│       ├── train.py          # Self-play trainer
│       ├── environment.py    # Gym-compatible wrappers
│       ├── state.py          # 14-channel state encoder
│       ├── difficulty.py     # Difficulty tiers
│       ├── parallel.py       # Vectorized environments
│       ├── config.py         # Training/reward configs
│       ├── model_manager.py  # Checkpoint versioning
│       ├── benchmark.py      # Throughput benchmarking
│       ├── cli.py            # CLI entry point
│       └── replay_buffer.py  # Replay buffers
├── frontend/                 # React + TypeScript + Vite
│   ├── src/
│   │   ├── components/       # GameCanvas, Lobby, GameView, etc.
│   │   ├── hooks/            # useWebSocket, useKeyboard
│   │   ├── api/              # REST client
│   │   └── types/            # TypeScript interfaces
│   └── ...
├── tests/                    # pytest test suite
├── scripts/                  # play.sh, train.sh
├── Dockerfile                # Multi-stage build
├── docker-compose.yml        # Single-service deployment
└── Makefile                  # install, test, lint, fmt, check
```

## Contributing

1. Fork the repository and create a feature branch.
2. Install all dependencies: `make install && cd frontend && npm install`
3. Make your changes, following the existing code style.
4. Add tests for any behavioral changes.
5. Run the full validation suite: `make check && cd frontend && npm run lint && npm run build`
6. Open a pull request with a clear description of what and why.
