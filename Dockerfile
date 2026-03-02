# -- Stage 1: build frontend -------------------------------------------------
FROM node:20-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# -- Stage 2: Python runtime -------------------------------------------------
FROM python:3.12-slim AS runtime
LABEL maintainer="devGunnin"
LABEL description="Smart Snake — multiplayer snake game with DQN AI"

WORKDIR /app

# Install Python package (server + ai extras).
COPY pyproject.toml ./
COPY src/ src/
RUN pip install --no-cache-dir ".[server,ai]"

# Copy built frontend assets.
COPY --from=frontend-build /app/frontend/dist frontend/dist

# Copy helper scripts and other root-level files.
COPY scripts/ scripts/
COPY Makefile ./

# Pre-create the checkpoints directory so the container can write to it.
RUN mkdir -p checkpoints

ENV BACKEND_HOST=0.0.0.0
ENV BACKEND_PORT=8000
ENV SMART_SNAKE_SERVE_FRONTEND=1
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "smart_snake.server.app:create_app", \
     "--factory", \
     "--host", "0.0.0.0", \
     "--port", "8000"]
