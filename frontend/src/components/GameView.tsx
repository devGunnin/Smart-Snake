import { useCallback, useEffect, useMemo, useState } from "react";
import { getGame, playerWsUrl, startGame } from "../api/client";
import { useKeyboard } from "../hooks/useKeyboard";
import { useWebSocket } from "../hooks/useWebSocket";
import type {
  GameDetail,
  JoinResponse,
  PlayerInfo,
} from "../types/game";
import GameCanvas from "./GameCanvas";
import GameOver from "./GameOver";
import ScoreOverlay from "./ScoreOverlay";

const STATUS_LABELS: Record<string, string> = {
  connected: "Connected",
  connecting: "Connecting...",
  disconnected: "Disconnected",
  error: "Connection Error",
};

const STATUS_COLORS: Record<string, string> = {
  connected: "#22d65a",
  connecting: "#ff9933",
  disconnected: "#666",
  error: "#ff3355",
};

interface GameViewProps {
  joinInfo: JoinResponse;
  isHost: boolean;
  onBackToLobby: () => void;
}

export default function GameView({
  joinInfo,
  isHost,
  onBackToLobby,
}: GameViewProps) {
  const [detail, setDetail] = useState<GameDetail | null>(null);
  const [countdown, setCountdown] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const wsUrl = useMemo(
    () => playerWsUrl(joinInfo.game_id, joinInfo.token),
    [joinInfo],
  );

  const { status, gameState, sendDirection } = useWebSocket(wsUrl);

  // Poll game detail while waiting.
  useEffect(() => {
    if (gameState && !gameState.game_over) return;
    const poll = async () => {
      try {
        const d = await getGame(joinInfo.game_id);
        setDetail(d);
      } catch {
        // Ignore.
      }
    };
    poll();
    const id = setInterval(poll, 2000);
    return () => clearInterval(id);
  }, [joinInfo.game_id, gameState]);

  // Keyboard controls.
  const isPlaying =
    gameState !== null &&
    !gameState.game_over &&
    status === "connected";
  useKeyboard(sendDirection, isPlaying);

  // Build nickname map from detail.
  const nicknames = useMemo(() => {
    const map: Record<number, string> = {};
    if (detail) {
      detail.players.forEach((p: PlayerInfo) => {
        map[p.snake_id] = p.nickname;
      });
    }
    map[joinInfo.snake_id] = joinInfo.nickname;
    return map;
  }, [detail, joinInfo]);

  // Start game handler (host only).
  const handleStart = useCallback(async () => {
    setError(null);
    setCountdown(3);
    for (let i = 2; i >= 0; i--) {
      await new Promise((r) => setTimeout(r, 1000));
      setCountdown(i);
    }
    try {
      await startGame(joinInfo.game_id, joinInfo.token);
      setCountdown(null);
    } catch (err: unknown) {
      setCountdown(null);
      setError(
        err instanceof Error ? err.message : "Failed to start.",
      );
    }
  }, [joinInfo]);

  // Game over state.
  if (gameState?.game_over) {
    return (
      <div className="game-view">
        <GameCanvas
          gameState={gameState}
          mySnakeId={joinInfo.snake_id}
        />
        <GameOver
          gameState={gameState}
          nicknames={nicknames}
          onBackToLobby={onBackToLobby}
        />
      </div>
    );
  }

  // Active game.
  if (gameState && !gameState.game_over) {
    return (
      <div className="game-view">
        <div className="game-hud">
          <ScoreOverlay
            gameState={gameState}
            mySnakeId={joinInfo.snake_id}
            nicknames={nicknames}
          />
          <div className="connection-status">
            <span
              className="status-dot"
              style={{
                backgroundColor: STATUS_COLORS[status] ?? "#666",
              }}
            />
            {STATUS_LABELS[status] ?? status}
          </div>
        </div>
        <GameCanvas
          gameState={gameState}
          mySnakeId={joinInfo.snake_id}
        />
        <p className="controls-hint">
          Use Arrow Keys or WASD to move
        </p>
      </div>
    );
  }

  // Waiting room.
  const players = detail?.players ?? [];
  const waiting = !detail || detail.status === "waiting";

  return (
    <div className="waiting-room">
      <h2>
        Game{" "}
        <span className="game-id">
          {joinInfo.game_id.slice(0, 6)}
        </span>
      </h2>

      <div className="connection-status">
        <span
          className="status-dot"
          style={{
            backgroundColor: STATUS_COLORS[status] ?? "#666",
          }}
        />
        {STATUS_LABELS[status] ?? status}
      </div>

      {error && <div className="error-banner">{error}</div>}

      {countdown !== null && (
        <div className="countdown">
          {countdown > 0 ? countdown : "GO!"}
        </div>
      )}

      <div className="player-list">
        <h3>
          Players ({players.length}/{detail?.max_players ?? "?"})
        </h3>
        {players.map((p: PlayerInfo) => (
          <div key={p.snake_id} className="player-item">
            <span
              className="status-dot"
              style={{
                backgroundColor: p.connected || p.is_ai
                  ? "#22d65a"
                  : "#666",
              }}
            />
            <span>{p.nickname}</span>
            {p.is_ai && <span className="ai-badge">AI</span>}
            {p.snake_id === joinInfo.snake_id && (
              <span className="you-badge">You</span>
            )}
          </div>
        ))}
        {detail &&
          players.length < detail.max_players &&
          Array.from(
            { length: detail.max_players - players.length },
            (_, i) => (
              <div key={`empty-${i}`} className="player-item empty">
                <span className="status-dot" />
                <span className="muted">Waiting for player...</span>
              </div>
            ),
          )}
      </div>

      {isHost && waiting && countdown === null && (
        <button
          className="btn btn-primary"
          onClick={handleStart}
          disabled={players.length < 2}
        >
          Start Game
        </button>
      )}

      <button
        className="btn btn-ghost"
        onClick={onBackToLobby}
      >
        Leave
      </button>
    </div>
  );
}
