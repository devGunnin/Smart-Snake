import { useCallback, useEffect, useState } from "react";
import {
  createGame,
  joinGame,
  listGames,
} from "../api/client";
import type {
  CreateGameOptions,
  DifficultyTier,
  GameSummary,
  JoinResponse,
} from "../types/game";

const DIFFICULTIES: DifficultyTier[] = [
  "beginner",
  "easy",
  "medium",
  "hard",
  "impossible",
];
const POLL_INTERVAL = 3000;

interface LobbyProps {
  onJoined: (join: JoinResponse, isHost: boolean) => void;
}

export default function Lobby({ onJoined }: LobbyProps) {
  const [games, setGames] = useState<GameSummary[]>([]);
  const [showCreate, setShowCreate] = useState(false);
  const [nickname, setNickname] = useState("Player");
  const [error, setError] = useState<string | null>(null);
  const [joining, setJoining] = useState<string | null>(null);

  // Create-game form state.
  const [playerCount, setPlayerCount] = useState(2);
  const [gridSize, setGridSize] = useState(20);
  const [aiCount, setAiCount] = useState(0);
  const [aiDifficulty, setAiDifficulty] =
    useState<DifficultyTier>("medium");
  const [tickRate, setTickRate] = useState(200);

  const refreshGames = useCallback(async () => {
    try {
      const list = await listGames();
      setGames(list);
    } catch {
      // Silently retry on next poll.
    }
  }, []);

  useEffect(() => {
    refreshGames();
    const id = setInterval(refreshGames, POLL_INTERVAL);
    return () => clearInterval(id);
  }, [refreshGames]);

  const handleJoin = async (gameId: string) => {
    setError(null);
    setJoining(gameId);
    try {
      const res = await joinGame(gameId, nickname);
      onJoined(res, false);
    } catch (err: unknown) {
      setError(
        err instanceof Error ? err.message : "Failed to join.",
      );
      setJoining(null);
    }
  };

  const handleCreate = async () => {
    setError(null);
    const effectiveAi = Math.min(aiCount, playerCount - 1);
    const aiOpponents: { difficulty: DifficultyTier }[] = [];
    for (let i = 0; i < effectiveAi; i++) {
      aiOpponents.push({ difficulty: aiDifficulty });
    }
    const opts: CreateGameOptions = {
      player_count: playerCount,
      grid_width: gridSize,
      grid_height: gridSize,
      tick_rate_ms: tickRate,
      ai_opponents: aiOpponents,
    };
    try {
      const game = await createGame(opts);
      const res = await joinGame(game.game_id, nickname);
      onJoined(res, true);
    } catch (err: unknown) {
      setError(
        err instanceof Error ? err.message : "Failed to create game.",
      );
    }
  };

  const handleQuickPlay = async () => {
    setError(null);
    const opts: CreateGameOptions = {
      player_count: 2,
      ai_opponents: [{ difficulty: "medium" }],
    };
    try {
      const game = await createGame(opts);
      const res = await joinGame(game.game_id, nickname);
      onJoined(res, true);
    } catch (err: unknown) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to start quick play.",
      );
    }
  };

  const waitingGames = games.filter((g) => g.status === "waiting");

  return (
    <div className="lobby">
      <header className="lobby-header">
        <h1 className="logo">Smart Snake</h1>
        <p className="subtitle">Multiplayer AI Snake Game</p>
      </header>

      <div className="lobby-nickname">
        <label htmlFor="nickname">Nickname</label>
        <input
          id="nickname"
          type="text"
          value={nickname}
          maxLength={32}
          onChange={(e) => setNickname(e.target.value || "Player")}
        />
      </div>

      {error && <div className="error-banner">{error}</div>}

      <div className="lobby-actions">
        <button className="btn btn-primary" onClick={handleQuickPlay}>
          Quick Play vs AI
        </button>
        <button
          className="btn btn-secondary"
          onClick={() => setShowCreate(!showCreate)}
        >
          {showCreate ? "Cancel" : "Create Game"}
        </button>
      </div>

      {showCreate && (
        <div className="create-form card">
          <h3>New Game</h3>
          <div className="form-row">
            <label>Players</label>
            <select
              value={playerCount}
              onChange={(e) => {
                const v = Number(e.target.value);
                setPlayerCount(v);
                if (aiCount >= v) setAiCount(v - 1);
              }}
            >
              <option value={2}>2</option>
              <option value={3}>3</option>
              <option value={4}>4</option>
            </select>
          </div>
          <div className="form-row">
            <label>Grid Size</label>
            <select
              value={gridSize}
              onChange={(e) => setGridSize(Number(e.target.value))}
            >
              <option value={15}>15 x 15</option>
              <option value={20}>20 x 20</option>
              <option value={25}>25 x 25</option>
              <option value={30}>30 x 30</option>
            </select>
          </div>
          <div className="form-row">
            <label>AI Opponents</label>
            <select
              value={aiCount}
              onChange={(e) => setAiCount(Number(e.target.value))}
            >
              {Array.from({ length: playerCount }, (_, i) => (
                <option key={i} value={i}>
                  {i}
                </option>
              ))}
            </select>
          </div>
          {aiCount > 0 && (
            <div className="form-row">
              <label>AI Difficulty</label>
              <select
                value={aiDifficulty}
                onChange={(e) =>
                  setAiDifficulty(e.target.value as DifficultyTier)
                }
              >
                {DIFFICULTIES.map((d) => (
                  <option key={d} value={d}>
                    {d.charAt(0).toUpperCase() + d.slice(1)}
                  </option>
                ))}
              </select>
            </div>
          )}
          <div className="form-row">
            <label>Tick Rate (ms)</label>
            <input
              type="number"
              min={50}
              max={2000}
              step={50}
              value={tickRate}
              onChange={(e) => setTickRate(Number(e.target.value))}
            />
          </div>
          <button className="btn btn-primary" onClick={handleCreate}>
            Create & Join
          </button>
        </div>
      )}

      <div className="game-list">
        <h3>Open Games {waitingGames.length > 0 && `(${waitingGames.length})`}</h3>
        {waitingGames.length === 0 ? (
          <p className="muted">No games available. Create one!</p>
        ) : (
          <div className="game-list-items">
            {waitingGames.map((g) => (
              <div key={g.game_id} className="game-list-item card">
                <div className="game-list-info">
                  <span className="game-id">
                    {g.game_id.slice(0, 6)}
                  </span>
                  <span>
                    {g.player_count}/{g.max_players} players
                  </span>
                  {g.ai_count > 0 && (
                    <span className="ai-badge">
                      {g.ai_count} AI
                    </span>
                  )}
                  <span className="muted">{g.tick_rate_ms}ms</span>
                </div>
                <button
                  className="btn btn-secondary btn-sm"
                  disabled={
                    joining === g.game_id ||
                    g.player_count >= g.max_players
                  }
                  onClick={() => handleJoin(g.game_id)}
                >
                  {g.player_count >= g.max_players
                    ? "Full"
                    : "Join"}
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
