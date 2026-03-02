import type { GameState } from "../types/game";

const PLAYER_COLORS = ["#22d65a", "#3b9dff", "#ff9933", "#c56dff"];

interface GameOverProps {
  gameState: GameState;
  nicknames?: Record<number, string>;
  onBackToLobby: () => void;
}

export default function GameOver({
  gameState,
  nicknames,
  onBackToLobby,
}: GameOverProps) {
  const { winner, players } = gameState;

  const sorted = [...players].sort((a, b) => {
    if (a.alive !== b.alive) return a.alive ? -1 : 1;
    if (a.score !== b.score) return b.score - a.score;
    return b.survival_ticks - a.survival_ticks;
  });

  const winnerName =
    winner !== null
      ? (nicknames?.[winner] ?? `Player ${winner + 1}`)
      : null;
  const winnerColor =
    winner !== null
      ? PLAYER_COLORS[winner % PLAYER_COLORS.length]
      : "#ccc";

  return (
    <div className="gameover-overlay">
      <div className="gameover-card">
        <h2 className="gameover-title">Game Over</h2>

        {winnerName ? (
          <div className="gameover-winner">
            <span style={{ color: winnerColor }}>
              {winnerName}
            </span>{" "}
            wins!
          </div>
        ) : (
          <div className="gameover-winner">It's a tie!</div>
        )}

        <table className="gameover-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Player</th>
              <th>Score</th>
              <th>Survived</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((p, i) => {
              const name =
                nicknames?.[p.snake_id] ??
                `Player ${p.snake_id + 1}`;
              const color =
                PLAYER_COLORS[p.snake_id % PLAYER_COLORS.length];
              return (
                <tr key={p.snake_id}>
                  <td>{i + 1}</td>
                  <td style={{ color }}>{name}</td>
                  <td>{p.score}</td>
                  <td>{p.survival_ticks} ticks</td>
                </tr>
              );
            })}
          </tbody>
        </table>

        <button className="btn btn-primary" onClick={onBackToLobby}>
          Back to Lobby
        </button>
      </div>
    </div>
  );
}
