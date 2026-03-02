import type { GameState, SnakeState } from "../types/game";

const PLAYER_COLORS = ["#22d65a", "#3b9dff", "#ff9933", "#c56dff"];

interface ScoreOverlayProps {
  gameState: GameState;
  mySnakeId?: number;
  nicknames?: Record<number, string>;
}

export default function ScoreOverlay({
  gameState,
  mySnakeId,
  nicknames,
}: ScoreOverlayProps) {
  const { players, snakes } = gameState;

  return (
    <div className="score-overlay">
      {players.map((p) => {
        const snake: SnakeState | undefined = snakes[p.snake_id];
        const color = PLAYER_COLORS[p.snake_id % PLAYER_COLORS.length];
        const isMe = p.snake_id === mySnakeId;
        const name =
          nicknames?.[p.snake_id] ?? `Player ${p.snake_id + 1}`;

        return (
          <div
            key={p.snake_id}
            className={`score-item ${!p.alive ? "dead" : ""} ${isMe ? "me" : ""}`}
          >
            <span
              className="score-dot"
              style={{ backgroundColor: color }}
            />
            <span className="score-name">{name}</span>
            <span className="score-length">
              {snake ? snake.body.length : 0}
            </span>
            <span className="score-points">{p.score} pts</span>
            {!p.alive && <span className="score-dead">DEAD</span>}
          </div>
        );
      })}
    </div>
  );
}
