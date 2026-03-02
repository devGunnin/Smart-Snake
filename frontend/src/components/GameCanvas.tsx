import {
  useCallback,
  useEffect,
  useRef,
  type CSSProperties,
} from "react";
import type { GameState } from "../types/game";

/** Per-player snake colours: head, body. */
const SNAKE_COLORS: [string, string][] = [
  ["#22d65a", "#1aad48"], // green
  ["#3b9dff", "#2b7ad9"], // blue
  ["#ff9933", "#d97d1f"], // orange
  ["#c56dff", "#a04dd9"], // purple
];

const APPLE_COLOR = "#ff3355";
const APPLE_GLOW = "#ff335544";
const OBSTACLE_COLOR = "#444c56";
const GRID_BG = "#0d1117";
const GRID_LINE = "#161b22";
const DEAD_SNAKE_ALPHA = 0.35;

interface GameCanvasProps {
  gameState: GameState;
  mySnakeId?: number;
  style?: CSSProperties;
}

export default function GameCanvas({
  gameState,
  mySnakeId,
  style,
}: GameCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animFrameRef = useRef<number>(0);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const { grid, snakes, apples } = gameState;
    const { width: cols, height: rows } = grid;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const cw = rect.width;
    const ch = rect.height;

    canvas.width = cw * dpr;
    canvas.height = ch * dpr;
    ctx.scale(dpr, dpr);

    const cellW = cw / cols;
    const cellH = ch / rows;

    // Background.
    ctx.fillStyle = GRID_BG;
    ctx.fillRect(0, 0, cw, ch);

    // Grid lines.
    ctx.strokeStyle = GRID_LINE;
    ctx.lineWidth = 1;
    for (let c = 1; c < cols; c++) {
      const x = c * cellW;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, ch);
      ctx.stroke();
    }
    for (let r = 1; r < rows; r++) {
      const y = r * cellH;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(cw, y);
      ctx.stroke();
    }

    // Obstacles (from grid cells == 3).
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        if (grid.cells[r][c] === 3) {
          ctx.fillStyle = OBSTACLE_COLOR;
          ctx.fillRect(
            c * cellW + 1,
            r * cellH + 1,
            cellW - 2,
            cellH - 2,
          );
        }
      }
    }

    // Apples.
    for (const [r, c] of apples.positions) {
      const cx = c * cellW + cellW / 2;
      const cy = r * cellH + cellH / 2;
      const radius = Math.min(cellW, cellH) * 0.38;

      // Glow.
      ctx.fillStyle = APPLE_GLOW;
      ctx.beginPath();
      ctx.arc(cx, cy, radius * 1.6, 0, Math.PI * 2);
      ctx.fill();

      // Apple.
      ctx.fillStyle = APPLE_COLOR;
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.fill();
    }

    // Snakes.
    for (let sid = 0; sid < snakes.length; sid++) {
      const snake = snakes[sid];
      if (!snake.body || snake.body.length === 0) continue;

      const [headColor, bodyColor] =
        SNAKE_COLORS[sid % SNAKE_COLORS.length];
      const alpha = snake.alive ? 1.0 : DEAD_SNAKE_ALPHA;

      ctx.globalAlpha = alpha;

      // Body segments.
      for (let i = 1; i < snake.body.length; i++) {
        const [br, bc] = snake.body[i];
        ctx.fillStyle = bodyColor;
        const pad = 1;
        ctx.beginPath();
        ctx.roundRect(
          bc * cellW + pad,
          br * cellH + pad,
          cellW - pad * 2,
          cellH - pad * 2,
          3,
        );
        ctx.fill();
      }

      // Head.
      const [hr, hc] = snake.body[0];
      ctx.fillStyle = headColor;
      ctx.beginPath();
      ctx.roundRect(
        hc * cellW + 0.5,
        hr * cellH + 0.5,
        cellW - 1,
        cellH - 1,
        4,
      );
      ctx.fill();

      // Eyes on head.
      if (snake.alive) {
        const eyeR = Math.min(cellW, cellH) * 0.12;
        ctx.fillStyle = "#fff";
        const ecx = hc * cellW + cellW / 2;
        const ecy = hr * cellH + cellH / 2;
        const offset = Math.min(cellW, cellH) * 0.18;

        let e1x = ecx - offset,
          e1y = ecy - offset;
        let e2x = ecx + offset,
          e2y = ecy - offset;
        if (snake.direction === "down") {
          e1y = ecy + offset;
          e2y = ecy + offset;
        } else if (snake.direction === "left") {
          e1x = ecx - offset;
          e1y = ecy - offset;
          e2x = ecx - offset;
          e2y = ecy + offset;
        } else if (snake.direction === "right") {
          e1x = ecx + offset;
          e1y = ecy - offset;
          e2x = ecx + offset;
          e2y = ecy + offset;
        }

        ctx.beginPath();
        ctx.arc(e1x, e1y, eyeR, 0, Math.PI * 2);
        ctx.fill();
        ctx.beginPath();
        ctx.arc(e2x, e2y, eyeR, 0, Math.PI * 2);
        ctx.fill();

        // Pupils.
        ctx.fillStyle = "#111";
        const pupR = eyeR * 0.55;
        ctx.beginPath();
        ctx.arc(e1x, e1y, pupR, 0, Math.PI * 2);
        ctx.fill();
        ctx.beginPath();
        ctx.arc(e2x, e2y, pupR, 0, Math.PI * 2);
        ctx.fill();
      }

      // Death X marker.
      if (!snake.alive && snake.body.length > 0) {
        const [dr, dc] = snake.body[0];
        const mx = dc * cellW + cellW / 2;
        const my = dr * cellH + cellH / 2;
        const ms = Math.min(cellW, cellH) * 0.25;
        ctx.strokeStyle = "#ff3355";
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(mx - ms, my - ms);
        ctx.lineTo(mx + ms, my + ms);
        ctx.moveTo(mx + ms, my - ms);
        ctx.lineTo(mx - ms, my + ms);
        ctx.stroke();
      }

      ctx.globalAlpha = 1.0;
    }

    // Highlight own snake head with a ring.
    if (
      mySnakeId !== undefined &&
      mySnakeId < snakes.length &&
      snakes[mySnakeId].alive &&
      snakes[mySnakeId].body.length > 0
    ) {
      const [hr, hc] = snakes[mySnakeId].body[0];
      const cx = hc * cellW + cellW / 2;
      const cy = hr * cellH + cellH / 2;
      const r = Math.min(cellW, cellH) * 0.48;
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.stroke();
    }
  }, [gameState, mySnakeId]);

  useEffect(() => {
    const tick = () => {
      draw();
      animFrameRef.current = requestAnimationFrame(tick);
    };
    animFrameRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animFrameRef.current);
  }, [draw]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        width: "100%",
        maxWidth: 700,
        aspectRatio: "1",
        borderRadius: 8,
        border: "2px solid #30363d",
        ...style,
      }}
    />
  );
}
