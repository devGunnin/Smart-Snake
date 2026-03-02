import { useEffect } from "react";
import type { Direction } from "../types/game";

const KEY_MAP: Record<string, Direction> = {
  ArrowUp: "up",
  ArrowDown: "down",
  ArrowLeft: "left",
  ArrowRight: "right",
  w: "up",
  W: "up",
  a: "left",
  A: "left",
  s: "down",
  S: "down",
  d: "right",
  D: "right",
};

/**
 * Listen for arrow key / WASD input and forward directions.
 * Active only when `enabled` is true.
 */
export function useKeyboard(
  onDirection: (dir: Direction) => void,
  enabled: boolean,
): void {
  useEffect(() => {
    if (!enabled) return;

    const handler = (e: KeyboardEvent) => {
      const dir = KEY_MAP[e.key];
      if (dir) {
        e.preventDefault();
        onDirection(dir);
      }
    };

    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onDirection, enabled]);
}
