import { useCallback, useEffect, useRef, useState } from "react";
import type { Direction, GameState } from "../types/game";

export type ConnectionStatus =
  | "connecting"
  | "connected"
  | "disconnected"
  | "error";

interface UseWebSocketReturn {
  status: ConnectionStatus;
  gameState: GameState | null;
  sendDirection: (dir: Direction) => void;
}

const RECONNECT_DELAY_MS = 2000;
const MAX_RECONNECTS = 5;

export function useWebSocket(
  url: string | null,
): UseWebSocketReturn {
  const [status, setStatus] =
    useState<ConnectionStatus>("disconnected");
  const [gameState, setGameState] = useState<GameState | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectCountRef = useRef(0);
  const reconnectTimerRef = useRef<number | null>(null);
  const connectRef = useRef<() => void>(() => {});

  const connect = useCallback(() => {
    if (!url) return;

    setStatus("connecting");
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus("connected");
      reconnectCountRef.current = 0;
    };

    ws.onmessage = (event) => {
      try {
        const state = JSON.parse(event.data) as GameState;
        setGameState(state);
      } catch {
        // Ignore non-JSON messages.
      }
    };

    ws.onclose = (event) => {
      wsRef.current = null;
      if (event.code === 1000 || event.code === 4008) {
        setStatus("disconnected");
        return;
      }
      if (reconnectCountRef.current < MAX_RECONNECTS) {
        setStatus("connecting");
        reconnectCountRef.current += 1;
        reconnectTimerRef.current = window.setTimeout(
          () => {
            connectRef.current();
          },
          RECONNECT_DELAY_MS,
        );
      } else {
        setStatus("error");
      }
    };

    ws.onerror = () => {
      setStatus("error");
    };
  }, [url]);

  useEffect(() => {
    connectRef.current = connect;
  }, [connect]);

  useEffect(() => {
    const initialConnectId = window.setTimeout(() => {
      connectRef.current();
    }, 0);
    return () => {
      clearTimeout(initialConnectId);
      if (reconnectTimerRef.current !== null) {
        clearTimeout(reconnectTimerRef.current);
      }
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [connect]);

  const sendDirection = useCallback((dir: Direction) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ direction: dir }));
    }
  }, []);

  return { status, gameState, sendDirection };
}
