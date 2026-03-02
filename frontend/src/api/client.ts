/** REST API client for the Smart Snake server. */

import type {
  CreateGameOptions,
  GameDetail,
  GameSummary,
  JoinResponse,
} from "../types/game";

const API_BASE =
  import.meta.env.VITE_API_URL ?? "http://localhost:8000";

async function request<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export function listGames(): Promise<GameSummary[]> {
  return request<GameSummary[]>("/games");
}

export function getGame(gameId: string): Promise<GameDetail> {
  return request<GameDetail>(`/games/${gameId}`);
}

export function createGame(
  opts: CreateGameOptions,
): Promise<GameSummary> {
  return request<GameSummary>("/games", {
    method: "POST",
    body: JSON.stringify(opts),
  });
}

export function joinGame(
  gameId: string,
  nickname: string,
): Promise<JoinResponse> {
  return request<JoinResponse>(`/games/${gameId}/join`, {
    method: "POST",
    body: JSON.stringify({ nickname }),
  });
}

export function startGame(
  gameId: string,
  token: string,
): Promise<{ status: string; game_id: string }> {
  return request(`/games/${gameId}/start`, {
    method: "POST",
    body: JSON.stringify({ token }),
  });
}

/** Build the WebSocket URL for a player connection. */
export function playerWsUrl(
  gameId: string,
  token: string,
): string {
  const base = API_BASE.replace(/^http/, "ws");
  return `${base}/games/${gameId}/play?token=${token}`;
}

/** Build the WebSocket URL for a spectator connection. */
export function spectatorWsUrl(gameId: string): string {
  const base = API_BASE.replace(/^http/, "ws");
  return `${base}/games/${gameId}/spectate`;
}
