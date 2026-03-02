/** Types matching the server Pydantic models. */

export type GameStatus = "waiting" | "active" | "finished";

export type Direction = "up" | "down" | "left" | "right";

export type DifficultyTier =
  | "beginner"
  | "easy"
  | "medium"
  | "hard"
  | "impossible";

export interface GameSummary {
  game_id: string;
  status: GameStatus;
  player_count: number;
  max_players: number;
  tick_rate_ms: number;
  ai_count: number;
}

export interface PlayerInfo {
  snake_id: number;
  nickname: string;
  connected: boolean;
  is_ai: boolean;
}

export interface JoinResponse {
  game_id: string;
  snake_id: number;
  token: string;
  nickname: string;
}

export interface GameDetail {
  game_id: string;
  status: GameStatus;
  player_count: number;
  max_players: number;
  tick_rate_ms: number;
  ai_count: number;
  host_snake_id: number | null;
  players: PlayerInfo[];
  state?: GameState;
}

export interface SnakeState {
  body: [number, number][];
  direction: string;
  alive: boolean;
}

export interface PlayerState {
  snake_id: number;
  score: number;
  survival_ticks: number;
  alive: boolean;
}

export interface AppleState {
  positions: [number, number][];
  max_apples: number;
}

export interface GridState {
  width: number;
  height: number;
  wall_mode: string;
  cells: number[][];
}

export interface GameConfig {
  player_count: number;
  grid_width: number;
  grid_height: number;
  wall_mode: string;
  max_apples: number;
  initial_snake_length: number;
  dead_body_mode: string;
}

export interface GameState {
  tick: number;
  game_over: boolean;
  winner: number | null;
  grid: GridState;
  snakes: SnakeState[];
  players: PlayerState[];
  apples: AppleState;
  config: GameConfig;
}

export interface CreateGameOptions {
  player_count: number;
  grid_width?: number;
  grid_height?: number;
  wall_mode?: string;
  max_apples?: number;
  tick_rate_ms?: number;
  ai_opponents?: { difficulty: DifficultyTier }[];
}
