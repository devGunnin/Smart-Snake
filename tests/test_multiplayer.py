"""Tests for the MultiplayerEngine module."""

import json

import pytest

from smart_snake.grid import CellType, WallMode
from smart_snake.multiplayer import (
    DeadBodyMode,
    MatchConfig,
    MultiplayerEngine,
    PlayerState,
)
from smart_snake.snake import Direction

# ---------------------------------------------------------------------------
# MatchConfig validation
# ---------------------------------------------------------------------------


class TestMatchConfig:
    def test_defaults(self):
        cfg = MatchConfig()
        assert cfg.player_count == 2
        assert cfg.effective_width == 20
        assert cfg.effective_height == 20

    def test_scaling_3_players(self):
        cfg = MatchConfig(player_count=3)
        assert cfg.effective_width == 25
        assert cfg.effective_height == 25

    def test_scaling_4_players(self):
        cfg = MatchConfig(player_count=4)
        assert cfg.effective_width == 30
        assert cfg.effective_height == 30

    def test_custom_grid_overrides_default(self):
        cfg = MatchConfig(player_count=2, grid_width=40, grid_height=40)
        assert cfg.effective_width == 40
        assert cfg.effective_height == 40

    def test_invalid_player_count_low(self):
        with pytest.raises(ValueError, match="player_count"):
            MatchConfig(player_count=1)

    def test_invalid_player_count_high(self):
        with pytest.raises(ValueError, match="player_count"):
            MatchConfig(player_count=5)

    def test_invalid_max_apples(self):
        with pytest.raises(ValueError, match="max_apples"):
            MatchConfig(max_apples=0)

    def test_invalid_initial_length(self):
        with pytest.raises(ValueError, match="initial_snake_length"):
            MatchConfig(initial_snake_length=0)

    def test_initial_length_must_fit_spawn_layout(self):
        with pytest.raises(ValueError, match="initial_snake_length"):
            MatchConfig(
                player_count=2, grid_width=20, grid_height=20, initial_snake_length=6,
            )

    def test_small_grid_rejected_when_spawn_does_not_fit(self):
        with pytest.raises(ValueError, match="initial_snake_length"):
            MatchConfig(
                player_count=2, grid_width=5, grid_height=5, initial_snake_length=3,
            )


# ---------------------------------------------------------------------------
# Engine initialization
# ---------------------------------------------------------------------------


class TestMultiplayerInit:
    def test_two_player_init(self):
        engine = MultiplayerEngine(MatchConfig(player_count=2, seed=0))
        assert len(engine.snakes) == 2
        assert len(engine.players) == 2
        assert not engine.game_over
        assert engine.tick == 0

    def test_three_player_init(self):
        engine = MultiplayerEngine(MatchConfig(player_count=3, seed=0))
        assert len(engine.snakes) == 3
        assert all(s.alive for s in engine.snakes)

    def test_four_player_init(self):
        engine = MultiplayerEngine(MatchConfig(player_count=4, seed=0))
        assert len(engine.snakes) == 4
        assert all(s.alive for s in engine.snakes)

    def test_snakes_spawn_spread_apart(self):
        engine = MultiplayerEngine(MatchConfig(player_count=4, seed=0))
        heads = [s.head for s in engine.snakes]
        # All heads should be at distinct positions.
        assert len(set(heads)) == 4
        # Heads should be in different quadrants.
        h, w = engine.grid.height, engine.grid.width
        mid_r, mid_c = h // 2, w // 2
        quadrants = set()
        for r, c in heads:
            quadrants.add((r < mid_r, c < mid_c))
        assert len(quadrants) == 4

    def test_apples_spawned_on_init(self):
        cfg = MatchConfig(player_count=2, max_apples=3, seed=0)
        engine = MultiplayerEngine(cfg)
        assert len(engine.apple_spawner.positions) == 3

    def test_default_config_used_when_none(self):
        engine = MultiplayerEngine()
        assert len(engine.snakes) == 2


# ---------------------------------------------------------------------------
# Direction input
# ---------------------------------------------------------------------------


class TestMultiplayerDirection:
    def test_set_direction_valid(self):
        engine = MultiplayerEngine(MatchConfig(seed=0))
        engine.set_direction(0, Direction.UP)
        engine.step()
        # Snake 0 starts facing RIGHT at (5, 5); UP moves row from 5 to 4.
        assert engine.snakes[0].direction == Direction.UP

    def test_set_direction_ignores_reverse(self):
        engine = MultiplayerEngine(MatchConfig(seed=0))
        # Snake 0 starts facing RIGHT; LEFT is a reversal.
        engine.set_direction(0, Direction.LEFT)
        engine.step()
        assert engine.snakes[0].direction == Direction.RIGHT

    def test_set_direction_invalid_id(self):
        engine = MultiplayerEngine(MatchConfig(seed=0))
        with pytest.raises(ValueError, match="snake_id"):
            engine.set_direction(5, Direction.UP)

    def test_set_direction_dead_snake_ignored(self):
        engine = MultiplayerEngine(MatchConfig(seed=0))
        engine.snakes[0].alive = False
        engine.set_direction(0, Direction.UP)
        # No error, just ignored.
        assert engine._pending_directions[0] is None

    def test_only_first_direction_per_tick_accepted(self):
        engine = MultiplayerEngine(MatchConfig(seed=0))
        engine.set_direction(0, Direction.UP)
        engine.set_direction(0, Direction.DOWN)
        engine.step()
        assert engine.snakes[0].direction == Direction.UP


# ---------------------------------------------------------------------------
# Wall collisions
# ---------------------------------------------------------------------------


class TestMultiplayerWallCollision:
    def test_death_on_wall(self):
        cfg = MatchConfig(
            player_count=2, grid_width=10, grid_height=10, seed=42,
        )
        engine = MultiplayerEngine(cfg)
        # Drive snake 0 straight into the wall.
        for _ in range(20):
            engine.step()
            if not engine.snakes[0].alive:
                break
        assert not engine.snakes[0].alive

    def test_wrap_mode_wraps_around(self):
        cfg = MatchConfig(
            player_count=2,
            grid_width=10,
            grid_height=10,
            wall_mode=WallMode.WRAP,
            seed=42,
        )
        engine = MultiplayerEngine(cfg)
        # Snake 0 starts at col 2 facing RIGHT on a 10-wide grid.
        # Drive it to the boundary and beyond to confirm wrapping.
        for _ in range(20):
            engine.step()
            if engine.game_over:
                break
        # The snake should have wrapped (not died from walls).
        # It may die from self-collision eventually, but tick > 7
        # proves it crossed the boundary without instant death.
        assert engine.tick > 7


# ---------------------------------------------------------------------------
# Self-collision
# ---------------------------------------------------------------------------


class TestMultiplayerSelfCollision:
    def test_dies_on_self_collision(self):
        cfg = MatchConfig(
            player_count=2, grid_width=20, grid_height=20, seed=42,
        )
        engine = MultiplayerEngine(cfg)
        sid = 0
        # Grow snake 0 by placing apples directly ahead.
        for _ in range(6):
            nr, nc = engine.snakes[sid].next_head()
            engine.grid.set(nr, nc, CellType.APPLE)
            if (nr, nc) not in engine.apple_spawner.positions:
                engine.apple_spawner.positions.append((nr, nc))
            engine.step()

        # Now turn into itself: RIGHT -> DOWN -> LEFT -> UP.
        engine.set_direction(sid, Direction.DOWN)
        engine.step()
        engine.set_direction(sid, Direction.LEFT)
        engine.step()
        engine.set_direction(sid, Direction.UP)
        engine.step()
        assert not engine.snakes[sid].alive


# ---------------------------------------------------------------------------
# Head-to-head collision
# ---------------------------------------------------------------------------


class TestHeadToHeadCollision:
    def test_both_die_on_head_to_head(self):
        cfg = MatchConfig(
            player_count=2, grid_width=20, grid_height=20, seed=0,
        )
        engine = MultiplayerEngine(cfg)
        s0, s1 = engine.snakes[0], engine.snakes[1]

        # Position snakes to collide head-on.
        # Clear grid and manually place snakes facing each other.
        engine.grid.clear()
        s0.body.clear()
        s0.body.extend([(5, 4), (5, 3), (5, 2)])
        s0.direction = Direction.RIGHT
        s1.body.clear()
        s1.body.extend([(5, 6), (5, 7), (5, 8)])
        s1.direction = Direction.LEFT

        for r, c in s0.body:
            engine.grid.set(r, c, CellType.SNAKE)
        for r, c in s1.body:
            engine.grid.set(r, c, CellType.SNAKE)

        # Both heads will move to (5, 5) — head-to-head collision.
        engine.step()
        assert not s0.alive
        assert not s1.alive
        assert engine.game_over


# ---------------------------------------------------------------------------
# Head-to-body collision
# ---------------------------------------------------------------------------


class TestHeadToBodyCollision:
    def test_head_snake_dies_body_snake_survives(self):
        cfg = MatchConfig(
            player_count=2, grid_width=20, grid_height=20, seed=0,
        )
        engine = MultiplayerEngine(cfg)
        s0, s1 = engine.snakes[0], engine.snakes[1]

        # Set up: snake 0 runs into snake 1's body.
        engine.grid.clear()
        s0.body.clear()
        s0.body.extend([(4, 5), (4, 4), (4, 3)])
        s0.direction = Direction.DOWN

        s1.body.clear()
        s1.body.extend([(5, 4), (5, 5), (5, 6)])
        s1.direction = Direction.LEFT

        for r, c in s0.body:
            engine.grid.set(r, c, CellType.SNAKE)
        for r, c in s1.body:
            engine.grid.set(r, c, CellType.SNAKE)

        # Snake 0 head moves to (5, 5) which is snake 1's body.
        engine.step()
        assert not s0.alive
        assert s1.alive


# ---------------------------------------------------------------------------
# Dead body modes
# ---------------------------------------------------------------------------


class TestDeadBodyMode:
    def test_remove_mode_clears_body(self):
        cfg = MatchConfig(
            player_count=2,
            grid_width=20,
            grid_height=20,
            dead_body_mode=DeadBodyMode.REMOVE,
            seed=0,
        )
        engine = MultiplayerEngine(cfg)
        s0 = engine.snakes[0]
        body_cells = list(s0.body)

        engine._kill_snake(0)
        for r, c in body_cells:
            assert engine.grid.get(r, c) == CellType.EMPTY

    def test_obstacle_mode_converts_body(self):
        cfg = MatchConfig(
            player_count=2,
            grid_width=20,
            grid_height=20,
            dead_body_mode=DeadBodyMode.OBSTACLE,
            seed=0,
        )
        engine = MultiplayerEngine(cfg)
        s0 = engine.snakes[0]
        body_cells = list(s0.body)

        engine._kill_snake(0)
        for r, c in body_cells:
            assert engine.grid.get(r, c) == CellType.OBSTACLE

    def test_obstacle_mode_blocks_movement(self):
        cfg = MatchConfig(
            player_count=2,
            grid_width=20,
            grid_height=20,
            dead_body_mode=DeadBodyMode.OBSTACLE,
            seed=0,
        )
        engine = MultiplayerEngine(cfg)
        s0, s1 = engine.snakes[0], engine.snakes[1]

        engine.grid.clear()
        s0.body.clear()
        s0.body.extend([(5, 5), (5, 6), (5, 7)])
        for r, c in s0.body:
            engine.grid.set(r, c, CellType.SNAKE)
        engine._kill_snake(0)

        s1.body.clear()
        s1.body.extend([(5, 4), (5, 3), (5, 2)])
        s1.direction = Direction.RIGHT
        for r, c in s1.body:
            engine.grid.set(r, c, CellType.SNAKE)

        engine.step()
        assert not s1.alive


# ---------------------------------------------------------------------------
# Win conditions
# ---------------------------------------------------------------------------


class TestWinConditions:
    def test_last_snake_standing_wins(self):
        cfg = MatchConfig(
            player_count=2, grid_width=20, grid_height=20, seed=0,
        )
        engine = MultiplayerEngine(cfg)
        # Kill snake 1 manually.
        engine._kill_snake(1)
        engine.tick += 1
        # Step should detect game over.
        engine.step()
        # Only snake 0 is alive => game over, snake 0 wins.
        assert engine.game_over
        assert engine.winner == 0

    def test_mutual_death_tiebreak_by_length(self):
        cfg = MatchConfig(
            player_count=2, grid_width=20, grid_height=20, seed=0,
        )
        engine = MultiplayerEngine(cfg)
        s0, s1 = engine.snakes[0], engine.snakes[1]

        # Make snake 0 longer than snake 1.
        s0.body.append((0, 0))

        # Set up head-to-head collision.
        engine.grid.clear()
        s0.body.clear()
        s0.body.extend([(5, 4), (5, 3), (5, 2), (5, 1)])
        s0.direction = Direction.RIGHT
        s1.body.clear()
        s1.body.extend([(5, 6), (5, 7), (5, 8)])
        s1.direction = Direction.LEFT

        for r, c in s0.body:
            engine.grid.set(r, c, CellType.SNAKE)
        for r, c in s1.body:
            engine.grid.set(r, c, CellType.SNAKE)

        engine.step()
        assert engine.game_over
        # Snake 0 has 4 segments, snake 1 has 3 => snake 0 wins tiebreak.
        assert engine.winner == 0

    def test_mutual_death_exact_tie_no_winner(self):
        cfg = MatchConfig(
            player_count=2, grid_width=20, grid_height=20, seed=0,
        )
        engine = MultiplayerEngine(cfg)

        # Set up equal-length head-to-head.
        engine.grid.clear()
        s0, s1 = engine.snakes[0], engine.snakes[1]
        s0.body.clear()
        s0.body.extend([(5, 4), (5, 3), (5, 2)])
        s0.direction = Direction.RIGHT
        s1.body.clear()
        s1.body.extend([(5, 6), (5, 7), (5, 8)])
        s1.direction = Direction.LEFT

        for r, c in s0.body:
            engine.grid.set(r, c, CellType.SNAKE)
        for r, c in s1.body:
            engine.grid.set(r, c, CellType.SNAKE)

        engine.step()
        assert engine.game_over
        assert engine.winner is None

    def test_three_player_game_ends_with_one(self):
        cfg = MatchConfig(player_count=3, grid_width=25, grid_height=25, seed=0)
        engine = MultiplayerEngine(cfg)
        engine._kill_snake(1)
        engine._kill_snake(2)
        # Trigger win check via step.
        engine.step()
        assert engine.game_over
        assert engine.winner == 0


# ---------------------------------------------------------------------------
# Games run to completion (2, 3, 4 players)
# ---------------------------------------------------------------------------


class TestGamesToCompletion:
    def test_two_player_game_completes(self):
        cfg = MatchConfig(
            player_count=2, grid_width=15, grid_height=15, seed=7,
        )
        engine = MultiplayerEngine(cfg)
        for _ in range(500):
            engine.step()
            if engine.game_over:
                break
        assert engine.game_over
        assert engine.tick > 0

    def test_three_player_game_completes(self):
        cfg = MatchConfig(
            player_count=3, grid_width=15, grid_height=15, seed=7,
        )
        engine = MultiplayerEngine(cfg)
        for _ in range(500):
            engine.step()
            if engine.game_over:
                break
        assert engine.game_over

    def test_four_player_game_completes(self):
        cfg = MatchConfig(
            player_count=4, grid_width=15, grid_height=15, seed=7,
        )
        engine = MultiplayerEngine(cfg)
        for _ in range(500):
            engine.step()
            if engine.game_over:
                break
        assert engine.game_over


# ---------------------------------------------------------------------------
# Apple consumption
# ---------------------------------------------------------------------------


class TestMultiplayerApples:
    def test_score_increases_on_apple(self):
        cfg = MatchConfig(
            player_count=2, grid_width=20, grid_height=20, seed=0,
        )
        engine = MultiplayerEngine(cfg)
        # Place apple directly ahead of snake 0.
        nr, nc = engine.snakes[0].next_head()
        engine.grid.set(nr, nc, CellType.APPLE)
        engine.apple_spawner.positions.append((nr, nc))

        engine.step()
        assert engine.players[0].score == 1
        assert len(engine.snakes[0].body) == 4


# ---------------------------------------------------------------------------
# Per-snake scoring
# ---------------------------------------------------------------------------


class TestPlayerState:
    def test_initial_state(self):
        ps = PlayerState(0)
        assert ps.snake_id == 0
        assert ps.score == 0
        assert ps.survival_ticks == 0
        assert ps.alive

    def test_to_dict(self):
        ps = PlayerState(1)
        ps.score = 5
        ps.survival_ticks = 42
        d = ps.to_dict()
        assert d["snake_id"] == 1
        assert d["score"] == 5
        assert d["survival_ticks"] == 42
        assert d["alive"]

    def test_survival_ticks_updated(self):
        cfg = MatchConfig(
            player_count=2, grid_width=20, grid_height=20, seed=0,
        )
        engine = MultiplayerEngine(cfg)
        engine.step()
        engine.step()
        for p in engine.players:
            if p.alive:
                assert p.survival_ticks == 2


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestMultiplayerSerialization:
    def test_state_is_json_serializable(self):
        cfg = MatchConfig(player_count=2, seed=42)
        engine = MultiplayerEngine(cfg)
        engine.step()
        state = engine.get_state()
        serialized = json.dumps(state)
        assert isinstance(serialized, str)

    def test_state_structure(self):
        engine = MultiplayerEngine(MatchConfig(seed=0))
        state = engine.get_state()
        assert "tick" in state
        assert "game_over" in state
        assert "winner" in state
        assert "grid" in state
        assert "snakes" in state
        assert "players" in state
        assert "apples" in state
        assert "config" in state

    def test_state_includes_all_players(self):
        cfg = MatchConfig(player_count=4, seed=0)
        engine = MultiplayerEngine(cfg)
        state = engine.get_state()
        assert len(state["snakes"]) == 4
        assert len(state["players"]) == 4

    def test_config_in_state(self):
        cfg = MatchConfig(
            player_count=3,
            grid_width=25,
            grid_height=25,
            max_apples=5,
            initial_snake_length=4,
            dead_body_mode=DeadBodyMode.OBSTACLE,
        )
        engine = MultiplayerEngine(cfg)
        state = engine.get_state()
        sc = state["config"]
        assert sc["player_count"] == 3
        assert sc["grid_width"] == 25
        assert sc["grid_height"] == 25
        assert sc["max_apples"] == 5
        assert sc["initial_snake_length"] == 4
        assert sc["dead_body_mode"] == "obstacle"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestMultiplayerDeterminism:
    def test_same_seed_same_outcome(self):
        actions = [
            (0, Direction.UP), (1, Direction.DOWN),
            (0, Direction.UP), (1, Direction.DOWN),
            (0, Direction.RIGHT), (1, Direction.LEFT),
        ]
        state_a = self._run_game(seed=123, actions=actions)
        state_b = self._run_game(seed=123, actions=actions)
        assert state_a == state_b

    def test_different_seeds_differ(self):
        actions = [(0, Direction.UP)] * 3
        state_a = self._run_game(seed=1, actions=actions)
        state_b = self._run_game(seed=2, actions=actions)
        assert state_a["apples"] != state_b["apples"]

    @staticmethod
    def _run_game(
        seed: int, actions: list[tuple[int, Direction]],
    ) -> dict:
        cfg = MatchConfig(player_count=2, grid_width=20, grid_height=20, seed=seed)
        engine = MultiplayerEngine(cfg)
        for snake_id, direction in actions:
            engine.set_direction(snake_id, direction)
            engine.step()
        return engine.get_state()


# ---------------------------------------------------------------------------
# Game-over halts further ticks
# ---------------------------------------------------------------------------


class TestGameOverHaltsTicks:
    def test_step_noop_after_game_over(self):
        engine = MultiplayerEngine(MatchConfig(seed=0))
        engine.game_over = True
        tick_before = engine.tick
        engine.step()
        assert engine.tick == tick_before
