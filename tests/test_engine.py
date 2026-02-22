"""Tests for the GameEngine module."""

import json

from smart_snake.engine import GameEngine
from smart_snake.grid import WallMode
from smart_snake.snake import Direction


class TestEngineInit:
    def test_default_init(self):
        engine = GameEngine(seed=0)
        assert engine.score == 0
        assert engine.tick == 0
        assert not engine.game_over
        assert engine.snake.alive

    def test_snake_starts_center(self):
        engine = GameEngine(width=20, height=20, seed=0)
        assert engine.snake.head == (10, 10)

    def test_apple_spawned_on_init(self):
        engine = GameEngine(seed=0)
        assert len(engine.apple_spawner.positions) == 1


class TestEngineMovement:
    def test_basic_step(self):
        engine = GameEngine(width=20, height=20, seed=0)
        old_head = engine.snake.head
        state = engine.step()
        new_head = engine.snake.head
        assert new_head[1] == old_head[1] + 1  # moved right
        assert state["tick"] == 1

    def test_direction_change(self):
        engine = GameEngine(width=20, height=20, seed=0)
        engine.set_direction(Direction.UP)
        engine.step()
        assert engine.snake.head == (9, 10)

    def test_game_over_stops_ticks(self):
        engine = GameEngine(width=20, height=20, seed=0)
        engine.game_over = True
        state = engine.step()
        assert state["tick"] == 0  # tick did not advance


class TestEngineWallCollision:
    def test_death_on_wall_hit(self):
        engine = GameEngine(width=10, height=10, seed=0)
        engine.set_direction(Direction.RIGHT)
        # Move right until hitting the wall.
        for _ in range(20):
            engine.step()
            if engine.game_over:
                break
        assert engine.game_over
        assert not engine.snake.alive

    def test_wrap_mode_no_death(self):
        engine = GameEngine(
            width=10, height=10, wall_mode=WallMode.WRAP, seed=0,
        )
        engine.set_direction(Direction.RIGHT)
        for _ in range(20):
            engine.step()
            if engine.game_over:
                break
        # In wrap mode the snake wraps, so it might die from self-collision
        # but not from walls. We just verify it didn't die immediately.
        assert engine.tick > 5


class TestEngineSelfCollision:
    def test_dies_on_self_collision(self):
        engine = GameEngine(width=20, height=20, seed=42)
        # Grow the snake by placing apples directly ahead.
        from smart_snake.grid import CellType

        for _ in range(6):
            nr, nc = engine.snake.next_head()
            engine.grid.set(nr, nc, CellType.APPLE)
            if (nr, nc) not in engine.apple_spawner.positions:
                engine.apple_spawner.positions.append((nr, nc))
            engine.step()
        # Snake is now long enough. Turn into itself: right -> down -> left -> up.
        engine.set_direction(Direction.DOWN)
        engine.step()
        engine.set_direction(Direction.LEFT)
        engine.step()
        engine.set_direction(Direction.UP)
        engine.step()
        assert engine.game_over


class TestEngineAppleConsumption:
    def test_score_increases_on_apple(self):
        engine = GameEngine(width=10, height=10, max_apples=1, seed=0)
        initial_score = engine.score
        # Manually place apple directly ahead.
        from smart_snake.grid import CellType
        nr, nc = engine.snake.next_head()
        engine.grid.set(nr, nc, CellType.APPLE)
        engine.apple_spawner.positions.append((nr, nc))
        engine.step()
        assert engine.score == initial_score + 1
        assert len(engine.snake.body) == 4  # grew by 1

    def test_new_apple_spawned_after_eating(self):
        engine = GameEngine(width=10, height=10, max_apples=1, seed=0)
        from smart_snake.grid import CellType
        nr, nc = engine.snake.next_head()
        engine.grid.set(nr, nc, CellType.APPLE)
        engine.apple_spawner.positions.append((nr, nc))
        engine.step()
        assert len(engine.apple_spawner.positions) == 1

    def test_grid_head_remains_snake_after_eating(self):
        engine = GameEngine(width=10, height=10, max_apples=1, seed=0)
        from smart_snake.grid import CellType

        nr, nc = engine.snake.next_head()
        engine.grid.set(nr, nc, CellType.APPLE)
        engine.apple_spawner.positions.append((nr, nc))
        engine.step()

        head_r, head_c = engine.snake.head
        assert engine.grid.get(head_r, head_c) == CellType.SNAKE


class TestEngineSerialization:
    def test_state_is_json_serializable(self):
        engine = GameEngine(width=10, height=10, seed=42)
        engine.step()
        state = engine.get_state()
        # Must not raise.
        serialized = json.dumps(state)
        assert isinstance(serialized, str)

    def test_state_structure(self):
        engine = GameEngine(seed=0)
        state = engine.get_state()
        assert "tick" in state
        assert "score" in state
        assert "game_over" in state
        assert "grid" in state
        assert "snake" in state
        assert "apples" in state


class TestEngineDeterminism:
    def test_same_seed_same_outcome(self):
        """Two games with the same seed and actions produce identical states."""
        actions = [
            Direction.RIGHT, Direction.RIGHT, Direction.DOWN,
            Direction.DOWN, Direction.LEFT,
        ]
        state_a = self._run_game(seed=123, actions=actions)
        state_b = self._run_game(seed=123, actions=actions)
        assert state_a == state_b

    def test_different_seeds_differ(self):
        actions = [Direction.RIGHT] * 5
        state_a = self._run_game(seed=1, actions=actions)
        state_b = self._run_game(seed=2, actions=actions)
        # Apple positions should differ at minimum.
        assert state_a["apples"] != state_b["apples"]

    @staticmethod
    def _run_game(seed: int, actions: list[Direction]) -> dict:
        engine = GameEngine(width=20, height=20, seed=seed)
        for action in actions:
            engine.set_direction(action)
            engine.step()
        return engine.get_state()


class TestEngineRunsToCompletion:
    def test_full_game(self):
        """A game runs until the snake hits a wall without crashing."""
        engine = GameEngine(width=10, height=10, seed=7)
        # Head right until wall death.
        for _ in range(20):
            engine.step()
            if engine.game_over:
                break
        assert engine.game_over
        assert engine.tick > 0
