"""Tests for the Snake module."""

import pytest

from smart_snake.snake import Direction, Snake


class TestSnakeInit:
    def test_default_creation(self):
        snake = Snake(5, 5)
        assert snake.head == (5, 5)
        assert len(snake.body) == 3
        assert snake.alive
        assert snake.direction == Direction.RIGHT

    def test_body_extends_opposite_to_direction(self):
        snake = Snake(5, 5, Direction.RIGHT, length=3)
        assert list(snake.body) == [(5, 5), (5, 4), (5, 3)]

    def test_body_extends_up(self):
        snake = Snake(5, 5, Direction.UP, length=3)
        assert list(snake.body) == [(5, 5), (6, 5), (7, 5)]

    def test_minimum_length(self):
        with pytest.raises(ValueError, match="at least 1"):
            Snake(0, 0, length=0)


class TestSnakeDirection:
    def test_set_valid_direction(self):
        snake = Snake(5, 5, Direction.RIGHT)
        snake.set_direction(Direction.UP)
        assert snake.direction == Direction.UP

    def test_ignore_180_reversal(self):
        snake = Snake(5, 5, Direction.RIGHT)
        snake.set_direction(Direction.LEFT)
        assert snake.direction == Direction.RIGHT

    def test_ignore_180_reversal_vertical(self):
        snake = Snake(5, 5, Direction.UP)
        snake.set_direction(Direction.DOWN)
        assert snake.direction == Direction.UP


class TestSnakeMovement:
    def test_next_head(self):
        snake = Snake(5, 5, Direction.RIGHT)
        assert snake.next_head() == (5, 6)

    def test_advance_without_growth(self):
        snake = Snake(5, 5, Direction.RIGHT, length=3)
        vacated = snake.advance()
        assert snake.head == (5, 6)
        assert len(snake.body) == 3
        assert vacated == (5, 3)

    def test_advance_with_growth(self):
        snake = Snake(5, 5, Direction.RIGHT, length=3)
        vacated = snake.advance(grow=True)
        assert snake.head == (5, 6)
        assert len(snake.body) == 4
        assert vacated is None

    def test_scheduled_growth(self):
        snake = Snake(5, 5, Direction.RIGHT, length=2)
        snake.schedule_growth(2)
        snake.advance()
        assert len(snake.body) == 3
        snake.advance()
        assert len(snake.body) == 4
        vacated = snake.advance()
        assert len(snake.body) == 4
        assert vacated is not None


class TestSnakeCollision:
    def test_occupies(self):
        snake = Snake(5, 5, Direction.RIGHT, length=3)
        assert snake.occupies(5, 5)
        assert snake.occupies(5, 4)
        assert not snake.occupies(0, 0)

    def test_self_collision(self):
        snake = Snake(5, 5, Direction.RIGHT, length=1)
        assert not snake.self_collision()
        # Manually create a self-collision scenario.
        snake.body.appendleft((5, 5))
        assert snake.self_collision()


class TestSnakeSerialization:
    def test_to_dict(self):
        snake = Snake(5, 5, Direction.RIGHT, length=2)
        d = snake.to_dict()
        assert d["body"] == [[5, 5], [5, 4]]
        assert d["direction"] == (0, 1)
        assert d["alive"] is True
