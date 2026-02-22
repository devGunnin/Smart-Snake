.PHONY: install test lint fmt check

install:
	pip install -e ".[dev]"

test:
	pytest --cov=smart_snake --cov-report=term-missing

lint:
	ruff check src/ tests/

fmt:
	ruff format src/ tests/
	ruff check --fix src/ tests/

check: lint test
