.PHONY: install test lint local-study

install:
	python -m pip install -e ".[dev,editor]"

test:
	pytest

lint:
	ruff check .

local-study:
	wsp-vision local-study --output-dir results/seed-0 --seed 0

