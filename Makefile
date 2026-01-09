.PHONY: install test run clean

install:
	python -m pip install -e ".[dev]"

test:
	pytest -q

run:
	python demo.py

clean:
	rm -rf dist build .pytest_cache .mypy_cache *.egg-info
