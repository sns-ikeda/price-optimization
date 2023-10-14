.PHONY: env iai pre-commit lint format clean test compare_exact_models compare_heuristics
export PYTHONPATH := $(PWD)

env:
	pip install poetry
	poetry install

iai:
	poetry run python src/iai.py

pre-commit:
	poetry run pre-commit install

lint:
	poetry run black . --check

format:
	poetry run isort . & poetry run black .
	poetry run autoflake -ri --remove-all-unused-imports --ignore-init-module-imports --remove-unused-variables .

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type f -name "*.json" -delete
	find . -type f -name "*.png" -delete
	find . -type f -name "*.html" -delete
	find . -type f -name "*.lp" -delete
	find . -type f -name "*.pickle" -delete
	find . -type f -name "*.ilp" -delete
	find ./data/*/processed/ -type f -name "*.csv" -delete
	find ./results/synthetic/ -type f -name "*.csv" -delete
	find . -type d -name "__pycache__" -delete

test:
	poetry run pytest .

compare_exact_models:
	poetry run python src/compare_exact_models.py

compare_heuristics:
	poetry run python src/compare_heuristics.py
