export PYTHONPATH := $(PWD)

.PHONY: environment
environment:
	pip install poetry
	poetry install

.PHONY: iai
iai:
	poetry run python src/iai.py

.PHONY: pre-commit
pre-commit:
	poetry run pre-commit install

.PHONY: lint
lint:
	poetry run flake8 .

.PHONY: format
format:
	poetry run isort . & poetry run black .
	poetry run autoflake -ri --remove-all-unused-imports --ignore-init-module-imports --remove-unused-variables .

.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type f -name "*.json" -delete
	find . -type f -name "*.png" -delete
	find . -type f -name "*.html" -delete
	find . -type f -name "*.lp" -delete
	find . -type f -name "*.csv" -delete
	find . -type d -name "__pycache__" -delete

.PHONY: test
test:
	poetry run pytest .

.PHONY: artificial
artificial:
	poetry run python src/main.py --data_type artificial

.PHONY: realworld
realworld:
	poetry run python src/main.py --data_type realworld

.PHONY: tune_params
tune_params:
	poetry run python src/tuner.py
