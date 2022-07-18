export PYTHONPATH := $(PWD)

.PHONY: format
pre-commit:
	poetry run pre-commit install

.PHONY: format
format:
	poetry run isort . & poetry run black .
	poetry run autoflake -ri --remove-all-unused-imports --ignore-init-module-imports --remove-unused-variables .
	poetry run flake8 .

.PHONY: test
test:
	poetry run pytest .

.PHONY: artificial
artificial:
	poetry run python src/run_artificial.py

.PHONY: realworld
realworld:
	poetry run python src/run_realworld.py
