export PYTHONPATH := $(PWD)

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
