export PYTHONPATH := $(PWD)

lint:
	poetry run flake8 .

format:
	poetry run isort . & poetry run black .
	poetry run autoflake -ri --remove-all-unused-imports --ignore-init-module-imports --remove-unused-variables .
	poetry run flake8 .

test:
	poetry run pytest .

simulation:
	poetry run python src/main.py
