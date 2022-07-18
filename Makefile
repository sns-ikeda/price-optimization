export PYTHONPATH := $(PWD)

pre-commit:
	poetry run pre-commit install

lint:
	poetry run flake8 .

format:
	poetry run isort . & poetry run black .
	poetry run autoflake -ri --remove-all-unused-imports --ignore-init-module-imports --remove-unused-variables .
	poetry run flake8 .

test:
	poetry run pytest .

artificial:
	poetry run python src/run_artificial.py

realworld:
	poetry run python src/run_realworld.py
