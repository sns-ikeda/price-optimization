.PHONY: env iai pre-commit lint format clean test artificial realworld tune_params data_preprocess train optimize
export PYTHONPATH := $(PWD)

env:
	pip install poetry
	poetry install

iai:
	poetry run python src/iai.py

pre-commit:
	poetry run pre-commit install

lint:
	poetry run flake8 .

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
	find ./data/*/processed/ -type f -name "*.csv" -delete
	find . -type d -name "__pycache__" -delete

test:
	poetry run pytest .

artificial:
	poetry run python src/main.py --data_type artificial

realworld:
	poetry run python src/main.py --data_type realworld

data_preprocess:
	poetry run python src/data_preprocess/preprocessor.py

train:
	poetry run python src/predict/predictor.py

optimize:
	poetry run python src/optimize/optimizer.py
