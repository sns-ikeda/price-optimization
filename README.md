# Prescriptive Price Optimization Using Oprimal Regression Trees

## Requirements

- Python: `>=3.8`
- Poetry: `1.1.13`

## Setup
- To set up the execution environment with poetry, run the command below.
(If poetry is not installed, it will be installed via pip)
```shell
$ make env
```
- If you're using the optimal decision tree library (interpretableai) for the first time, run the following command. (A license for interpretableai is required)  
```shell
$ make iai
```
- If you want to add a library, use poetry add {hoge} to do so.

## How to Run
### preparation
- Configure the simulation with the following yaml files:
  - Algorithm settings: src/optimize/algorithms/algo_config.yaml
### execution
- Run the simulation with artificial data using the commands below.  
（A license file named iai.lic needs to be located somewhere on the executing PC）

For exact model simulation：  
```shell
$ make compare_exact_models
```
For heuristics simulation：  
```shell
$ make compare_heuristics
```
(To remove the generated files, run the command below)
```shell
$ make clean
```
## Code Formatting
### formatting
- Perform formatting using isort, black, and flake8. (Setup execution is required)

```shell
$ make format
```
### pre-commit
- Automatically perform static code analysis during commit and format before pushing to run CI/CD.
- Install pre-commit:
```shell
$ poetry run install pre-commit
```
## Test

```shell
$ make test
```
