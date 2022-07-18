# Price Optimization

## Requirements

- Python: `>=3.8`
- Poetry: `1.1.13`

## Setup
### Poetry
- poetryがインストールされていない場合は，pipにてインストール
```shell
$ pip install poetry
```
- 仮想環境を構築
```shell
$ poetry install
```
- ライブラリ追加時は `poetry add {hoge}` で追加する

## How to Run
### preparation
- 以下のyamlファイルにてシミュレーションの設定を行う
    - 実験の設定：`src/config_simulation.yaml`
    - 使用するデータの設定：`data/config_data.yaml`
    - 最適化モデルの設定：`src/optimization/config_optimization.yaml`
    - アルゴリズムの設定：`src/optimization/algorithms/config_algorithm.yaml`
    - 予測モデルの設定：`src/prediction/config_prediction.yaml`
### execution
- 下記コマンドにてシミュレーションを実行

人工データの場合：
```shell
$ make artificial
```
実データの場合：
```shell
$ make realworld
```
## Code Formatting
### formatting
- isort, black, flake8によるformattingを実施（Setupの実施が必要）

```shell
$ make format
```
### pre-commit
- commit時に自動で静的解析を行い，pushしてCI/CDを回す前にフォーマットを行う
- pre-commitをインストール
```shell
$ poetry run install pre-commit
```
## Test

```shell
$ make test
```
