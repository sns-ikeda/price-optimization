# Price Optimization

## Requirements

- Python: `>=3.8`
- Poetry: `1.1.13`

## Setup
- poetryにて実行環境を構築するため，下記コマンドを実行  
（poetryがインストールされていない場合は，pipにてインストールされる）
```shell
$ make environment
```
- ライブラリを追加したい場合は `poetry add {hoge}` にて追加する

## How to Run
### preparation
- 以下のyamlファイルにてシミュレーションの設定を行う
    - 実験の設定：`src/config_simulation.yaml`
    - 使用するデータの設定：`data/config_data.yaml`
    - 最適化モデルの設定：`src/optimize/config_optimize.yaml`
    - アルゴリズムの設定：`src/optimize/algorithms/config_algorithm.yaml`
    - 予測モデルの設定：`src/predict/config_predict.yaml`
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
