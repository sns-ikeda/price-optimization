# Price Optimization

## Requirements

- Python: `>=3.8`
- Poetry: `1.1.13`

## Setup
- poetryにて実行環境を構築するため，下記コマンドを実行  
（poetryがインストールされていない場合は，pipにてインストールされる）
```shell
$ make env
```
- 最適決定木のライブラリ(interpretableai)を初めて利用する場合は，下記コマンドを実行  
```shell
$ make iai
```
- ライブラリを追加したい場合は `poetry add {hoge}` にて追加する

## How to Run
### preparation
- 以下のyamlファイルにてシミュレーションの設定を行う
    - 実験の設定：`src/config.yaml`
    - データ処理の設定：`src/data_preprocess/data_config.yaml`
    - アルゴリズムの設定：`src/optimize/algorithms/algo_config.yaml`
### execution
- 下記コマンドにて人工データによるシミュレーションを実行  
（ライセンスファイルiai.licが実行するPCのどこかに置かれている必要がある）

厳密モデル検証の場合：  
```shell
$ make compare_exact_models
```
ヒューリスティクス検証の場合：  
```shell
$ make compare_heuristics
```
（生成されたファイルを削除する場合は下記コマンドを実行）
```shell
$ make clean
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
