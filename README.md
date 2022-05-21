# Price Optimization

## Requirements

- Python: `>=3.8`
- Poetry: `1.1.13`

## Setup
### Poetry
- poetryにて仮想環境を構築
```shell
$ poetry install
```
- ライブラリ追加時は `poetry add {hoge}` で追加する
- poetryがインストールされていない場合は，pipにてインストール
```shell
$ pip install poetry
```
## Simulation
- config.yamlにて実験の設定を行い，下記コマンドを実行
```shell
$ make simulation
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
