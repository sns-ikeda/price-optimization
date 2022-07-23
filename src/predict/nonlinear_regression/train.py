import matplotlib.pyplot as plt
import numpy as np
from logzero import logger
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.configs import CONFIG_DATA
from src.predict.processing.preprocessor import load_preprocess, make_target_data


def train(dataset: str):
    config = CONFIG_DATA["realworld"][dataset]
    df = load_preprocess(dataset=dataset)
    if dataset == "breakfast":
        target_cols = [col for col in df.columns if "UNITS" in col]
        feature_cols = [col for col in df.columns if "PRICE" in col] + config["master_cols"]
        for target_col in tqdm(target_cols):
            product = target_col.split("_")[1]
            logger.info(product)
            X, y = make_target_data(df=df, target_col=target_col, feature_cols=feature_cols)
            # データを学習用と検証用に分割
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=0.7, test_size=0.3, shuffle=False
            )
            # 学習
            lr = LinearRegression()
            lr.fit(X_train, y_train)

            y_pred = lr.predict(X_test)  # 検証データを用いて目的変数を予測
            y_train_pred = lr.predict(X_train)  # 学習データに対する目的変数を予測
            logger.info(
                "RMSE train data: {0}".format(np.sqrt(mean_squared_error(y_train, y_train_pred)))
            )  # 学習データを用いたときの平均二乗誤差を出力
            logger.info(
                "RMSE test data: {0}".format(np.sqrt(mean_squared_error(y_test, y_pred)))
            )  # 検証データを用いたときの平均二乗誤差を出力

            # 結果をプロット
            x_col = "PRICE" + "_" + product
            plt.scatter(X[x_col], y, color="blue", label="actual")  # 説明変数と目的変数のデータ点の散布図をプロット
            plt.scatter(X[x_col], lr.predict(X), color="red", label="predicted")  # 回帰直線をプロット

            plt.title(f"Regression: {product}")  # 図のタイトル
            plt.xlabel("Price[$]")  # x軸のラベル
            plt.ylabel("#Units")  # y軸のラベル
            plt.grid()  # グリッド線を表示
            plt.legend()

            plt.show()
