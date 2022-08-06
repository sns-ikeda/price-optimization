import time
from functools import wraps


def stop_watch(func):
    """関数の実行時間を出力するデコレータ"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # 処理開始直前の時間
        start = time.time()

        # 処理実行
        result = func(*args, **kwargs)

        # 処理終了直後の時間から処理時間を算出
        elapsed_time = time.time() - start

        # 処理時間を出力
        print(f"{int(elapsed_time)} s in {func.__name__}")

        return result

    return wrapper
