import sys
from functools import wraps
from logging import DEBUG, INFO, FileHandler, Formatter, Logger, StreamHandler, getLogger
from pathlib import Path
from typing import Optional


def set_logger(
    log_name,
    log_level: int = DEBUG,
    slevel: int = DEBUG,
    fpath: Optional[Path] = None,
    flevel: int = INFO,
) -> Logger:
    """loggerを生成"""
    logger = getLogger(log_name)
    logger.setLevel(slevel)
    formatter = Formatter("%(asctime)s - %(filename)s - %(levelname)s - %(message)s")

    if not logger.handlers:
        sh = StreamHandler(sys.stdout)
        sh.setLevel(slevel)
        sh.setFormatter(formatter)

        logger.setLevel(log_level)
        logger.addHandler(sh)

        if fpath is None:
            return logger

        fh = FileHandler(fpath)
        fh.setLevel(flevel)
        fh.setFormatter(formatter)

        logger.addHandler(fh)

    return logger


def logging_function(func):
    """関数が利用されたことをloggingするデコレータ"""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        logger = set_logger(func.__name__)

        if len(args):
            logger.info(args)
        if len(kwargs):
            logger.info(kwargs)

        return func(self, *args, **kwargs)

    return wrapper
