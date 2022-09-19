from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
DATA_DIR = ROOT_DIR / "data"
SRC_DIR = ROOT_DIR / "src"
RESULT_DIR = ROOT_DIR / "results"
OPT_DIR = SRC_DIR / "optimize"
OPT_MODEL_DIR = OPT_DIR / "models"
PRED_DIR = SRC_DIR / "predict"
DATA_PRE_DIR = SRC_DIR / "data_preprocess"
ALGO_DIR = OPT_DIR / "algorithms"
MODEL_DIR = ROOT_DIR / "models"
