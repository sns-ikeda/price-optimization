from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
DATA_DIR = ROOT_DIR / "data"
SRC_DIR = ROOT_DIR / "src"
RESULT_DIR = DATA_DIR / "results"
OPT_DIR = SRC_DIR / "optimization"
REG_DIR = SRC_DIR / "prediction"
ALGO_DIR = OPT_DIR / "algorithms"
MODEL_DIR = OPT_DIR / "models"
