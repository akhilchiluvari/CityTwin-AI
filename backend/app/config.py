from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "backend" / "models"

MODEL_FILE = MODELS_DIR / "best_model.pkl"
MODEL_METADATA_FILE = MODELS_DIR / "best_model_metrics.json"
FEATURE_GRID_FILE = PROCESSED_DIR / "feature_grid.tif"
TRAINING_DATASET_FILE = PROCESSED_DIR / "training_dataset.csv"
RISK_MAP_FILE = PROCESSED_DIR / "risk_map.tif"

for path in [RAW_DIR, PROCESSED_DIR, MODELS_DIR]:
    path.mkdir(parents=True, exist_ok=True)
