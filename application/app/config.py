from pathlib import Path
import os

PROJECT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = Path(os.getenv("MODELS_DIR", PROJECT_DIR / "models_a53"))

MODEL_FILES = {
    "M1": "M1_hgb_historique_causal.joblib",
    "M2": "M2_hgb_historique_causal.joblib",
    "M3": "M3_baseline_historique.joblib",
    "M4": "M4_baseline_historique.joblib",
}