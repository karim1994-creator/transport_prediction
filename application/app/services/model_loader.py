from pathlib import Path
import joblib
from app.config import MODEL_FILES, MODELS_DIR

class ModelLoader:
    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.models = {}

    def load_all(self):
        loaded = {}
        for name, filename in MODEL_FILES.items():
            path = self.models_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"{name}: artefact introuvable : {path}")
            loaded[name] = joblib.load(path)
        self.models = loaded

    def get(self, name):
        if name not in self.models:
            raise KeyError(f"Modèle non chargé : {name}")
        return self.models[name]

    def status(self):
        return {
            name: {
                "file": str(self.models_dir / filename),
                "exists": (self.models_dir / filename).exists(),
                "loaded": name in self.models,
            }
            for name, filename in MODEL_FILES.items()
        }
