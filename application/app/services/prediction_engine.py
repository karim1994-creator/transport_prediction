import numpy as np
import pandas as pd

class PredictionEngine:
    def __init__(self, loader):
        self.loader = loader

    def predict_ml(self, model_name, rows):
        artifact = self.loader.get(model_name)
        if "modele" not in artifact or "features" not in artifact:
            raise ValueError(f"{model_name} n'est pas un artefact ML M1/M2 compatible.")
        df = pd.DataFrame(rows)
        features = list(artifact["features"])
        missing = [c for c in features if c not in df.columns]
        if missing:
            raise ValueError(f"{model_name}: variables manquantes : {missing}")
        pred = np.clip(
            np.asarray(artifact["modele"].predict(df[features]), dtype=float),
            0, None
        )
        return {
            "model": model_name,
            "target": artifact.get("cible"),
            "n_rows": len(df),
            "predictions": pred.tolist(),
            "features": features,
        }

    def predict_full_profile(self, model_name, profile):
        artifact = self.loader.get(model_name)
        if "baseline" not in artifact or "profil_keys" not in artifact:
            raise ValueError(f"{model_name} n'est pas un artefact baseline M3/M4 compatible.")

        keys = list(artifact["profil_keys"])
        missing = [k for k in keys if k not in profile]
        if missing:
            raise ValueError(f"Variables profil manquantes : {missing}")

        baseline = artifact["baseline"]
        rows = [{**{k: profile[k] for k in keys}, "heure_debut": h} for h in range(24)]
        query = pd.DataFrame(rows)
        merge_keys = keys + ["heure_debut"]

        out = query.merge(
            baseline[merge_keys + ["pourc_validations"]],
            on=merge_keys,
            how="left"
        )
        out["prediction"] = out["pourc_validations"].fillna(100.0 / 24.0).astype(float)
        out = out.drop(columns=["pourc_validations"])

        total = out["prediction"].sum()
        if total > 0:
            out["prediction"] = out["prediction"] / total * 100.0
        else:
            out["prediction"] = 100.0 / 24.0

        return {
            "model": model_name,
            "target": artifact.get("cible"),
            "strategy": artifact.get("strategie"),
            "profile_keys": keys,
            "profile": [
                {
                    "heure_debut": int(r.heure_debut),
                    "prediction_percent": float(r.prediction)
                }
                for r in out.itertuples()
            ],
            "total_percent": float(out["prediction"].sum()),
        }
