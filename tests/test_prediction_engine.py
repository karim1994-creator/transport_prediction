import pandas as pd
from app.services.prediction_engine import PredictionEngine

def test_m3_profile_sums_to_100():
    baseline = pd.DataFrame({
        "code_arret": ["A"] * 24,
        "id_zdc": ["Z"] * 24,
        "cat_jour": ["JOHV"] * 24,
        "heure_debut": list(range(24)),
        "pourc_validations": [1.0] * 24,
    })
    artifact = {
        "baseline": baseline,
        "profil_keys": ["code_arret", "id_zdc", "cat_jour"],
        "cles": ["code_arret", "id_zdc", "cat_jour", "heure_debut"],
        "cible": "pourc_validations",
        "strategie": "baseline_historique",
    }
    class Loader:
        def get(self, name): return artifact
    result = PredictionEngine(Loader()).predict_full_profile(
        "M3", {"code_arret": "A", "id_zdc": "Z", "cat_jour": "JOHV"}
    )
    assert len(result["profile"]) == 24
    assert abs(result["total_percent"] - 100.0) < 1e-6
