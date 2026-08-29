# ============================================================
# TEST DE CHARGEMENT DES 4 MODELES
# ============================================================

from pathlib import Path
import joblib


# ============================================================
# CHEMIN RACINE DU PROJET
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parents[1]
)


MODELS_DIR = (
    BASE_DIR
    / "models_a53"
)


MODEL_FILES = {
    "M1": "M1_hgb_historique_causal.joblib",
    "M2": "M2_hgb_historique_causal.joblib",
    "M3": "M3_baseline_historique.joblib",
    "M4": "M4_baseline_historique.joblib",
}


# ============================================================
# TEST 1 : LES FICHIERS EXISTENT
# ============================================================

def test_model_files_exist():

    for model_name, filename in MODEL_FILES.items():

        model_path = (
            MODELS_DIR
            / filename
        )

        assert model_path.exists(), (
            f"{model_name} absent : {model_path}"
        )

        assert model_path.is_file(), (
            f"{model_name} n'est pas un fichier : {model_path}"
        )


# ============================================================
# TEST 2 : LES FICHIERS SONT CHARGEABLES
# ============================================================

def test_models_can_be_loaded():

    for model_name, filename in MODEL_FILES.items():

        model_path = (
            MODELS_DIR
            / filename
        )

        artifact = joblib.load(
            model_path
        )

        assert artifact is not None, (
            f"{model_name} n'a pas pu être chargé"
        )

        assert isinstance(
            artifact,
            dict
        ), (
            f"{model_name} doit contenir un dictionnaire"
        )


# ============================================================
# TEST 3 : M1 / M2 CONTIENNENT UN MODELE ML
# ============================================================

def test_m1_m2_artifacts_structure():

    for model_name in ["M1", "M2"]:

        model_path = (
            MODELS_DIR
            / MODEL_FILES[model_name]
        )

        artifact = joblib.load(
            model_path
        )

        assert "modele" in artifact
        assert "features" in artifact
        assert "cible" in artifact
        assert "perimetre" in artifact
        assert "nom_modele" in artifact


# ============================================================
# TEST 4 : M3 / M4 CONTIENNENT UNE BASELINE
# ============================================================

def test_m3_m4_artifacts_structure():

    for model_name in ["M3", "M4"]:

        model_path = (
            MODELS_DIR
            / MODEL_FILES[model_name]
        )

        artifact = joblib.load(
            model_path
        )

        assert "baseline" in artifact
        assert "cles" in artifact
        assert "cible" in artifact
        assert "perimetre" in artifact
        assert "nom_modele" in artifact


# ============================================================
# TEST 5 : NOMS DES MODELES
# ============================================================

def test_model_names():

    expected_names = {
        "M1": "M1_hgb_historique_causal",
        "M2": "M2_hgb_historique_causal",
        "M3": "M3_baseline_historique",
        "M4": "M4_baseline_historique",
    }

    for model_name, expected in expected_names.items():

        model_path = (
            MODELS_DIR
            / MODEL_FILES[model_name]
        )

        artifact = joblib.load(
            model_path
        )

        assert artifact["nom_modele"] == expected