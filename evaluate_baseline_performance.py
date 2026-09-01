from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR / "models_a53"
BACKUP_DIR = MODELS_DIR / "backup"


MODEL_FILES = {
    "M1": "M1_hgb_historique_causal.joblib",
    "M2": "M2_hgb_historique_causal.joblib",
}


def evaluate_model(model_name):

    model_path = (
        BACKUP_DIR
        / MODEL_FILES[model_name]
    )

    print()
    print("=" * 80)
    print(f"EVALUATION MODELE ACTUEL : {model_name}")
    print("=" * 80)

    artifact = joblib.load(
        model_path
    )

    model = artifact["modele"]

    features = artifact["features"]

    # --------------------------------------------------------
    # Jeu synthétique de contrôle
    # --------------------------------------------------------
    #
    # Ici nous vérifions seulement que le modèle produit
    # une sortie numérique cohérente.
    #
    # La vraie comparaison scientifique devra utiliser
    # un jeu de test temporel figé.
    # --------------------------------------------------------

    row = {}

    for feature in features:

        if feature in artifact.get(
            "features_categorielles",
            []
        ):

            row[feature] = "TEST"

        else:

            row[feature] = 0

    X = pd.DataFrame(
        [row]
    )

    prediction = model.predict(
        X
    )

    prediction = np.asarray(
        prediction,
        dtype=float
    )

    print(
        "Prediction de contrôle :",
        prediction
    )

    print(
        "Prediction finie :",
        np.isfinite(prediction).all()
    )

    return {
        "model": model_name,
        "prediction": float(prediction[0]),
    }


def main():

    print()
    print("=" * 80)
    print("REFERENCE DES MODELES ACTUELS")
    print("=" * 80)

    results = []

    for model_name in MODEL_FILES:

        results.append(
            evaluate_model(
                model_name
            )
        )

    print()
    print("=" * 80)
    print("REFERENCE TERMINEE")
    print("=" * 80)

    for result in results:

        print(
            result["model"],
            "->",
            result["prediction"]
        )


if __name__ == "__main__":
    main()