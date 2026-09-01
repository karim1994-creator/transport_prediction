from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR / "models_a53" / "backup"

REFERENCE_DIR = (
    BASE_DIR
    / "ci_reports"
    / "reference_test_sets"
)


MODEL_FILES = {
    "M1": "M1_hgb_historique_causal.joblib",
    "M2": "M2_hgb_historique_causal.joblib",
}


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model_name):

    print()
    print("=" * 80)
    print(f"PERFORMANCE MODELE ACTUEL : {model_name}")
    print("=" * 80)

    model_path = (
        MODELS_DIR
        / MODEL_FILES[model_name]
    )

    x_path = (
        REFERENCE_DIR
        / f"{model_name}_X_test.parquet"
    )

    y_path = (
        REFERENCE_DIR
        / f"{model_name}_y_test.parquet"
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"Modèle absent : {model_path}"
        )

    if not x_path.exists():

        raise FileNotFoundError(
            f"Jeu X absent : {x_path}"
        )

    if not y_path.exists():

        raise FileNotFoundError(
            f"Jeu y absent : {y_path}"
        )

    # --------------------------------------------------------
    # Chargement
    # --------------------------------------------------------

    artifact = joblib.load(
        model_path
    )

    model = artifact["modele"]

    X_test = pd.read_parquet(
        x_path
    )

    y_test = pd.read_parquet(
        y_path
    )["nb_vald"]

    # --------------------------------------------------------
    # Vérification features
    # --------------------------------------------------------

    features = artifact["features"]

    missing = [
        col
        for col in features
        if col not in X_test.columns
    ]

    if missing:

        raise ValueError(
            f"{model_name} : features manquantes : {missing}"
        )

    X_test = X_test[
        features
    ]

    # --------------------------------------------------------
    # Prédiction
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    predictions = np.asarray(
        predictions,
        dtype=float
    )

    predictions = np.clip(
        predictions,
        0,
        None
    )

    # --------------------------------------------------------
    # Métriques
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    print(
        f"Nombre observations : {len(y_test)}"
    )

    print(
        f"MAE  : {mae:.6f}"
    )

    print(
        f"RMSE : {rmse:.6f}"
    )

    print(
        f"R²   : {r2:.6f}"
    )

    return {
        "model": model_name,
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "samples": int(len(y_test)),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("PERFORMANCES DE REFERENCE DES MODELES")
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
    print("RESULTATS DE REFERENCE")
    print("=" * 80)

    df = pd.DataFrame(
        results
    )

    print(
        df.to_string(
            index=False
        )
    )

    output_file = (
        BASE_DIR
        / "ci_reports"
        / "reference_model_performance.json"
    )

    df.to_json(
        output_file,
        orient="records",
        indent=2
    )

    print()
    print(
        "Rapport :"
    )

    print(
        output_file
    )


if __name__ == "__main__":

    main()