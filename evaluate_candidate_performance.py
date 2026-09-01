from pathlib import Path
import json

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

MODELS_DIR = (
    BASE_DIR
    / "models_a53"
    / "candidate"
)

REFERENCE_DIR = (
    BASE_DIR
    / "ci_reports"
    / "reference_test_sets"
)


MODEL_FILES = {

    "M1":
        "M1_hgb_historique_causal.joblib",

    "M2":
        "M2_hgb_historique_causal.joblib",
}


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model_name):

    print()
    print("=" * 80)
    print(
        f"EVALUATION DU CANDIDAT : {model_name}"
    )
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
            f"Candidat absent : {model_path}"
        )

    artifact = joblib.load(
        model_path
    )

    model = artifact["modele"]

    features = artifact["features"]

    X_test = pd.read_parquet(
        x_path
    )

    y_test = pd.read_parquet(
        y_path
    )["nb_vald"]

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

    print("=" * 80)
    print("EVALUATION DES MODELES CANDIDATS")
    print("=" * 80)

    results = []

    for model_name in MODEL_FILES:

        results.append(
            evaluate_model(
                model_name
            )
        )

    output = (
        BASE_DIR
        / "ci_reports"
        / "candidate_model_performance.json"
    )

    with open(
        output,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 80)
    print("EVALUATION CANDIDATS TERMINEE")
    print("=" * 80)

    print(
        output
    )


if __name__ == "__main__":

    main()