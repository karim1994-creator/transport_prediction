# ============================================================
# EVALUATION AUTOMATIQUE DES MODELES POUR CI/CD
#
# Objectif :
# - charger M1/M2/M3/M4
# - vérifier leur intégrité
# - effectuer des tests fonctionnels
# - produire un rapport JSON/CSV
# - définir un statut PASS / FAIL
#
# IMPORTANT :
# Les seuils sont volontairement techniques.
# Ils servent à détecter une régression du système.
# ============================================================

from pathlib import Path
import json
import sys
import gc

import numpy as np
import pandas as pd
import joblib


# ============================================================
# CHEMIN PROJET
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
)

MODELS_DIR = (
    BASE_DIR
    / "models_a53"
)

REPORT_DIR = (
    BASE_DIR
    / "ci_reports"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MODELES
# ============================================================

MODEL_FILES = {

    "M1":
        "M1_hgb_historique_causal.joblib",

    "M2":
        "M2_hgb_historique_causal.joblib",

    "M3":
        "M3_baseline_historique.joblib",

    "M4":
        "M4_baseline_historique.joblib",
}


# ============================================================
# SEUILS
# ============================================================
#
# Ces seuils ne remplacent PAS l'évaluation scientifique
# réalisée dans la validation temporelle.
#
# Ils servent ici à empêcher qu'un modèle cassé,
# incomplet ou incohérent soit déployé.
# ============================================================

MIN_PROFILE_SUM = 99.99
MAX_PROFILE_SUM = 100.01


# ============================================================
# RESULTATS
# ============================================================

results = []


# ============================================================
# TEST M1 / M2
# ============================================================

def evaluate_ml_artifact(
    model_name,
    artifact
):

    checks = []

    # --------------------------------------------------------
    # Présence des clés
    # --------------------------------------------------------

    required_keys = [
        "modele",
        "features",
        "features_categorielles",
        "features_numeriques",
        "cible",
        "perimetre",
        "nom_modele",
    ]

    missing = [
        key
        for key in required_keys
        if key not in artifact
    ]

    checks.append(
        {
            "test":
                "structure",

            "success":
                len(missing) == 0,

            "details":
                missing,
        }
    )

    # --------------------------------------------------------
    # Modèle
    # --------------------------------------------------------

    checks.append(
        {
            "test":
                "modele_present",

            "success":
                artifact.get(
                    "modele"
                ) is not None,

            "details":
                type(
                    artifact.get(
                        "modele"
                    )
                ).__name__,
        }
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    features = artifact.get(
        "features",
        []
    )

    checks.append(
        {
            "test":
                "features_non_vides",

            "success":
                len(features) > 0,

            "details":
                features,
        }
    )

    # --------------------------------------------------------
    # Test d'inférence minimal
    #
    # On utilise une seule ligne synthétique.
    # Les variables catégorielles reçoivent une valeur.
    # --------------------------------------------------------

    try:

        model = artifact[
            "modele"
        ]

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

        success = (
            len(prediction) == 1
            and np.isfinite(
                prediction[0]
            )
        )

        details = float(
            prediction[0]
        )

    except Exception as exc:

        success = False

        details = str(
            exc
        )

    checks.append(
        {
            "test":
                "inference",

            "success":
                bool(success),

            "details":
                details,
        }
    )

    # --------------------------------------------------------
    # Résultat final
    # --------------------------------------------------------

    status = all(
        check["success"]
        for check in checks
    )

    return {
        "model":
            model_name,

        "type":
            "ML",

        "status":
            "PASS" if status else "FAIL",

        "checks":
            checks,

        "cible":
            artifact.get(
                "cible"
            ),

        "perimetre":
            artifact.get(
                "perimetre"
            ),
    }


# ============================================================
# TEST M3 / M4
# ============================================================

# ============================================================
# TEST M3 / M4
# ============================================================

def evaluate_profile_artifact(
    model_name,
    artifact
):

    checks = []

    # --------------------------------------------------------
    # Structure obligatoire
    # --------------------------------------------------------

    required_keys = [
        "baseline",
        "cles",
        "colonne_prediction",
        "cible",
        "perimetre",
        "nom_modele",
    ]

    missing = [
        key
        for key in required_keys
        if key not in artifact
    ]

    structure_ok = (
        len(missing) == 0
    )

    checks.append(
        {
            "test":
                "structure",

            "success":
                structure_ok,

            "details":
                missing,
        }
    )

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    baseline = artifact.get(
        "baseline"
    )

    baseline_exists = (
        isinstance(
            baseline,
            pd.DataFrame
        )
        and not baseline.empty
    )

    checks.append(
        {
            "test":
                "baseline_present",

            "success":
                baseline_exists,

            "details":
                None
                if not baseline_exists
                else list(
                    baseline.shape
                ),
        }
    )

    if not baseline_exists:

        return {
            "model":
                model_name,

            "type":
                "Baseline",

            "status":
                "FAIL",

            "checks":
                checks,

            "cible":
                artifact.get(
                    "cible"
                ),

            "perimetre":
                artifact.get(
                    "perimetre"
                ),
        }

    # --------------------------------------------------------
    # Clés du profil
    # --------------------------------------------------------

    keys = artifact.get(
        "cles",
        []
    )

    keys_ok = (
        len(keys) > 0
        and all(
            key in baseline.columns
            for key in keys
        )
    )

    checks.append(
        {
            "test":
                "cles_presentes",

            "success":
                keys_ok,

            "details":
                keys,
        }
    )

    # --------------------------------------------------------
    # Colonne heure
    # --------------------------------------------------------

    heure_ok = (
        "heure_debut"
        in baseline.columns
    )

    checks.append(
        {
            "test":
                "heure_debut",

            "success":
                heure_ok,

            "details":
                "heure_debut"
                if heure_ok
                else "colonne absente",
        }
    )

    # --------------------------------------------------------
    # Vérification des 24 heures
    #
    # On vérifie les heures réellement présentes dans
    # le baseline.
    # --------------------------------------------------------

    hours_ok = False
    hours = []

    if heure_ok:

        try:

            hours = (
                pd.to_numeric(
                    baseline[
                        "heure_debut"
                    ],
                    errors="coerce"
                )
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            )

            hours = sorted(
                hours
            )

            hours_ok = (
                hours
                == list(
                    range(24)
                )
            )

        except Exception:

            hours_ok = False

    checks.append(
        {
            "test":
                "24_heures",

            "success":
                hours_ok,

            "details":
                hours,
        }
    )

    # --------------------------------------------------------
    # Vérification de la colonne cible
    # --------------------------------------------------------

    target_column = (
        artifact.get(
            "colonne_prediction",
            "pourc_validations"
        )
    )

    target_ok = (
        target_column
        in baseline.columns
    )

    checks.append(
        {
            "test":
                "colonne_prediction",

            "success":
                target_ok,

            "details":
                target_column,
        }
    )

    # --------------------------------------------------------
    # Vérification des valeurs
    # --------------------------------------------------------

    values_ok = False
    value_min = None
    value_max = None

    if target_ok:

        try:

            values = pd.to_numeric(
                baseline[
                    target_column
                ],
                errors="coerce"
            )

            values_ok = (
                values.notna().all()
                and np.isfinite(
                    values.to_numpy()
                ).all()
                and (
                    values >= 0
                ).all()
            )

            value_min = float(
                values.min()
            )

            value_max = float(
                values.max()
            )

        except Exception:

            values_ok = False

    checks.append(
        {
            "test":
                "valeurs_valides",

            "success":
                values_ok,

            "details":
                {
                    "min":
                        value_min,

                    "max":
                        value_max,
                },
        }
    )

    # --------------------------------------------------------
    # Vérification des sommes à 100 %
    #
    # IMPORTANT :
    # dropna=False permet de ne pas perdre les profils
    # contenant éventuellement une valeur manquante.
    #
    # observed=True évite les groupes catégoriels inutilisés.
    # --------------------------------------------------------

    sums_ok = False
    sum_min = None
    sum_max = None
    nombre_profils = 0

    if (
        keys_ok
        and heure_ok
        and target_ok
    ):

        try:

            profile_keys = [
                key
                for key in keys
                if key != "heure_debut"
            ]

            if len(profile_keys) > 0:

                totals = (
                    baseline
                    .groupby(
                        profile_keys,
                        dropna=False,
                        observed=True
                    )[
                        target_column
                    ]
                    .sum()
                )

            else:

                totals = pd.Series(
                    [
                        baseline[
                            target_column
                        ].sum()
                    ]
                )

            nombre_profils = len(
                totals
            )

            sum_min = float(
                totals.min()
            )

            sum_max = float(
                totals.max()
            )

            # Tolérance numérique
            sums_ok = (
                np.isfinite(
                    totals.to_numpy()
                ).all()
                and
                (
                    np.abs(
                        totals.to_numpy()
                        - 100.0
                    )
                    <= 0.01
                ).all()
            )

        except Exception as exc:

            sums_ok = False

            sum_min = None
            sum_max = None

            nombre_profils = 0

    checks.append(
        {
            "test":
                "profils_100_pourcent",

            "success":
                sums_ok,

            "details":
                {
                    "nombre_profils":
                        nombre_profils,

                    "min":
                        sum_min,

                    "max":
                        sum_max,
                },
        }
    )

    # --------------------------------------------------------
    # Résultat final
    # --------------------------------------------------------

    status = all(
        check["success"]
        for check in checks
    )

    return {
        "model":
            model_name,

        "type":
            "Baseline",

        "status":
            "PASS"
            if status
            else "FAIL",

        "checks":
            checks,

        "cible":
            artifact.get(
                "cible"
            ),

        "perimetre":
            artifact.get(
                "perimetre"
            ),
    }
# ============================================================
# EVALUATION D'UN MODELE
# ============================================================

def evaluate_model(
    model_name,
    filename
):

    print()
    print(
        "-" * 80
    )

    print(
        f"EVALUATION {model_name}"
    )

    print(
        "-" * 80
    )

    path = (
        MODELS_DIR
        / filename
    )

    # --------------------------------------------------------
    # Fichier
    # --------------------------------------------------------

    if not path.exists():

        return {
            "model":
                model_name,

            "status":
                "FAIL",

            "error":
                f"Fichier absent : {path}",
        }

    # --------------------------------------------------------
    # Chargement
    # --------------------------------------------------------

    try:

        artifact = joblib.load(
            path
        )

    except Exception as exc:

        return {
            "model":
                model_name,

            "status":
                "FAIL",

            "error":
                str(exc),
        }

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    if model_name in {
        "M1",
        "M2",
    }:

        result = evaluate_ml_artifact(
            model_name,
            artifact
        )

    else:

        result = evaluate_profile_artifact(
            model_name,
            artifact
        )

    print(
        "STATUT :",
        result["status"]
    )

    # --------------------------------------------------------
    # Libération
    # --------------------------------------------------------

    del artifact

    gc.collect()

    return result


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main():

    print()
    print(
        "=" * 100
    )

    print(
        "EVALUATION CI/CD — MODELES M1 / M2 / M3 / M4"
    )

    print(
        "=" * 100
    )

    results.clear()

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    for model_name, filename in (
        MODEL_FILES.items()
    ):

        result = evaluate_model(
            model_name,
            filename
        )

        results.append(
            result
        )

    # --------------------------------------------------------
    # Tableau
    # --------------------------------------------------------

    rows = []

    for result in results:

        rows.append(
            {
                "model":
                    result["model"],

                "type":
                    result.get(
                        "type",
                        ""
                    ),

                "status":
                    result["status"],

                "target":
                    result.get(
                        "cible",
                        ""
                    ),

                "perimeter":
                    result.get(
                        "perimetre",
                        ""
                    ),
            }
        )

    result_df = pd.DataFrame(
        rows
    )

    print()
    print(
        "=" * 100
    )

    print(
        "RESULTATS"
    )

    print(
        "=" * 100
    )

    print(
        result_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # JSON détaillé
    # --------------------------------------------------------

    json_file = (
        REPORT_DIR
        / "model_evaluation_report.json"
    )

    with open(
        json_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False,
            default=str
        )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    csv_file = (
        REPORT_DIR
        / "model_evaluation_report.csv"
    )

    result_df.to_csv(
        csv_file,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Statut global
    # --------------------------------------------------------

    global_success = all(
        result["status"] == "PASS"
        for result in results
    )

    global_status = (
        "PASS"
        if global_success
        else "FAIL"
    )

    summary = {

        "status":
            global_status,

        "models":
            len(results),

        "passed":
            sum(
                result["status"] == "PASS"
                for result in results
            ),

        "failed":
            sum(
                result["status"] == "FAIL"
                for result in results
            ),
    }

    summary_file = (
        REPORT_DIR
        / "model_evaluation_summary.json"
    )

    with open(
        summary_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # Affichage
    # --------------------------------------------------------

    print()
    print(
        "=" * 100
    )

    print(
        "STATUT GLOBAL :",
        global_status
    )

    print(
        "Modèles PASS :",
        summary["passed"]
    )

    print(
        "Modèles FAIL :",
        summary["failed"]
    )

    print()
    print(
        "Rapport JSON :"
    )

    print(
        json_file
    )

    print()
    print(
        "Rapport CSV :"
    )

    print(
        csv_file
    )

    print()
    print(
        "Résumé :"
    )

    print(
        summary_file
    )

    print()
    print(
        "=" * 100
    )

    # --------------------------------------------------------
    # Important :
    # le code de sortie sera utilisé par GitHub Actions.
    # --------------------------------------------------------

    if not global_success:

        print(
            "CI/CD : DEPLOIEMENT A BLOQUER"
        )

        sys.exit(1)

    print(
        "CI/CD : MODELES VALIDES"
    )

    sys.exit(0)


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":
    main()