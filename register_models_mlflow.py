# ============================================================
# REGISTRE DES MODELES M1 / M2 / M3 / M4
#
# Objectif :
# - enregistrer les 4 artefacts existants dans MLflow
# - conserver leur version
# - conserver les métadonnées
# - permettre leur réutilisation dans le pipeline CI/CD
#
# M1 : nb_vald - Ferré
# M2 : nb_vald - Surface
# M3 : profil horaire - Ferré - baseline historique
# M4 : profil horaire - Surface - baseline historique
# ============================================================

from pathlib import Path
import sys
import json
from datetime import datetime

import joblib
import mlflow


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


# ============================================================
# DOSSIER DE REGISTRE LOCAL
# ============================================================

REGISTRY_DIR = (
    BASE_DIR
    / "mlflow_registry"
)

REGISTRY_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# URI MLflow
# ============================================================

MLFLOW_DB = (
    BASE_DIR
    / "mlflow.db"
)

mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB}"
)


# ============================================================
# EXPERIENCE
# ============================================================

EXPERIMENT_NAME = (
    "transport_prediction"
)

mlflow.set_experiment(
    EXPERIMENT_NAME
)


# ============================================================
# MODELES
# ============================================================

MODEL_FILES = {

    "M1": {
        "file":
            "M1_hgb_historique_causal.joblib",

        "description":
            "Modèle HistGradientBoosting "
            "pour prédiction du nombre de validations "
            "sur le réseau ferré.",

        "target":
            "nb_vald",

        "perimeter":
            "Ferre",
    },

    "M2": {
        "file":
            "M2_hgb_historique_causal.joblib",

        "description":
            "Modèle HistGradientBoosting "
            "pour prédiction du nombre de validations "
            "sur le réseau de surface.",

        "target":
            "nb_vald",

        "perimeter":
            "Surface",
    },

    "M3": {
        "file":
            "M3_baseline_historique.joblib",

        "description":
            "Baseline historique pour prédiction "
            "du profil horaire des validations "
            "sur le réseau ferré.",

        "target":
            "pourc_validations",

        "perimeter":
            "Ferre",
    },

    "M4": {
        "file":
            "M4_baseline_historique.joblib",

        "description":
            "Baseline historique pour prédiction "
            "du profil horaire des validations "
            "sur le réseau de surface.",

        "target":
            "pourc_validations",

        "perimeter":
            "Surface",
    },
}


# ============================================================
# REGISTRE DES MODELES
# ============================================================

def register_model(
    model_name,
    information
):

    model_file = (
        MODELS_DIR
        / information["file"]
    )

    if not model_file.exists():

        raise FileNotFoundError(
            f"Artefact absent : {model_file}"
        )


    # --------------------------------------------------------
    # Chargement
    # --------------------------------------------------------

    artifact = joblib.load(
        model_file
    )


    if not isinstance(
        artifact,
        dict
    ):

        raise TypeError(
            f"{model_name} : "
            "l'artefact doit être un dictionnaire."
        )


    # --------------------------------------------------------
    # Nom logique du modèle
    # --------------------------------------------------------

    registry_name = (
        f"TransportPrediction_{model_name}"
    )


    # --------------------------------------------------------
    # Début du run MLflow
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name=model_name
    ) as run:

        # ----------------------------------------------------
        # Tags
        # ----------------------------------------------------

        mlflow.set_tag(
            "model_name",
            model_name
        )

        mlflow.set_tag(
            "target",
            information["target"]
        )

        mlflow.set_tag(
            "perimeter",
            information["perimeter"]
        )

        mlflow.set_tag(
            "strategy",
            artifact.get(
                "nom_modele",
                "unknown"
            )
        )

        mlflow.set_tag(
            "registered_at",
            datetime.now().isoformat()
        )


        # ----------------------------------------------------
        # Paramètres
        # ----------------------------------------------------

        mlflow.log_param(
            "artifact_file",
            information["file"]
        )

        mlflow.log_param(
            "model_directory",
            str(MODELS_DIR)
        )


        # ----------------------------------------------------
        # Métadonnées
        # ----------------------------------------------------

        metadata = {

            "model_name":
                model_name,

            "registry_name":
                registry_name,

            "artifact_file":
                information["file"],

            "target":
                information["target"],

            "perimeter":
                information["perimeter"],

            "description":
                information["description"],

            "artifact_keys":
                list(
                    artifact.keys()
                ),
        }


        # ----------------------------------------------------
        # Informations supplémentaires M1/M2
        # ----------------------------------------------------

        if model_name in {
            "M1",
            "M2"
        }:

            metadata[
                "features"
            ] = artifact.get(
                "features",
                []
            )

            metadata[
                "features_categorielles"
            ] = artifact.get(
                "features_categorielles",
                []
            )

            metadata[
                "features_numeriques"
            ] = artifact.get(
                "features_numeriques",
                []
            )


        # ----------------------------------------------------
        # Informations supplémentaires M3/M4
        # ----------------------------------------------------

        if model_name in {
            "M3",
            "M4"
        }:

            metadata[
                "cles"
            ] = artifact.get(
                "cles",
                []
            )

            baseline = artifact.get(
                "baseline"
            )

            if baseline is not None:

                metadata[
                    "baseline_shape"
                ] = list(
                    baseline.shape
                )


        # ----------------------------------------------------
        # JSON metadata
        # ----------------------------------------------------

        metadata_file = (
            REGISTRY_DIR
            / f"{model_name}_metadata.json"
        )

        with open(
            metadata_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                metadata,
                file,
                indent=4,
                ensure_ascii=False,
                default=str
            )


        # ----------------------------------------------------
        # Enregistrement de l'artefact joblib
        # ----------------------------------------------------

        mlflow.log_artifact(
            str(model_file),
            artifact_path=model_name
        )

        mlflow.log_artifact(
            str(metadata_file),
            artifact_path="metadata"
        )


        # ----------------------------------------------------
        # Résultat
        # ----------------------------------------------------

        result = {

            "model":
                model_name,

            "registry_name":
                registry_name,

            "run_id":
                run.info.run_id,

            "artifact":
                str(model_file),

            "target":
                information["target"],

            "perimeter":
                information["perimeter"],

        }

    return result


# ============================================================
# VALIDATION
# ============================================================

def validate_registry(results):

    print()
    print(
        "=" * 90
    )

    print(
        "VALIDATION DU REGISTRE"
    )

    print(
        "=" * 90
    )

    for result in results:

        print()
        print(
            f"{result['model']}"
        )

        print(
            "Registry name :",
            result["registry_name"]
        )

        print(
            "Run ID :",
            result["run_id"]
        )

        print(
            "Artefact :",
            result["artifact"]
        )

        if not Path(
            result["artifact"]
        ).exists():

            raise FileNotFoundError(
                result["artifact"]
            )


    print()
    print(
        "Les 4 modèles sont enregistrés."
    )


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "=" * 90
    )

    print(
        "ENREGISTREMENT M1 / M2 / M3 / M4"
    )

    print(
        "=" * 90
    )

    results = []


    for model_name, information in (
        MODEL_FILES.items()
    ):

        print()
        print(
            "-" * 70
        )

        print(
            f"REGISTREMENT : {model_name}"
        )

        print(
            "-" * 70
        )

        result = register_model(
            model_name,
            information
        )

        results.append(
            result
        )

        print(
            "OK :",
            model_name
        )

        # ----------------------------------------------------
        # Libération
        # ----------------------------------------------------

        del result


    # ========================================================
    # VALIDATION
    # ========================================================

    validate_registry(
        results
    )


    # ========================================================
    # EXPORT DU MANIFEST
    # ========================================================

    manifest_file = (
        REGISTRY_DIR
        / "model_registry_manifest.json"
    )

    with open(
        manifest_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False
        )


    print()
    print(
        "Manifest :"
    )

    print(
        manifest_file
    )


    print()
    print(
        "=" * 90
    )

    print(
        "REGISTRE TERMINE"
    )

    print(
        "=" * 90
    )