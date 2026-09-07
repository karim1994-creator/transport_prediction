from pathlib import Path
import shutil
import subprocess
import sys


BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR / "models_a53"
CANDIDATE_DIR = MODELS_DIR / "candidate"

MODEL_FILES = {
    "M1": "M1_hgb_historique_causal.joblib",
    "M2": "M2_hgb_historique_causal.joblib",
}


def backup_current_models():
    """
    Sauvegarde les modèles actuellement en production
    avant tout réentraînement.
    """

    backup_dir = MODELS_DIR / "backup"

    backup_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for model_name, filename in MODEL_FILES.items():

        source = MODELS_DIR / filename
        destination = backup_dir / filename

        if not source.exists():

            raise FileNotFoundError(
                f"Modèle de production absent : {source}"
            )

        shutil.copy2(
            source,
            destination
        )

        print(
            f"{model_name} : backup créé"
        )


def run_training():
    """
    Exécute le script d'entraînement existant.
    """

    print()
    print("=" * 80)
    print("REENTRAINEMENT DES MODELES")
    print("=" * 80)

    result = subprocess.run(
        [
            sys.executable,
            "train_models_ML.py",
        ],
        cwd=BASE_DIR,
        check=False,
    )

    if result.returncode != 0:

        raise RuntimeError(
            "Le réentraînement a échoué."
        )


def copy_to_candidates():
    """
    Copie les nouveaux artefacts vers le répertoire candidat.
    """

    CANDIDATE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for model_name, filename in MODEL_FILES.items():

        source = MODELS_DIR / filename
        destination = CANDIDATE_DIR / filename

        if not source.exists():

            raise FileNotFoundError(
                f"Artefact réentraîné absent : {source}"
            )

        shutil.copy2(
            source,
            destination
        )

        print(
            f"{model_name} : candidat créé"
        )


def main():

    print("=" * 80)
    print("PREPARATION DU REENTRAINEMENT")
    print("=" * 80)

    backup_current_models()

    print()
    print(
        "Backup terminé."
    )

    print()
    print(
        "IMPORTANT : le script suivant va réentraîner"
    )

    print(
        "les modèles et modifier temporairement les"
    )

    print(
        "artefacts présents dans models_a53/."
    )

    print()
    print(
        "AUCUN lancement automatique pour le moment."
    )


if __name__ == "__main__":
    main()