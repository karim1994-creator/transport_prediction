from pathlib import Path
import shutil


BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR / "models_a53"

CANDIDATE_DIR = MODELS_DIR / "candidate"


MODEL_FILES = {
    "M1": "M1_hgb_historique_causal.joblib",
    "M2": "M2_hgb_historique_causal.joblib",
}


def prepare_candidate(model_name):

    source = (
        MODELS_DIR
        / MODEL_FILES[model_name]
    )

    destination = (
        CANDIDATE_DIR
        / MODEL_FILES[model_name]
    )

    if not source.exists():
        raise FileNotFoundError(
            f"Modèle absent : {source}"
        )

    CANDIDATE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    shutil.copy2(
        source,
        destination
    )

    print(
        f"{model_name} candidat préparé :"
    )

    print(
        destination
    )


def main():

    print("=" * 80)
    print("PREPARATION DES MODELES CANDIDATS")
    print("=" * 80)

    prepare_candidate("M1")
    prepare_candidate("M2")

    print()
    print("Préparation terminée.")


if __name__ == "__main__":
    main()