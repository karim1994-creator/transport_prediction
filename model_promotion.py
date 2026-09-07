from pathlib import Path
import shutil


BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR / "models_a53"

BACKUP_DIR = MODELS_DIR / "backup"


MODEL_FILES = {
    "M1": "M1_hgb_historique_causal.joblib",
    "M2": "M2_hgb_historique_causal.joblib",
    "M3": "M3_baseline_historique.joblib",
    "M4": "M4_baseline_historique.joblib",
}


def backup_model(model_name):

    if model_name not in MODEL_FILES:
        raise ValueError(f"Modèle inconnu : {model_name}")

    source = MODELS_DIR / MODEL_FILES[model_name]

    if not source.exists():
        raise FileNotFoundError(
            f"Modèle absent : {source}"
        )

    BACKUP_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    destination = (
        BACKUP_DIR
        / MODEL_FILES[model_name]
    )

    shutil.copy2(
        source,
        destination
    )

    print(
        f"{model_name} sauvegardé : {destination}"
    )


def main():

    print("=" * 80)
    print("SAUVEGARDE DES MODELES ACTUELS")
    print("=" * 80)

    backup_model("M1")
    backup_model("M2")

    print()
    print("Sauvegarde terminée.")


if __name__ == "__main__":
    main()