from pathlib import Path
import json
import shutil
import subprocess
import sys
from datetime import datetime

from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os


# ============================================================
# CHEMINS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

ENV_FILE = (
    BASE_DIR
    / "application"
    / "config"
    / ".env"
)

load_dotenv(ENV_FILE)


CI_DIR = BASE_DIR / "ci_reports"

MODELS_DIR = BASE_DIR / "models_a53"

BACKUP_DIR = MODELS_DIR / "backup"

CANDIDATE_DIR = MODELS_DIR / "candidate"


CURRENT_SNAPSHOT = (
    CI_DIR
    / "training_data_snapshot.json"
)

PREVIOUS_SNAPSHOT = (
    CI_DIR
    / "training_data_previous_snapshot.json"
)

REFERENCE_PERFORMANCE = (
    CI_DIR
    / "reference_model_performance.json"
)

CANDIDATE_PERFORMANCE = (
    CI_DIR
    / "candidate_model_performance.json"
)

PROMOTION_REPORT = (
    CI_DIR
    / "model_promotion_report.json"
)

PIPELINE_REPORT = (
    CI_DIR
    / "ml_lifecycle_pipeline_report.json"
)


# ============================================================
# MODELES
# ============================================================

MODEL_FILES = {

    "M1":
        "M1_hgb_historique_causal.joblib",

    "M2":
        "M2_hgb_historique_causal.joblib",
}


# ============================================================
# POSTGRESQL
# ============================================================

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")


DATABASE_URL = (
    f"postgresql+psycopg2://"
    f"{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)


engine = create_engine(
    DATABASE_URL
)


# ============================================================
# SOURCES D'APPRENTISSAGE
# ============================================================

SOURCES = {

    "M1": {
        "table":
            "vues_metier.ferre_analyse",

        "date_column":
            "jour",
    },

    "M2": {
        "table":
            "vues_metier.surface_analyse",

        "date_column":
            "jour",
    },

    "M3": {
        "table":
            "transport.fa_profils_horaires_ferre",

        "date_column":
            "date_debut",
    },

    "M4": {
        "table":
            "transport.fa_profils_horaires_surface",

        "date_column":
            "date_debut",
    },
}


# ============================================================
# EXECUTION COMMANDE
# ============================================================

def run_command(
    command,
    title,
):

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    print(
        " ".join(command)
    )

    result = subprocess.run(
        command,
        cwd=BASE_DIR,
        check=False,
    )

    if result.returncode != 0:

        raise RuntimeError(
            f"Echec : {title}"
        )


# ============================================================
# SNAPSHOT POSTGRESQL
# ============================================================

def collect_snapshot():

    snapshot = {

        "snapshot_timestamp":
            datetime.now().isoformat(),

        "sources": {},
    }

    print()
    print("=" * 80)
    print("COLLECTE DES NOUVELLES DONNEES")
    print("=" * 80)

    for model_name, config in SOURCES.items():

        query = text(
            f"""
            SELECT
                MIN({config["date_column"]})
                    AS date_min,

                MAX({config["date_column"]})
                    AS date_max,

                COUNT(*)
                    AS nb_lignes

            FROM {config["table"]}
            """
        )

        with engine.connect() as connection:

            row = (
                connection
                .execute(query)
                .mappings()
                .one()
            )

        snapshot["sources"][model_name] = {

            "table":
                config["table"],

            "date_column":
                config["date_column"],

            "date_min":
                str(row["date_min"]),

            "date_max":
                str(row["date_max"]),

            "nb_lignes":
                int(row["nb_lignes"]),
        }

        print(
            model_name,
            "=>",
            row["date_max"],
            "|",
            row["nb_lignes"],
            "lignes"
        )

    CI_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        CURRENT_SNAPSHOT,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            snapshot,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return snapshot


# ============================================================
# DETECTION
# ============================================================

def detect_new_data(
    current_snapshot,
):

    if not PREVIOUS_SNAPSHOT.exists():

        print()
        print(
            "Aucun snapshot précédent."
        )

        print(
            "Première collecte : aucune mise à jour "
            "automatique déclenchée."
        )

        return False

    with open(
        PREVIOUS_SNAPSHOT,
        "r",
        encoding="utf-8",
    ) as file:

        previous_snapshot = json.load(
            file
        )

    detected = False

    print()
    print("=" * 80)
    print("DETECTION DES NOUVELLES DONNEES")
    print("=" * 80)

    for model_name in SOURCES:

        old = previous_snapshot[
            "sources"
        ][model_name]

        new = current_snapshot[
            "sources"
        ][model_name]

        old_date = old[
            "date_max"
        ]

        new_date = new[
            "date_max"
        ]

        old_rows = old[
            "nb_lignes"
        ]

        new_rows = new[
            "nb_lignes"
        ]

        date_changed = (
            new_date
            > old_date
        )

        rows_changed = (
            new_rows
            > old_rows
        )

        changed = (
            date_changed
            or rows_changed
        )

        print()
        print(
            model_name
        )

        print(
            "Ancienne date :",
            old_date
        )

        print(
            "Nouvelle date :",
            new_date
        )

        print(
            "Ancien volume :",
            old_rows
        )

        print(
            "Nouveau volume :",
            new_rows
        )

        print(
            "Nouvelles données :",
            "OUI" if changed else "NON"
        )

        if changed:

            detected = True

    return detected


# ============================================================
# BACKUP
# ============================================================

def backup_models():

    BACKUP_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("=" * 80)
    print("BACKUP DES MODELES DE PRODUCTION")
    print("=" * 80)

    for model_name, filename in MODEL_FILES.items():

        source = (
            MODELS_DIR
            / filename
        )

        destination = (
            BACKUP_DIR
            / filename
        )

        if not source.exists():

            raise FileNotFoundError(
                f"Modèle absent : {source}"
            )

        shutil.copy2(
            source,
            destination
        )

        print(
            model_name,
            "→ backup"
        )


# ============================================================
# COPIE CANDIDAT
# ============================================================

def prepare_candidates():

    CANDIDATE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("=" * 80)
    print("PREPARATION DES CANDIDATS")
    print("=" * 80)

    for model_name, filename in MODEL_FILES.items():

        source = (
            MODELS_DIR
            / filename
        )

        destination = (
            CANDIDATE_DIR
            / filename
        )

        if not source.exists():

            raise FileNotFoundError(
                f"Artefact absent : {source}"
            )

        shutil.copy2(
            source,
            destination
        )

        print(
            model_name,
            "→ candidate"
        )


# ============================================================
# RESTAURATION
# ============================================================

def restore_models():

    print()
    print("=" * 80)
    print("RESTAURATION DES MODELES DE PRODUCTION")
    print("=" * 80)

    for model_name, filename in MODEL_FILES.items():

        source = (
            BACKUP_DIR
            / filename
        )

        destination = (
            MODELS_DIR
            / filename
        )

        if not source.exists():

            raise FileNotFoundError(
                f"Backup absent : {source}"
            )

        shutil.copy2(
            source,
            destination
        )

        print(
            model_name,
            "→ ancien modèle restauré"
        )


# ============================================================
# PROMOTION
# ============================================================

def promote_models():

    print()
    print("=" * 80)
    print("PROMOTION DES MODELES VALIDES")
    print("=" * 80)

    for model_name, filename in MODEL_FILES.items():

        source = (
            CANDIDATE_DIR
            / filename
        )

        destination = (
            MODELS_DIR
            / filename
        )

        if not source.exists():

            raise FileNotFoundError(
                f"Candidat absent : {source}"
            )

        shutil.copy2(
            source,
            destination
        )

        print(
            model_name,
            "→ PROMOTE"
        )


# ============================================================
# SNAPSHOT FINAL
# ============================================================

def update_previous_snapshot(
    snapshot,
):

    with open(
        PREVIOUS_SNAPSHOT,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            snapshot,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# PIPELINE
# ============================================================

def main():

    started_at = datetime.now()

    print()
    print("=" * 100)
    print("PIPELINE AUTOMATIQUE DU CYCLE DE VIE ML")
    print("=" * 100)

    pipeline_status = "NO_NEW_DATA"
    promotion_status = {}

    # --------------------------------------------------------
    # 1. COLLECTE
    # --------------------------------------------------------

    current_snapshot = collect_snapshot()

    # --------------------------------------------------------
    # 2. DETECTION
    # --------------------------------------------------------

    new_data = detect_new_data(
        current_snapshot
    )

    # Première exécution :
    # on initialise la référence sans entraîner.
    if not PREVIOUS_SNAPSHOT.exists():

        update_previous_snapshot(
            current_snapshot
        )

        pipeline_status = (
            "FIRST_SNAPSHOT"
        )

        print()
        print(
            "Première exécution : snapshot initialisé."
        )

        return

    # --------------------------------------------------------
    # 3. AUCUNE DONNEE NOUVELLE
    # --------------------------------------------------------

    if not new_data:

        pipeline_status = (
            "NO_NEW_DATA"
        )

        print()
        print(
            "Aucune nouvelle donnée."
        )

        print(
            "Aucun réentraînement lancé."
        )

        print()
        print("=" * 100)
        print(
            "PIPELINE TERMINEE"
        )
        print("=" * 100)

        return

    # --------------------------------------------------------
    # 4. NOUVELLES DONNEES
    # --------------------------------------------------------

    pipeline_status = (
        "NEW_DATA_DETECTED"
    )

    print()
    print(
        "NOUVELLES DONNEES DETECTEES."
    )

    # --------------------------------------------------------
    # 5. BACKUP
    # --------------------------------------------------------

    backup_models()

    # --------------------------------------------------------
    # 6. REENTRAINEMENT
    # --------------------------------------------------------

    run_command(
        [
            sys.executable,
            "train_models_ML.py",
        ],
        "REENTRAINEMENT DES MODELES",
    )

    # --------------------------------------------------------
    # 7. CANDIDATS
    # --------------------------------------------------------

    prepare_candidates()

    # --------------------------------------------------------
    # 8. EVALUATION
    # --------------------------------------------------------

    run_command(
        [
            sys.executable,
            "evaluate_candidate_performance.py",
        ],
        "EVALUATION DES MODELES CANDIDATS",
    )

    # --------------------------------------------------------
    # 9. COMPARAISON
    # --------------------------------------------------------

    reference = None
    candidate = None

    with open(
        REFERENCE_PERFORMANCE,
        "r",
        encoding="utf-8",
    ) as file:

        reference = json.load(
            file
        )

    with open(
        CANDIDATE_PERFORMANCE,
        "r",
        encoding="utf-8",
    ) as file:

        candidate = json.load(
            file
        )

    print()
    print("=" * 80)
    print("DECISION DE PROMOTION")
    print("=" * 80)

    global_acceptance = True

    for old in reference:

        model_name = old[
            "model"
        ]

        new = next(
            item
            for item in candidate
            if item["model"] == model_name
        )

        tolerance = 0.001

        mae_ok = (
            new["MAE"]
            <= old["MAE"] * (1 + tolerance)
        )

        rmse_ok = (
            new["RMSE"]
            <= old["RMSE"] * (1 + tolerance)
        )

        r2_ok = (
            new["R2"]
            >= old["R2"] * (1 - tolerance)
        )

        accepted = (
            mae_ok
            and rmse_ok
            and r2_ok
        )

        promotion_status[
            model_name
        ] = {

            "accepted":
                accepted,

            "MAE":
                {
                    "old":
                        old["MAE"],

                    "new":
                        new["MAE"],

                    "ok":
                        mae_ok,
                },

            "RMSE":
                {
                    "old":
                        old["RMSE"],

                    "new":
                        new["RMSE"],

                    "ok":
                        rmse_ok,
                },

            "R2":
                {
                    "old":
                        old["R2"],

                    "new":
                        new["R2"],

                    "ok":
                        r2_ok,
                },
        }

        print()
        print(
            model_name,
            "→",
            "PROMOTE"
            if accepted
            else "REJECT"
        )

        if not accepted:

            global_acceptance = False

    # --------------------------------------------------------
    # 10. PROMOTION OU RESTAURATION
    # --------------------------------------------------------

    if global_acceptance:

        promote_models()

        pipeline_status = (
            "PROMOTED"
        )

    else:

        restore_models()

        pipeline_status = (
            "REJECTED_RESTORED"
        )

    # --------------------------------------------------------
    # 11. SNAPSHOT
    # --------------------------------------------------------

    update_previous_snapshot(
        current_snapshot
    )

    # --------------------------------------------------------
    # 12. RAPPORT
    # --------------------------------------------------------

    finished_at = datetime.now()

    report = {

        "started_at":
            started_at.isoformat(),

        "finished_at":
            finished_at.isoformat(),

        "status":
            pipeline_status,

        "new_data_detected":
            new_data,

        "promotion":
            promotion_status,
    }

    with open(
        PIPELINE_REPORT,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 100)
    print(
        "STATUT PIPELINE :",
        pipeline_status
    )

    print(
        "Rapport :",
        PIPELINE_REPORT
    )

    print("=" * 100)


if __name__ == "__main__":

    main()