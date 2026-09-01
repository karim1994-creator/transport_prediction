from pathlib import Path
import json
import shutil
import subprocess
import sys
import os
from datetime import datetime

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(
    BASE_DIR
    / "application"
    / "config"
    / ".env"
)


CI_DIR = BASE_DIR / "ci_reports"

MODELS_DIR = BASE_DIR / "models_a53"

BACKUP_DIR = MODELS_DIR / "backup"

CANDIDATE_DIR = MODELS_DIR / "candidate"


PREVIOUS_SNAPSHOT = (
    CI_DIR
    / "training_data_previous_snapshot.json"
)

CURRENT_SNAPSHOT = (
    CI_DIR
    / "training_data_snapshot.json"
)

REFERENCE_PERFORMANCE = (
    CI_DIR
    / "reference_model_performance.json"
)

CANDIDATE_PERFORMANCE = (
    CI_DIR
    / "candidate_model_performance.json"
)

SIMULATION_FILE = (
    CI_DIR
    / "training_data_simulated.json"
)

SIMULATION_MODE = os.getenv(
    "ML_LIFECYCLE_SIMULATION",
    "false"
).lower() == "true"

DRY_RUN = os.getenv(
    "ML_LIFECYCLE_DRY_RUN",
    "false"
).lower() == "true"

PIPELINE_REPORT = (
    CI_DIR
    / "ml_lifecycle_m1_m2_report.json"
)



MODEL_FILES = {
    "M1": "M1_hgb_historique_causal.joblib",
    "M2": "M2_hgb_historique_causal.joblib",
}


SOURCES = {
    "M1": {
        "table": "vues_metier.ferre_analyse",
        "date_column": "jour",
    },
    "M2": {
        "table": "vues_metier.surface_analyse",
        "date_column": "jour",
    },
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
# COLLECTE SNAPSHOT M1/M2
# ============================================================

def collect_snapshot():

    snapshot = {
        "snapshot_timestamp":
            datetime.now().isoformat(),
        "sources": {},
    }

    for model_name, config in SOURCES.items():

        query = text(
            f"""
            SELECT
                MIN({config["date_column"]}) AS date_min,
                MAX({config["date_column"]}) AS date_max,
                COUNT(*) AS nb_lignes
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

        return {
            "M1": False,
            "M2": False,
        }

    with open(
        PREVIOUS_SNAPSHOT,
        "r",
        encoding="utf-8",
    ) as file:

        previous = json.load(file)

    result = {}

    for model_name in ["M1", "M2"]:

        old = previous[
            "sources"
        ][model_name]

        new = current_snapshot[
            "sources"
        ][model_name]

        result[model_name] = (
            new["date_max"]
            > old["date_max"]
            or
            new["nb_lignes"]
            > old["nb_lignes"]
        )

    return result


# ============================================================
# BACKUP
# ============================================================

def backup_models():

    BACKUP_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for filename in MODEL_FILES.values():

        source = MODELS_DIR / filename

        destination = BACKUP_DIR / filename

        if not source.exists():

            raise FileNotFoundError(
                source
            )

        shutil.copy2(
            source,
            destination
        )


# ============================================================
# REENTRAINEMENT
# ============================================================

def retrain():

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
            "Echec du réentraînement."
        )


# ============================================================
# COPIE CANDIDATS
# ============================================================

def create_candidates():

    CANDIDATE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for filename in MODEL_FILES.values():

        source = MODELS_DIR / filename

        destination = (
            CANDIDATE_DIR / filename
        )

        shutil.copy2(
            source,
            destination
        )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_candidates():

    result = subprocess.run(
        [
            sys.executable,
            "evaluate_candidate_performance.py",
        ],
        cwd=BASE_DIR,
        check=False,
    )

    if result.returncode != 0:

        raise RuntimeError(
            "Evaluation des candidats échouée."
        )


# ============================================================
# DECISION
# ============================================================

def evaluate_promotion():

    with open(
        REFERENCE_PERFORMANCE,
        "r",
        encoding="utf-8",
    ) as file:

        reference = json.load(file)

    with open(
        CANDIDATE_PERFORMANCE,
        "r",
        encoding="utf-8",
    ) as file:

        candidate = json.load(file)

    decisions = {}

    tolerance = 0.001

    for old in reference:

        model_name = old["model"]

        if model_name not in MODEL_FILES:
            continue

        new = next(
            item
            for item in candidate
            if item["model"] == model_name
        )

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

        decisions[model_name] = {
            "accepted":
                mae_ok and rmse_ok and r2_ok,

            "MAE":
                {
                    "old": old["MAE"],
                    "new": new["MAE"],
                    "ok": mae_ok,
                },

            "RMSE":
                {
                    "old": old["RMSE"],
                    "new": new["RMSE"],
                    "ok": rmse_ok,
                },

            "R2":
                {
                    "old": old["R2"],
                    "new": new["R2"],
                    "ok": r2_ok,
                },
        }

    return decisions


# ============================================================
# PROMOTION / RESTAURATION
# ============================================================

def promote():

    for filename in MODEL_FILES.values():

        source = CANDIDATE_DIR / filename

        destination = MODELS_DIR / filename

        shutil.copy2(
            source,
            destination
        )


def restore():

    for filename in MODEL_FILES.values():

        source = BACKUP_DIR / filename

        destination = MODELS_DIR / filename

        shutil.copy2(
            source,
            destination
        )


# ============================================================
# MAIN
# ============================================================

def main():

    start = datetime.now()

    print("=" * 90)
    print("PIPELINE AUTOMATIQUE M1 / M2")
    print("=" * 90)

    current = collect_snapshot()

    if SIMULATION_MODE and SIMULATION_FILE.exists():

        print()
        print("=" * 90)
        print("MODE SIMULATION ACTIVE")
        print("=" * 90)

        with open(
            SIMULATION_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            current = json.load(file)

    changes = detect_new_data(
        current
    )

    print()
    print("Détection :")
    print("M1 :", "OUI" if changes["M1"] else "NON")
    print("M2 :", "OUI" if changes["M2"] else "NON")

    if DRY_RUN:

        print()
        print("=" * 90)
        print("MODE DRY-RUN : AUCUNE MODIFICATION")
        print("=" * 90)

        print(
            "Réentraînement : NON"
        )

        print(
            "Promotion : NON"
        )

        print(
            "Restauration : NON"
        )

        print(
            "Snapshot précédent : NON MODIFIE"
        )

        report = {
            "status":
                "DRY_RUN",

            "simulation_mode":
                SIMULATION_MODE,

            "detected":
                changes,
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
        print(
            "Rapport :",
            PIPELINE_REPORT
        )

        return

    if not any(changes.values()):

        print()
        print(
            "Aucune nouvelle donnée."
        )

        print(
            "Aucun réentraînement."
        )

        print(
            "STATUT : NO_NEW_DATA"
        )

        update_snapshot = True

        with open(
            PREVIOUS_SNAPSHOT,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                current,
                file,
                indent=2,
                ensure_ascii=False,
            )

        report = {
            "status":
                "NO_NEW_DATA",

            "detected":
                changes,
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

        return

    print()
    print(
        "NOUVELLES DONNEES DETECTEES."
    )

    # --------------------------------------------------------
    # BACKUP
    # --------------------------------------------------------

    backup_models()

    # --------------------------------------------------------
    # REENTRAINEMENT
    # --------------------------------------------------------

    retrain()

    # --------------------------------------------------------
    # CANDIDATS
    # --------------------------------------------------------

    create_candidates()

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    evaluate_candidates()

    # --------------------------------------------------------
    # DECISION
    # --------------------------------------------------------

    decisions = evaluate_promotion()

    global_accept = all(
        item["accepted"]
        for item in decisions.values()
    )

    if global_accept:

        promote()

        status = "PROMOTED"

        print(
            "PROMOTION : OK"
        )

    else:

        restore()

        status = "REJECTED_RESTORED"

        print(
            "PROMOTION : REFUSEE"
        )

        print(
            "Ancien modèle restauré."
        )

    # --------------------------------------------------------
    # SNAPSHOT
    # --------------------------------------------------------

    with open(
        PREVIOUS_SNAPSHOT,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            current,
            file,
            indent=2,
            ensure_ascii=False,
        )

    report = {

        "started_at":
            start.isoformat(),

        "finished_at":
            datetime.now().isoformat(),

        "status":
            status,

        "detected":
            changes,

        "decisions":
            decisions,
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
    print("=" * 90)
    print(
        "PIPELINE TERMINEE :",
        status
    )
    print("=" * 90)


if __name__ == "__main__":
    main()