from pathlib import Path
import os

from dotenv import load_dotenv


# ============================================================
# REPERTOIRE DU PROJET
# ============================================================

PROJECT_DIR = (
    Path(__file__)
    .resolve()
    .parents[2]
)


# ============================================================
# VARIABLES D'ENVIRONNEMENT
# ============================================================

ENV_FILE = (
    PROJECT_DIR
    / "application"
    / "config"
    / ".env"
)

load_dotenv(
    ENV_FILE
)


# ============================================================
# MODELES
# ============================================================

MODELS_DIR = Path(
    os.getenv(
        "MODELS_DIR",
        PROJECT_DIR / "models_a53"
    )
)

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
# POSTGRESQL
# ============================================================

DB_HOST = os.getenv(
    "DB_HOST"
)

DB_PORT = os.getenv(
    "DB_PORT"
)

DB_NAME = os.getenv(
    "DB_NAME"
)

DB_USER = os.getenv(
    "DB_USER"
)

DB_PASSWORD = os.getenv(
    "DB_PASSWORD"
)


# ============================================================
# VALIDATION CONFIGURATION
# ============================================================

missing = []

for name, value in {

    "DB_HOST": DB_HOST,
    "DB_PORT": DB_PORT,
    "DB_NAME": DB_NAME,
    "DB_USER": DB_USER,
    "DB_PASSWORD": DB_PASSWORD,

}.items():

    if not value:
        missing.append(name)


if missing:

    raise RuntimeError(
        "Variables PostgreSQL manquantes : "
        + ", ".join(missing)
    )


# ============================================================
# URL SQLALCHEMY
# ============================================================

DATABASE_URL = (
    f"postgresql+psycopg2://"
    f"{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)