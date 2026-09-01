from pathlib import Path
import os
import json
from datetime import datetime

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

ENV_FILE = (
    BASE_DIR
    / "application"
    / "config"
    / ".env"
)

load_dotenv(ENV_FILE)


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
# SOURCES
# ============================================================

SOURCES = {

    "M1": {
        "table": "vues_metier.ferre_analyse",
        "date_column": "jour",
    },

    "M2": {
        "table": "vues_metier.surface_analyse",
        "date_column": "jour",
    },

    "M3": {
        "table": "transport.fa_profils_horaires_ferre",
        "date_column": "date_debut",
    },

    "M4": {
        "table": "transport.fa_profils_horaires_surface",
        "date_column": "date_debut",
    },
}


# ============================================================
# COLLECTE
# ============================================================

snapshot = {
    "snapshot_timestamp":
        datetime.now().isoformat(),

    "sources": {}
}


print("=" * 90)
print("COLLECTE DU SNAPSHOT DES DONNEES D'APPRENTISSAGE")
print("=" * 90)


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

    print()
    print("-" * 70)
    print(model_name)
    print("-" * 70)

    with engine.connect() as connection:

        row = (
            connection
            .execute(query)
            .mappings()
            .one()
        )

    date_min = row["date_min"]
    date_max = row["date_max"]
    nb_lignes = row["nb_lignes"]

    print(
        "Date min :",
        date_min
    )

    print(
        "Date max :",
        date_max
    )

    print(
        "Nombre de lignes :",
        nb_lignes
    )

    snapshot["sources"][model_name] = {

        "table":
            config["table"],

        "date_column":
            config["date_column"],

        "date_min":
            str(date_min),

        "date_max":
            str(date_max),

        "nb_lignes":
            int(nb_lignes),
    }


# ============================================================
# SAUVEGARDE
# ============================================================

OUTPUT_DIR = (
    BASE_DIR
    / "ci_reports"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "training_data_snapshot.json"
)


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        snapshot,
        file,
        indent=2,
        ensure_ascii=False
    )


print()
print("=" * 90)
print("SNAPSHOT SAUVEGARDE")
print("=" * 90)

print(
    OUTPUT_FILE
)

print("=" * 90)