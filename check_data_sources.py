from pathlib import Path
import os

from sqlalchemy import create_engine, text
from dotenv import load_dotenv


# ============================================================
# CHEMIN DU .ENV
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

ENV_FILE = (
    BASE_DIR
    / "application"
    / "config"
    / ".env"
)

load_dotenv(ENV_FILE)


# ============================================================
# CONFIGURATION POSTGRESQL
# ============================================================

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")


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
        "Variables manquantes dans .env : "
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


engine = create_engine(
    DATABASE_URL
)


# ============================================================
# SOURCES DE DONNEES
# ============================================================

QUERIES = {

    "M1_Ferre": """
        SELECT
            MIN(jour) AS date_min,
            MAX(jour) AS date_max,
            COUNT(*) AS nb_lignes
        FROM vues_metier.ferre_analyse
    """,

    "M2_Surface": """
        SELECT
            MIN(jour) AS date_min,
            MAX(jour) AS date_max,
            COUNT(*) AS nb_lignes
        FROM vues_metier.surface_analyse
    """,

    "M3_Ferre_Profils": """
        SELECT
            MIN(date_debut) AS date_min,
            MAX(date_debut) AS date_max,
            COUNT(*) AS nb_lignes
        FROM transport.fa_profils_horaires_ferre
    """,

    "M4_Surface_Profils": """
        SELECT
            MIN(date_debut) AS date_min,
            MAX(date_debut) AS date_max,
            COUNT(*) AS nb_lignes
        FROM transport.fa_profils_horaires_surface
    """,
}


# ============================================================
# EXECUTION
# ============================================================

print("=" * 90)
print("VERIFICATION DES SOURCES DE DONNEES")
print("=" * 90)

print()
print("Base :", DB_NAME)
print("Hôte :", DB_HOST)
print("Port :", DB_PORT)

for name, query in QUERIES.items():

    print()
    print("-" * 70)
    print(name)
    print("-" * 70)

    try:

        with engine.connect() as connection:

            row = (
                connection
                .execute(
                    text(query)
                )
                .mappings()
                .one()
            )

        print(
            "Date min :",
            row["date_min"]
        )

        print(
            "Date max :",
            row["date_max"]
        )

        print(
            "Nombre de lignes :",
            row["nb_lignes"]
        )

    except Exception as exc:

        print(
            "ERREUR :",
            exc
        )


print()
print("=" * 90)
print("VERIFICATION TERMINEE")
print("=" * 90)