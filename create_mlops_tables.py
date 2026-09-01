from pathlib import Path
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


# ============================================================
# CHEMIN .ENV
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
# CONNEXION
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
# LECTURE DU SQL
# ============================================================

SQL_FILE = (
    BASE_DIR
    / "sql"
    / "create_mlops_tables.sql"
)

sql = SQL_FILE.read_text(
    encoding="utf-8"
)


# ============================================================
# EXECUTION
# ============================================================

print("=" * 80)
print("CREATION DES TABLES MLOPS")
print("=" * 80)

with engine.begin() as connection:

    for statement in sql.split(";"):

        statement = statement.strip()

        if statement:

            connection.execute(
                text(statement)
            )


# ============================================================
# VERIFICATION
# ============================================================

verification = """
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'prediction'
  AND table_name IN (
      'ml_prediction_history',
      'ml_model_versions'
  )
ORDER BY table_name
"""

with engine.connect() as connection:

    rows = connection.execute(
        text(verification)
    ).fetchall()


print()
print("Tables MLOps présentes :")

for row in rows:

    print(
        " -",
        row[0]
    )


print()
print("=" * 80)
print("CREATION TERMINEE")
print("=" * 80)