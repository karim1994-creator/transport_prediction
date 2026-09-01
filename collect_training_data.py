from datetime import datetime

from sqlalchemy import create_engine, text

from app.config import DATABASE_URL


engine = create_engine(
    DATABASE_URL
)


QUERIES = {

    "M1": """
        SELECT
            MIN(jour) AS date_min,
            MAX(jour) AS date_max,
            COUNT(*) AS lignes
        FROM vues_metier.ferre_analyse
    """,

    "M2": """
        SELECT
            MIN(jour) AS date_min,
            MAX(jour) AS date_max,
            COUNT(*) AS lignes
        FROM vues_metier.surface_analyse
    """,

    "M3": """
        SELECT
            MIN(date_debut) AS date_min,
            MAX(date_debut) AS date_max,
            COUNT(*) AS lignes
        FROM transport.fa_profils_horaires_ferre
    """,

    "M4": """
        SELECT
            MIN(date_debut) AS date_min,
            MAX(date_debut) AS date_max,
            COUNT(*) AS lignes
        FROM transport.fa_profils_horaires_surface
    """,
}


def collect():

    print(
        "=" * 80
    )

    print(
        "CONTROLE DES NOUVELLES DONNEES D'APPRENTISSAGE"
    )

    print(
        "=" * 80
    )

    for model, query in QUERIES.items():

        with engine.connect() as connection:

            result = connection.execute(
                text(query)
            ).mappings().one()

        print()
        print(
            f"{model}"
        )

        print(
            "Date min :",
            result["date_min"]
        )

        print(
            "Date max :",
            result["date_max"]
        )

        print(
            "Nombre de lignes :",
            result["lignes"]
        )

    print()
    print(
        "CONTROLE TERMINE"
    )


if __name__ == "__main__":

    collect()