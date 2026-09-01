from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# ============================================================
# IMPORT CONFIG
# ============================================================

APPLICATION_DIR = (
    Path(__file__).resolve().parent / "application"
)

sys.path.insert(
    0,
    str(APPLICATION_DIR)
)

from app.config import DATABASE_URL


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

REFERENCE_DIR = (
    BASE_DIR
    / "ci_reports"
    / "reference_test_sets"
)

REFERENCE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

engine = create_engine(
    DATABASE_URL
)


# ============================================================
# FEATURES TEMPORELLES
# ============================================================

def add_temporal_features(df):

    df = df.copy()

    df["jour"] = pd.to_datetime(
        df["jour"],
        errors="coerce"
    )

    df["annee"] = df["jour"].dt.year
    df["mois"] = df["jour"].dt.month
    df["trimestre"] = df["jour"].dt.quarter

    df["semaine"] = (
        df["jour"]
        .dt.isocalendar()
        .week
        .astype("Int64")
    )

    df["jour_semaine"] = (
        df["jour"].dt.dayofweek
    )

    df["jour_du_mois"] = (
        df["jour"].dt.day
    )

    df["jour_annee"] = (
        df["jour"].dt.dayofyear
    )

    df["est_debut_mois"] = (
        df["jour"].dt.day <= 7
    ).astype(int)

    df["est_fin_mois"] = (
        df["jour"].dt.day >= 24
    ).astype(int)

    return df


# ============================================================
# SPLIT TEMPOREL
# ============================================================

def temporal_split(
    df,
    date_col="jour"
):

    df = df.sort_values(
        date_col
    ).copy()

    unique_dates = np.array(
        sorted(
            df[date_col]
            .dropna()
            .unique()
        )
    )

    n_dates = len(
        unique_dates
    )

    train_end = int(
        n_dates * 0.60
    )

    val_end = int(
        n_dates * 0.80
    )

    test_dates = unique_dates[
        val_end:
    ]

    test = df[
        df[date_col].isin(
            test_dates
        )
    ].copy()

    return test


# ============================================================
# PREPARATION M1
# ============================================================

def prepare_m1():

    query = """

    SELECT

        jour,

        code_arret,

        id_zdc,

        MAX(cat_jour)
            AS cat_jour,

        MAX(temperature_moyenne)
            AS temperature_moyenne,

        MAX(pluie_totale)
            AS pluie_totale,

        MAX(vitesse_vent_moyenne)
            AS vitesse_vent_moyenne,

        MAX(code_meteo)
            AS code_meteo,

        SUM(nb_vald)
            AS nb_vald

    FROM vues_metier.ferre_analyse

    GROUP BY

        jour,
        code_arret,
        id_zdc

    ORDER BY

        jour,
        code_arret,
        id_zdc

    """

    df = pd.read_sql(
        query,
        engine
    )

    df = add_temporal_features(
        df
    )

    df = df.sort_values(
        [
            "code_arret",
            "id_zdc",
            "jour",
        ]
    ).copy()

    df[
        "moyenne_historique_causale"
    ] = (

        df
        .groupby(
            [
                "code_arret",
                "id_zdc",
            ]
        )["nb_vald"]
        .transform(
            lambda x:
                x
                .shift(1)
                .expanding()
                .mean()
        )
    )

    df[
        "mediane_historique_causale"
    ] = (

        df
        .groupby(
            [
                "code_arret",
                "id_zdc",
            ]
        )["nb_vald"]
        .transform(
            lambda x:
                x
                .shift(1)
                .expanding()
                .median()
        )
    )

    df[
        "moyenne_historique_cat_causale"
    ] = (

        df
        .groupby(
            [
                "code_arret",
                "id_zdc",
                "cat_jour",
            ]
        )["nb_vald"]
        .transform(
            lambda x:
                x
                .shift(1)
                .expanding()
                .mean()
        )
    )

    df[
        "mediane_historique_cat_causale"
    ] = (

        df
        .groupby(
            [
                "code_arret",
                "id_zdc",
                "cat_jour",
            ]
        )["nb_vald"]
        .transform(
            lambda x:
                x
                .shift(1)
                .expanding()
                .median()
        )
    )

    historical_cols = [

        "moyenne_historique_causale",

        "mediane_historique_causale",

        "moyenne_historique_cat_causale",

        "mediane_historique_cat_causale",
    ]

    df[
        historical_cols
    ] = (
        df[
            historical_cols
        ]
        .fillna(0)
    )

    test = temporal_split(
        df
    )

    features = [

        "code_arret",
        "id_zdc",
        "mois",
        "trimestre",
        "semaine",
        "jour_semaine",
        "jour_du_mois",
        "jour_annee",
        "est_debut_mois",
        "est_fin_mois",
        "temperature_moyenne",
        "pluie_totale",
        "vitesse_vent_moyenne",
        "cat_jour",
        "code_meteo",
        "moyenne_historique_causale",
        "mediane_historique_causale",
        "moyenne_historique_cat_causale",
        "mediane_historique_cat_causale",
    ]

    test[
        features
    ].to_parquet(
        REFERENCE_DIR
        / "M1_X_test.parquet"
    )

    test[
        ["nb_vald"]
    ].to_parquet(
        REFERENCE_DIR
        / "M1_y_test.parquet"
    )

    print(
        "M1 test :",
        test.shape
    )

    print(
        "M1 dates :",
        test["jour"].min(),
        "->",
        test["jour"].max()
    )


# ============================================================
# PREPARATION M2
# ============================================================

def prepare_m2():

    query = """

    SELECT

        jour,

        code_ligne,

        id_groupoflines,

        MAX(cat_jour)
            AS cat_jour,

        MAX(temperature_moyenne)
            AS temperature_moyenne,

        MAX(pluie_totale)
            AS pluie_totale,

        MAX(vitesse_vent_moyenne)
            AS vitesse_vent_moyenne,

        MAX(code_meteo)
            AS code_meteo,

        SUM(nb_vald)
            AS nb_vald

    FROM vues_metier.surface_analyse

    GROUP BY

        jour,
        code_ligne,
        id_groupoflines

    ORDER BY

        jour,
        code_ligne,
        id_groupoflines

    """

    df = pd.read_sql(
        query,
        engine
    )

    df = add_temporal_features(
        df
    )

    df = df.sort_values(
        [
            "code_ligne",
            "id_groupoflines",
            "jour",
        ]
    ).copy()

    df[
        "moyenne_historique_causale"
    ] = (

        df
        .groupby(
            [
                "code_ligne",
                "id_groupoflines",
            ]
        )["nb_vald"]
        .transform(
            lambda x:
                x
                .shift(1)
                .expanding()
                .mean()
        )
    )

    df[
        "mediane_historique_causale"
    ] = (

        df
        .groupby(
            [
                "code_ligne",
                "id_groupoflines",
            ]
        )["nb_vald"]
        .transform(
            lambda x:
                x
                .shift(1)
                .expanding()
                .median()
        )
    )

    df[
        "moyenne_historique_cat_causale"
    ] = (

        df
        .groupby(
            [
                "code_ligne",
                "id_groupoflines",
                "cat_jour",
            ]
        )["nb_vald"]
        .transform(
            lambda x:
                x
                .shift(1)
                .expanding()
                .mean()
        )
    )

    df[
        "mediane_historique_cat_causale"
    ] = (

        df
        .groupby(
            [
                "code_ligne",
                "id_groupoflines",
                "cat_jour",
            ]
        )["nb_vald"]
        .transform(
            lambda x:
                x
                .shift(1)
                .expanding()
                .median()
        )
    )

    historical_cols = [

        "moyenne_historique_causale",

        "mediane_historique_causale",

        "moyenne_historique_cat_causale",

        "mediane_historique_cat_causale",
    ]

    df[
        historical_cols
    ] = (
        df[
            historical_cols
        ]
        .fillna(0)
    )

    test = temporal_split(
        df
    )

    features = [

        "code_ligne",
        "id_groupoflines",
        "mois",
        "trimestre",
        "semaine",
        "jour_semaine",
        "jour_du_mois",
        "jour_annee",
        "est_debut_mois",
        "est_fin_mois",
        "temperature_moyenne",
        "pluie_totale",
        "vitesse_vent_moyenne",
        "cat_jour",
        "code_meteo",
        "moyenne_historique_causale",
        "mediane_historique_causale",
        "moyenne_historique_cat_causale",
        "mediane_historique_cat_causale",
    ]

    test[
        features
    ].to_parquet(
        REFERENCE_DIR
        / "M2_X_test.parquet"
    )

    test[
        ["nb_vald"]
    ].to_parquet(
        REFERENCE_DIR
        / "M2_y_test.parquet"
    )

    print(
        "M2 test :",
        test.shape
    )

    print(
        "M2 dates :",
        test["jour"].min(),
        "->",
        test["jour"].max()
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 90)
    print("CREATION DES JEUX DE TEST DE REFERENCE")
    print("=" * 90)

    prepare_m1()
    prepare_m2()

    print()
    print("=" * 90)
    print("JEUX DE TEST CREES")
    print("=" * 90)