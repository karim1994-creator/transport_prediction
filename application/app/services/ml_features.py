# ============================================================
# FEATURES COMMUNES M1 / M2
# ============================================================
#
# Objectif :
#   Garantir que l'entraînement et l'inférence utilisent
#   exactement la même construction des variables.
#
# M1 : Ferré
# M2 : Surface
#
# 24 features :
#   15 features de base
#   + 9 variables historiques
#
# ============================================================

import numpy as np
import pandas as pd


# ============================================================
# LISTES DE FEATURES
# ============================================================

BASE_FEATURES_M1 = [

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
]


BASE_FEATURES_M2 = [

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
]


HISTORICAL_FEATURES = [

    "moyenne_historique_causale",

    "mediane_historique_causale",

    "moyenne_historique_cat_causale",

    "mediane_historique_cat_causale",

    "lag_1",

    "lag_7",

    "lag_14",

    "moyenne_mobile_7",

    "moyenne_mobile_28",
]


FEATURES_M1 = (
    BASE_FEATURES_M1
    + HISTORICAL_FEATURES
)


FEATURES_M2 = (
    BASE_FEATURES_M2
    + HISTORICAL_FEATURES
)


CATEGORICAL_FEATURES_M1 = [

    "code_arret",
    "id_zdc",
    "cat_jour",
    "code_meteo",
]


CATEGORICAL_FEATURES_M2 = [

    "code_ligne",
    "id_groupoflines",
    "cat_jour",
    "code_meteo",
]


# ============================================================
# FEATURES TEMPORELLES
# ============================================================

def add_temporal_features(df):

    df = df.copy()

    if "jour" not in df.columns:

        raise ValueError(
            "Colonne 'jour' obligatoire."
        )

    df["jour"] = pd.to_datetime(
        df["jour"],
        errors="coerce"
    )

    if df["jour"].isna().any():

        raise ValueError(
            "Certaines valeurs de 'jour' sont invalides."
        )

    df["annee"] = (
        df["jour"].dt.year
    )

    df["mois"] = (
        df["jour"].dt.month
    )

    df["trimestre"] = (
        df["jour"].dt.quarter
    )

    df["semaine"] = (
        df["jour"]
        .dt
        .isocalendar()
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
# FEATURES HISTORIQUES M1 / M2
# ============================================================

def add_historical_features(
    df,
    group_cols,
    cat_group_cols,
    target_col="nb_vald",
):

    df = df.copy()

    required = (
        list(group_cols)
        + list(cat_group_cols)
        + [
            "jour",
            target_col,
        ]
    )

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "Colonnes nécessaires absentes : "
            f"{missing}"
        )

    df = df.sort_values(
        [
            *group_cols,
            "jour",
        ]
    ).copy()


    # ========================================================
    # HISTORIQUE GLOBAL
    # ========================================================

    df[
        "moyenne_historique_causale"
    ] = (

        df
        .groupby(
            list(group_cols)
        )[target_col]
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
            list(group_cols)
        )[target_col]
        .transform(
            lambda x:
            x
            .shift(1)
            .expanding()
            .median()
        )
    )


    # ========================================================
    # HISTORIQUE PAR CATEGORIE DE JOUR
    # ========================================================

    df[
        "moyenne_historique_cat_causale"
    ] = (

        df
        .groupby(
            list(cat_group_cols)
        )[target_col]
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
            list(cat_group_cols)
        )[target_col]
        .transform(
            lambda x:
            x
            .shift(1)
            .expanding()
            .median()
        )
    )


    # ========================================================
    # LAGS
    # ========================================================

    grouped_target = df.groupby(
        list(group_cols)
    )[target_col]


    df["lag_1"] = (
        grouped_target.shift(1)
    )

    df["lag_7"] = (
        grouped_target.shift(7)
    )

    df["lag_14"] = (
        grouped_target.shift(14)
    )


    # ========================================================
    # MOYENNES MOBILES CAUSALES
    # ========================================================

    df["moyenne_mobile_7"] = (

        grouped_target
        .transform(
            lambda x:
            x
            .shift(1)
            .rolling(
                7,
                min_periods=1
            )
            .mean()
        )
    )


    df["moyenne_mobile_28"] = (

        grouped_target
        .transform(
            lambda x:
            x
            .shift(1)
            .rolling(
                28,
                min_periods=1
            )
            .mean()
        )
    )


    # ========================================================
    # REMPLACEMENT NaN
    # ========================================================

    df[HISTORICAL_FEATURES] = (

        df[HISTORICAL_FEATURES]

        .replace(
            [np.inf, -np.inf],
            np.nan
        )

        .fillna(0)
    )

    return df


# ============================================================
# PREPARATION M1
# ============================================================

def prepare_m1_features(df):

    df = add_temporal_features(
        df
    )

    df = add_historical_features(

        df,

        group_cols=[
            "code_arret",
            "id_zdc",
        ],

        cat_group_cols=[
            "code_arret",
            "id_zdc",
            "cat_jour",
        ],
    )

    return df


# ============================================================
# PREPARATION M2
# ============================================================

def prepare_m2_features(df):

    df = add_temporal_features(
        df
    )

    df = add_historical_features(

        df,

        group_cols=[
            "code_ligne",
            "id_groupoflines",
        ],

        cat_group_cols=[
            "code_ligne",
            "id_groupoflines",
            "cat_jour",
        ],
    )

    return df


# ============================================================
# VERIFICATION FEATURES
# ============================================================

def verify_features(
    df,
    features,
    model_name,
):

    missing = [
        col
        for col in features
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{model_name}: variables manquantes : "
            f"{missing}"
        )

    return True