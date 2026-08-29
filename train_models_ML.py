# ============================================================
# TRAINING FINAL DES 4 MODELES
#
# M1 : Ferré   -> nb_vald
# M2 : Surface -> nb_vald
# M3 : Ferré   -> profil horaire des validations
# M4 : Surface -> profil horaire des validations
#
# ============================================================
#
# M1 / M2 :
#
# Méthode :
#     HistGradientBoostingRegressor
#
# Stratégie :
#     Historique causal
#
# Artefacts :
#
#     M1_hgb_historique_causal.joblib
#     M2_hgb_historique_causal.joblib
#
# ============================================================
#
# M3 / M4 :
#
# Méthode :
#     Baseline historique
#
# Pas de modèle ML.
#
# Artefacts :
#
#     M3_baseline_historique.joblib
#     M4_baseline_historique.joblib
#
# ============================================================
#
# IMPORTANT :
#
# Ce fichier est placé à la RACINE du projet.
#
# Structure :
#
# projet/
# ├── train_models_M1_M2_M3_M4.py
# ├── application/
# │   └── app/
# │       └── config.py
# ├── models_a53/
# └── ...
#
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
import sys
import gc

import numpy as np
import pandas as pd
import joblib

from sqlalchemy import create_engine

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# CONFIGURATION DU PROJET
#
# Le fichier est maintenant à la RACINE.
# ============================================================

PROJECT_DIR = (
    Path(__file__)
    .resolve()
    .parent
)


# ============================================================
# IMPORT CONFIGURATION
#
# application/
#     app/
#         config.py
# ============================================================

sys.path.insert(
    0,
    str(
        PROJECT_DIR
        / "application"
    )
)

from app.config import DATABASE_URL


# ============================================================
# CONFIGURATION GENERALE
# ============================================================

RANDOM_STATE = 42


BASE_DIR = PROJECT_DIR


MODELS_DIR = (
    BASE_DIR
    / "models_a53"
)


MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONNEXION BASE DE DONNEES
# ============================================================

engine = create_engine(
    DATABASE_URL
)


# ============================================================
# NOMS EXACTS DES ARTEFACTS
# ============================================================

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
# ============================================================
#                         M1 / M2
# ============================================================
# ============================================================


# ============================================================
# FONCTION FEATURES TEMPORELLES
# ============================================================

def add_temporal_features(df):

    df = df.copy()

    df["jour"] = pd.to_datetime(
        df["jour"],
        errors="coerce"
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
# SPLIT TEMPOREL
#
# 60 % TRAIN
# 20 % VALIDATION
# 20 % TEST
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

    train_dates = unique_dates[
        :train_end
    ]

    val_dates = unique_dates[
        train_end:val_end
    ]

    test_dates = unique_dates[
        val_end:
    ]

    train = df[
        df[date_col].isin(
            train_dates
        )
    ].copy()

    val = df[
        df[date_col].isin(
            val_dates
        )
    ].copy()

    test = df[
        df[date_col].isin(
            test_dates
        )
    ].copy()

    print()
    print(
        "DATES TRAIN :",
        train_dates[0],
        "->",
        train_dates[-1]
    )

    print(
        "DATES VALIDATION :",
        val_dates[0],
        "->",
        val_dates[-1]
    )

    print(
        "DATES TEST :",
        test_dates[0],
        "->",
        test_dates[-1]
    )

    print()

    print(
        "Shape TRAIN :",
        train.shape
    )

    print(
        "Shape VALIDATION :",
        val.shape
    )

    print(
        "Shape TEST :",
        test.shape
    )

    return (
        train,
        val,
        test
    )


# ============================================================
# METRIQUES
# ============================================================

def evaluate_model(
    model,
    X,
    y,
    model_name
):

    predictions = model.predict(
        X
    )

    predictions = np.clip(
        predictions,
        0,
        None
    )

    mae = mean_absolute_error(
        y,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y,
            predictions
        )
    )

    r2 = r2_score(
        y,
        predictions
    )

    print()
    print("=" * 60)
    print(model_name)
    print("=" * 60)

    print(
        f"MAE  : {mae:.4f}"
    )

    print(
        f"RMSE : {rmse:.4f}"
    )

    print(
        f"R²   : {r2:.4f}"
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


# ============================================================
# CREATION PIPELINE
# ============================================================

def create_model_pipeline(
    numeric_features,
    categorical_features
):

    preprocessor = ColumnTransformer(

        transformers=[

            (
                "num",

                "passthrough",

                numeric_features,
            ),

            (
                "cat",

                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),

                categorical_features,
            ),
        ]
    )

    model = HistGradientBoostingRegressor(

        loss="squared_error",

        learning_rate=0.08,

        max_iter=300,

        max_leaf_nodes=31,

        min_samples_leaf=30,

        l2_regularization=1.0,

        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(

        steps=[

            (
                "preprocessing",
                preprocessor,
            ),

            (
                "model",
                model,
            ),
        ]
    )

    return pipeline


# ============================================================
# M1 — FERRE
# ============================================================

def train_m1():

    print()
    print("=" * 60)
    print("M1 — FERRE — NB_VALD")
    print("=" * 60)

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

    print(
        "Shape M1 :",
        df.shape
    )

    # --------------------------------------------------------
    # HISTORIQUE CAUSAL
    # --------------------------------------------------------

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

    df[historical_cols] = (
        df[historical_cols]
        .fillna(0)
    )

    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    train, val, test = temporal_split(
        df
    )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

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

    categorical_features = [

        "code_arret",

        "id_zdc",

        "cat_jour",

        "code_meteo",
    ]

    numeric_features = [

        col

        for col in features

        if col not in categorical_features
    ]

    # --------------------------------------------------------
    # PIPELINE
    # --------------------------------------------------------

    model = create_model_pipeline(
        numeric_features,
        categorical_features
    )

    X_train = train[features]
    y_train = train["nb_vald"]

    X_val = val[features]
    y_val = val["nb_vald"]

    X_test = test[features]
    y_test = test["nb_vald"]

    # --------------------------------------------------------
    # ENTRAINEMENT
    # --------------------------------------------------------

    print()
    print("Entraînement M1...")

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    evaluate_model(
        model,
        X_val,
        y_val,
        "M1 - VALIDATION"
    )

    evaluate_model(
        model,
        X_test,
        y_test,
        "M1 - TEST"
    )

    # --------------------------------------------------------
    # ARTEFACT
    # --------------------------------------------------------

    artifact = {

        "modele":
            model,

        "features":
            features,

        "features_categorielles":
            categorical_features,

        "features_numeriques":
            numeric_features,

        "cible":
            "nb_vald",

        "perimetre":
            "Ferre",

        "nom_modele":
            "M1_hgb_historique_causal",
    }

    model_path = (
        MODELS_DIR
        / MODEL_FILES["M1"]
    )

    joblib.dump(
        artifact,
        model_path
    )

    print()
    print(
        "M1 sauvegardé :",
        model_path
    )

    print(
        "Clés :",
        list(artifact.keys())
    )

    del df, train, val, test
    gc.collect()


# ============================================================
# M2 — SURFACE
# ============================================================

def train_m2():

    print()
    print("=" * 60)
    print("M2 — SURFACE — NB_VALD")
    print("=" * 60)

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

    print(
        "Shape M2 :",
        df.shape
    )

    # --------------------------------------------------------
    # HISTORIQUE CAUSAL
    # --------------------------------------------------------

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

    df[historical_cols] = (
        df[historical_cols]
        .fillna(0)
    )

    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    train, val, test = temporal_split(
        df
    )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

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

    categorical_features = [

        "code_ligne",

        "id_groupoflines",

        "cat_jour",

        "code_meteo",
    ]

    numeric_features = [

        col

        for col in features

        if col not in categorical_features
    ]

    # --------------------------------------------------------
    # PIPELINE
    # --------------------------------------------------------

    model = create_model_pipeline(
        numeric_features,
        categorical_features
    )

    X_train = train[features]
    y_train = train["nb_vald"]

    X_val = val[features]
    y_val = val["nb_vald"]

    X_test = test[features]
    y_test = test["nb_vald"]

    # --------------------------------------------------------
    # ENTRAINEMENT
    # --------------------------------------------------------

    print()
    print("Entraînement M2...")

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    evaluate_model(
        model,
        X_val,
        y_val,
        "M2 - VALIDATION"
    )

    evaluate_model(
        model,
        X_test,
        y_test,
        "M2 - TEST"
    )

    # --------------------------------------------------------
    # ARTEFACT
    # --------------------------------------------------------

    artifact = {

        "modele":
            model,

        "features":
            features,

        "features_categorielles":
            categorical_features,

        "features_numeriques":
            numeric_features,

        "cible":
            "nb_vald",

        "perimetre":
            "Surface",

        "nom_modele":
            "M2_hgb_historique_causal",
    }

    model_path = (
        MODELS_DIR
        / MODEL_FILES["M2"]
    )

    joblib.dump(
        artifact,
        model_path
    )

    print()
    print(
        "M2 sauvegardé :",
        model_path
    )

    print(
        "Clés :",
        list(artifact.keys())
    )

    del df, train, val, test
    gc.collect()


# ============================================================
# VALIDATION DES ARTEFACTS M1 / M2
# ============================================================

def validate_m1_m2_artifacts():

    print()
    print("=" * 60)
    print("VALIDATION DES ARTEFACTS M1 / M2")
    print("=" * 60)

    for name in ["M1", "M2"]:

        filename = MODEL_FILES[name]

        path = (
            MODELS_DIR
            / filename
        )

        print()
        print(
            "-" * 60
        )

        print(
            name
        )

        print(
            "Fichier :",
            path
        )

        print(
            "Existe :",
            path.exists()
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Artefact absent : {path}"
            )

        artifact = joblib.load(
            path
        )

        print(
            "Clés :",
            list(artifact.keys())
        )

        print(
            "Cible :",
            artifact.get(
                "cible"
            )
        )

        print(
            "Périmètre :",
            artifact.get(
                "perimetre"
            )
        )

        print(
            "Nom modèle :",
            artifact.get(
                "nom_modele"
            )
        )

        print(
            "Nombre features :",
            len(
                artifact[
                    "features"
                ]
            )
        )

        print(
            "Features catégorielles :",
            artifact[
                "features_categorielles"
            ]
        )

        print(
            "Features numériques :",
            artifact[
                "features_numeriques"
            ]
        )

        if len(
            artifact["features"]
        ) != 19:

            raise ValueError(
                f"{name} doit contenir "
                f"19 features, mais en contient "
                f"{len(artifact['features'])}"
            )

    print()
    print("=" * 60)
    print("VALIDATION DES ARTEFACTS M1 / M2 OK")
    print("=" * 60)


# ============================================================
# ============================================================
#                         M3 / M4
# ============================================================
# ============================================================


# ============================================================
# EXTRACTION DE L'HEURE
# ============================================================

def extract_hour(value):
    """
    Convertit une valeur de type :

        08H-09H
        17H-18H

    en heure entière :

        8
        17

    Valeur invalide -> NaN
    """

    if pd.isna(value):

        return np.nan

    value = (
        str(value)
        .strip()
        .upper()
    )

    if value in {
        "ND",
        "",
        "NAN",
        "NONE",
        "NULL",
    }:

        return np.nan

    value = (
        value
        .replace(
            " ",
            ""
        )
    )

    if "H-" in value:

        try:

            heure = (
                value
                .split(
                    "H-"
                )[0]
            )

            heure = int(
                heure
            )

            if 0 <= heure <= 23:

                return heure

        except (
            ValueError,
            TypeError,
        ):

            return np.nan

    return np.nan


# ============================================================
# PREPARATION COMMUNE DES DONNEES
# ============================================================

def prepare_profile_data(df):

    df = df.copy()

    # ========================================================
    # DATE DEBUT
    # ========================================================

    df["date_debut"] = pd.to_datetime(
        df["date_debut"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Suppression des dates invalides
    # --------------------------------------------------------

    invalid_dates = (
        df["date_debut"]
        .isna()
        .sum()
    )

    print(
        "Dates invalides supprimées :",
        invalid_dates
    )

    df = df[
        df["date_debut"].notna()
    ].copy()

    # ========================================================
    # ANNEE
    # ========================================================

    df["annee"] = (
        df["date_debut"]
        .dt
        .year
        .astype("int16")
    )

    # ========================================================
    # TRIMESTRE NUMERIQUE
    # ========================================================

    df["trimestre_num"] = (
        df["date_debut"]
        .dt
        .quarter
        .astype("int8")
    )

    # ========================================================
    # CODE TRIMESTRE
    #
    # YYYY_Tn
    # ========================================================

    df["code_trimestre"] = (
        df["annee"]
        .astype(str)

        + "_T"

        + df["trimestre_num"]
        .astype(str)
    )

    # ========================================================
    # HEURE
    # ========================================================

    df["heure_debut"] = (
        df["trnc_horr_60"]
        .apply(
            extract_hour
        )
    )

    invalid_hours = (
        df["heure_debut"]
        .isna()
        .sum()
    )

    print(
        "Heures invalides supprimées :",
        invalid_hours
    )

    df = df[
        df["heure_debut"].notna()
    ].copy()

    df["heure_debut"] = (
        df["heure_debut"]
        .astype("int8")
    )

    # --------------------------------------------------------
    # Controle heure
    # --------------------------------------------------------

    invalid_hours_range = (
        ~df["heure_debut"]
        .between(
            0,
            23
        )
    ).sum()

    print(
        "Heures hors [0,23] supprimées :",
        invalid_hours_range
    )

    df = df[
        df["heure_debut"]
        .between(
            0,
            23
        )
    ].copy()

    # ========================================================
    # POURCENTAGE
    # ========================================================

    df["pourc_validations"] = (

        pd.to_numeric(
            df["pourc_validations"],
            errors="coerce"
        )

        .fillna(0)

        .clip(
            lower=0
        )

        .astype("float32")
    )

    # ========================================================
    # CATEGORIEL
    # ========================================================

    if "cat_jour" in df.columns:

        df["cat_jour"] = (
            df["cat_jour"]
            .astype("category")
        )

    return df


# ============================================================
# CREATION BASELINE HISTORIQUE
# ============================================================

def create_historical_baseline(
    df,
    profile_keys
):
    """
    Construit une baseline historique sur TOUT l'historique.

    Exemple M3 :

        code_arret
        id_zdc
        cat_jour
        heure_debut

    Exemple M4 :

        code_ligne
        id_groupofligne
        cat_jour
        heure_debut

    Chaque profil est ensuite densifié sur 24 heures et
    normalisé à 100 %.
    """

    print()
    print(
        "Création de la baseline historique..."
    )

    # ========================================================
    # CLES COMPLETES
    # ========================================================

    keys = (
        profile_keys
        + [
            "heure_debut"
        ]
    )

    print()
    print(
        "Clés utilisées :"
    )

    print(
        keys
    )

    # ========================================================
    # AGREGER L'HISTORIQUE
    # ========================================================

    baseline = (

        df

        .groupby(
            keys,
            observed=True,
            as_index=False
        )[

            "pourc_validations"

        ]

        .mean()
    )

    print()
    print(
        "Shape baseline avant densification :",
        baseline.shape
    )

    # ========================================================
    # IDENTIFICATION DES PROFILS
    # ========================================================

    profiles = (
        baseline[
            profile_keys
        ]

        .drop_duplicates()

        .copy()
    )

    print(
        "Nombre de profils avant densification :",
        len(profiles)
    )

    # ========================================================
    # GRILLE DES 24 HEURES
    # ========================================================

    hours = pd.DataFrame(
        {
            "heure_debut":
                np.arange(
                    24,
                    dtype=np.int8
                )
        }
    )

    # --------------------------------------------------------
    # Produit cartésien profil x 24 heures
    # --------------------------------------------------------

    profiles["_tmp_key"] = 1
    hours["_tmp_key"] = 1

    grid = (
        profiles

        .merge(
            hours,
            on="_tmp_key",
            how="outer"
        )

        .drop(
            columns="_tmp_key"
        )
    )

    # ========================================================
    # MERGE AVEC LES VALEURS HISTORIQUES
    # ========================================================

    baseline = (
        grid

        .merge(
            baseline,
            on=keys,
            how="left"
        )
    )

    # ========================================================
    # VALEURS MANQUANTES
    # ========================================================

    baseline[
        "pourc_validations"
    ] = (

        baseline[
            "pourc_validations"
        ]

        .fillna(0)

        .astype("float32")
    )

    # ========================================================
    # NORMALISATION A 100 %
    # ========================================================

    totals = (

        baseline

        .groupby(
            profile_keys,
            observed=True
        )[

            "pourc_validations"

        ]

        .transform(
            "sum"
        )
    )

    # --------------------------------------------------------
    # Profils avec historique disponible
    # --------------------------------------------------------

    mask_positive = (
        totals > 0
    )

    baseline.loc[
        mask_positive,
        "pourc_validations"
    ] = (

        baseline.loc[
            mask_positive,
            "pourc_validations"
        ]

        / totals.loc[
            mask_positive
        ]

        * 100
    )

    # --------------------------------------------------------
    # Profils sans historique exploitable
    # --------------------------------------------------------

    baseline.loc[
        ~mask_positive,
        "pourc_validations"
    ] = (
        100.0 / 24.0
    )

    # ========================================================
    # TYPE FINAL
    # ========================================================

    baseline[
        "pourc_validations"
    ] = (
        baseline[
            "pourc_validations"
        ]
        .astype("float32")
    )

    # ========================================================
    # TRI
    # ========================================================

    baseline = (
        baseline

        .sort_values(
            keys
        )

        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # CONTROLE
    # ========================================================

    controle = (

        baseline

        .groupby(
            profile_keys,
            observed=True
        )[

            "pourc_validations"

        ]

        .sum()
    )

    print()
    print(
        "=================================================="
    )

    print(
        "CONTROLE BASELINE"
    )

    print(
        "=================================================="
    )

    print(
        "Nombre profils :",
        len(controle)
    )

    print(
        "Nombre lignes :",
        len(baseline)
    )

    print(
        "Minimum total % :",
        round(
            controle.min(),
            6
        )
    )

    print(
        "Maximum total % :",
        round(
            controle.max(),
            6
        )
    )

    print(
        "Moyenne total % :",
        round(
            controle.mean(),
            6
        )
    )

    print(
        "Nombre profils dont somme != 100 :",
        int(
            (
                np.abs(
                    controle - 100
                ) > 0.001
            ).sum()
        )
    )

    print(
        "Shape finale baseline :",
        baseline.shape
    )

    print(
        "=================================================="
    )

    return baseline


# ============================================================
# CREATION METADONNEES
# ============================================================

def create_metadata(
    df,
    profile_keys,
    dataset_name,
    strategy_name
):
    """
    Métadonnées stockées avec l'artefact.
    """

    periods = (
        df["code_trimestre"]
        .dropna()
        .unique()
        .tolist()
    )

    def period_key(value):

        year, quarter = (
            value.split(
                "_T"
            )
        )

        return (
            int(year),
            int(quarter)
        )

    periods = sorted(
        periods,
        key=period_key
    )

    metadata = {

        "dataset":
            dataset_name,

        "strategie":
            strategy_name,

        "type_modele":
            "baseline_historique",

        "code_trimestre_format":
            "YYYY_Tn",

        "code_trimestre_exemple":
            [
                "2025_T1",
                "2025_T2",
                "2025_T3",
                "2025_T4",
            ],

        "code_trimestre_utilise_comme_cle":
            False,

        "profil_keys":
            list(profile_keys),

        "heure_range":
            [
                0,
                23
            ],

        "normalisation":
            "24 heures = 100 %",

        "date_min":
            str(
                df["date_debut"].min()
            ),

        "date_max":
            str(
                df["date_debut"].max()
            ),

        "periode_min":
            periods[0]
            if periods
            else None,

        "periode_max":
            periods[-1]
            if periods
            else None,

        "nombre_periodes":
            len(periods),

        "periodes":
            periods,
    }

    return metadata


# ============================================================
# M3 — FERRE
# ============================================================

def train_m3():

    print()
    print(
        "=" * 80
    )

    print(
        "M3 — FERRE — BASELINE HISTORIQUE"
    )

    print(
        "=" * 80
    )

    # ========================================================
    # REQUETE
    # ========================================================

    query = """

    SELECT

        date_debut,

        date_fin,

        code_stif_trns,

        code_stif_res,

        code_stif_arret,

        libelle_arret,

        id_zdc,

        cat_jour,

        trnc_horr_60,

        pourc_validations

    FROM transport.fa_profils_horaires_ferre

    """

    df = pd.read_sql(
        query,
        engine
    )

    print(
        "Shape brute M3 :",
        df.shape
    )

    # ========================================================
    # CODE ARRET
    # ========================================================

    df["code_arret"] = (

        df["code_stif_trns"]
        .astype(str)

        + "_"

        + df["code_stif_res"]
        .astype(str)

        + "_"

        + df["code_stif_arret"]
        .astype(str)
    )

    # ========================================================
    # PREPARATION
    # ========================================================

    df = prepare_profile_data(
        df
    )

    print(
        "Shape préparée M3 :",
        df.shape
    )

    # ========================================================
    # PROFIL
    # ========================================================

    profile_keys = [

        "code_arret",

        "id_zdc",

        "cat_jour",
    ]

    # ========================================================
    # BASELINE
    # ========================================================

    baseline = create_historical_baseline(

        df,

        profile_keys
    )

    # ========================================================
    # METADONNEES
    # ========================================================

    metadata = create_metadata(

        df,

        profile_keys,

        "M3",

        "Baseline historique"
    )

    # ========================================================
    # ARTEFACT
    # ========================================================

    artifact = {

        "baseline":
            baseline,

        "cles":
            profile_keys
            + [
                "heure_debut"
            ],

        "profil_keys":
            profile_keys,

        "colonne_prediction":
            "pourc_validations",

        "cible":
            "pourc_validations",

        "perimetre":
            "Ferre",

        "nom_modele":
            "M3_baseline_historique",

        "strategie":
            "baseline_historique",

        "metadata":
            metadata,
    }

    # ========================================================
    # SAUVEGARDE
    # ========================================================

    model_path = (
        MODELS_DIR
        / MODEL_FILES["M3"]
    )

    joblib.dump(
        artifact,
        model_path
    )

    print()
    print(
        "M3 sauvegardé :"
    )

    print(
        model_path
    )

    print()
    print(
        "Clés artefact :"
    )

    print(
        list(
            artifact.keys()
        )
    )

    print()
    print(
        "Clés profil :"
    )

    print(
        artifact["cles"]
    )

    print()
    print(
        "Historique utilisé :",
        metadata["periode_min"],
        "->",
        metadata["periode_max"]
    )

    print(
        "Date min :",
        metadata["date_min"]
    )

    print(
        "Date max :",
        metadata["date_max"]
    )

    print(
        "Nombre périodes :",
        metadata["nombre_periodes"]
    )

    print(
        "Shape baseline :",
        baseline.shape
    )

    # ========================================================
    # LIBERATION
    # ========================================================

    del df
    del baseline
    del artifact
    del metadata

    gc.collect()

    return model_path


# ============================================================
# M4 — SURFACE
# ============================================================

def train_m4():

    print()
    print(
        "=" * 80
    )

    print(
        "M4 — SURFACE — BASELINE HISTORIQUE"
    )

    print(
        "=" * 80
    )

    # ========================================================
    # REQUETE
    # ========================================================

    query = """

    SELECT

        date_debut,

        date_fin,

        code_stif_trns,

        code_stif_res,

        code_stif_ligne,

        libelle_ligne,

        id_groupofligne,

        cat_jour,

        trnc_horr_60,

        pourc_validations

    FROM transport.fa_profils_horaires_surface

    """

    df = pd.read_sql(
        query,
        engine
    )

    print(
        "Shape brute M4 :",
        df.shape
    )

    # ========================================================
    # CODE LIGNE
    # ========================================================

    df["code_ligne"] = (

        df["code_stif_trns"]
        .astype(str)

        + "_"

        + df["code_stif_res"]
        .astype(str)

        + "_"

        + df["code_stif_ligne"]
        .astype(str)
    )

    # ========================================================
    # PREPARATION
    # ========================================================

    df = prepare_profile_data(
        df
    )

    print(
        "Shape préparée M4 :",
        df.shape
    )

    # ========================================================
    # PROFIL
    # ========================================================

    profile_keys = [

        "code_ligne",

        "id_groupofligne",

        "cat_jour",
    ]

    # ========================================================
    # BASELINE
    # ========================================================

    baseline = create_historical_baseline(

        df,

        profile_keys
    )

    # ========================================================
    # METADONNEES
    # ========================================================

    metadata = create_metadata(

        df,

        profile_keys,

        "M4",

        "Baseline historique"
    )

    # ========================================================
    # ARTEFACT
    # ========================================================

    artifact = {

        "baseline":
            baseline,

        "cles":
            profile_keys
            + [
                "heure_debut"
            ],

        "profil_keys":
            profile_keys,

        "colonne_prediction":
            "pourc_validations",

        "cible":
            "pourc_validations",

        "perimetre":
            "Surface",

        "nom_modele":
            "M4_baseline_historique",

        "strategie":
            "baseline_historique",

        "metadata":
            metadata,
    }

    # ========================================================
    # SAUVEGARDE
    # ========================================================

    model_path = (
        MODELS_DIR
        / MODEL_FILES["M4"]
    )

    joblib.dump(
        artifact,
        model_path
    )

    print()
    print(
        "M4 sauvegardé :"
    )

    print(
        model_path
    )

    print()
    print(
        "Clés artefact :"
    )

    print(
        list(
            artifact.keys()
        )
    )

    print()
    print(
        "Clés profil :"
    )

    print(
        artifact["cles"]
    )

    print()
    print(
        "Historique utilisé :",
        metadata["periode_min"],
        "->",
        metadata["periode_max"]
    )

    print(
        "Date min :",
        metadata["date_min"]
    )

    print(
        "Date max :",
        metadata["date_max"]
    )

    print(
        "Nombre périodes :",
        metadata["nombre_periodes"]
    )

    print(
        "Shape baseline :",
        baseline.shape
    )

    # ========================================================
    # LIBERATION
    # ========================================================

    del df
    del baseline
    del artifact
    del metadata

    gc.collect()

    return model_path


# ============================================================
# VALIDATION DES ARTEFACTS M3 / M4
# ============================================================

def validate_m3_m4_artifacts():

    print()
    print(
        "=" * 80
    )

    print(
        "VALIDATION DES ARTEFACTS M3 / M4"
    )

    print(
        "=" * 80
    )

    for name in ["M3", "M4"]:

        filename = MODEL_FILES[name]

        path = (
            MODELS_DIR
            / filename
        )

        print()
        print(
            "-" * 70
        )

        print(
            name
        )

        print(
            "Fichier :",
            path
        )

        print(
            "Existe :",
            path.exists()
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Artefact absent : {path}"
            )

        artifact = joblib.load(
            path
        )

        required_keys = {

            "baseline",

            "cles",

            "profil_keys",

            "colonne_prediction",

            "cible",

            "perimetre",

            "nom_modele",

            "strategie",

            "metadata",
        }

        missing_keys = (
            required_keys
            - set(
                artifact.keys()
            )
        )

        if missing_keys:

            raise ValueError(
                f"{name} : clés manquantes : "
                f"{missing_keys}"
            )

        baseline = (
            artifact[
                "baseline"
            ]
        )

        keys = (
            artifact[
                "cles"
            ]
        )

        profile_keys = (
            artifact[
                "profil_keys"
            ]
        )

        # ----------------------------------------------------
        # Controle colonnes
        # ----------------------------------------------------

        expected_columns = (
            keys
            + [
                "pourc_validations"
            ]
        )

        missing_columns = [
            col
            for col in expected_columns
            if col not in baseline.columns
        ]

        if missing_columns:

            raise ValueError(
                f"{name} : colonnes manquantes "
                f"dans baseline : {missing_columns}"
            )

        # ----------------------------------------------------
        # Nombre de profils
        # ----------------------------------------------------

        n_profiles = (
            baseline[
                profile_keys
            ]
            .drop_duplicates()
            .shape[0]
        )

        # ----------------------------------------------------
        # Controle 24 heures
        # ----------------------------------------------------

        hours_per_profile = (
            baseline
            .groupby(
                profile_keys,
                observed=True
            )[
                "heure_debut"
            ]
            .nunique()
        )

        bad_profiles = int(
            (
                hours_per_profile
                != 24
            )
            .sum()
        )

        # ----------------------------------------------------
        # Controle somme 100
        # ----------------------------------------------------

        totals = (
            baseline
            .groupby(
                profile_keys,
                observed=True
            )[
                "pourc_validations"
            ]
            .sum()
        )

        invalid_totals = int(
            (
                np.abs(
                    totals
                    - 100
                )
                > 0.001
            )
            .sum()
        )

        print(
            "Nom modèle :",
            artifact[
                "nom_modele"
            ]
        )

        print(
            "Périmètre :",
            artifact[
                "perimetre"
            ]
        )

        print(
            "Stratégie :",
            artifact[
                "strategie"
            ]
        )

        print(
            "Clés :",
            keys
        )

        print(
            "Nombre profils :",
            n_profiles
        )

        print(
            "Nombre lignes baseline :",
            len(baseline)
        )

        print(
            "Profils avec != 24 heures :",
            bad_profiles
        )

        print(
            "Profils avec somme != 100 % :",
            invalid_totals
        )

        print(
            "Période min :",
            artifact[
                "metadata"
            ].get(
                "periode_min"
            )
        )

        print(
            "Période max :",
            artifact[
                "metadata"
            ].get(
                "periode_max"
            )
        )

        print(
            "Date min :",
            artifact[
                "metadata"
            ].get(
                "date_min"
            )
        )

        print(
            "Date max :",
            artifact[
                "metadata"
            ].get(
                "date_max"
            )
        )

        print(
            "Code trimestre comme clé :",
            artifact[
                "metadata"
            ].get(
                "code_trimestre_utilise_comme_cle"
            )
        )

        if bad_profiles != 0:

            raise ValueError(
                f"{name} : certains profils "
                f"ne possèdent pas 24 heures."
            )

        if invalid_totals != 0:

            raise ValueError(
                f"{name} : certains profils "
                f"ne totalisent pas 100 %."
            )

        del artifact
        del baseline
        del totals
        del hours_per_profile

        gc.collect()

    print()
    print(
        "=" * 80
    )

    print(
        "VALIDATION DES 2 ARTEFACTS M3 / M4 OK"
    )

    print(
        "=" * 80
    )


# ============================================================
# VALIDATION GLOBALE DES 4 ARTEFACTS
# ============================================================

def validate_all_artifacts():

    print()
    print(
        "=" * 100
    )

    print(
        "VALIDATION GLOBALE DES 4 ARTEFACTS"
    )

    print(
        "=" * 100
    )

    # --------------------------------------------------------
    # M1 / M2
    # --------------------------------------------------------

    validate_m1_m2_artifacts()

    # --------------------------------------------------------
    # M3 / M4
    # --------------------------------------------------------

    validate_m3_m4_artifacts()

    print()
    print(
        "=" * 100
    )

    print(
        "VALIDATION GLOBALE DES 4 ARTEFACTS OK"
    )

    print(
        "=" * 100
    )


# ============================================================
# EXECUTION COMPLETE
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "=" * 100
    )

    print(
        "ENTRAINEMENT FINAL DES 4 MODELES"
    )

    print(
        "M1 / M2 : HISTORIQUE CAUSAL"
    )

    print(
        "M3 / M4 : BASELINE HISTORIQUE"
    )

    print(
        "=" * 100
    )

    print()
    print(
        "Répertoire projet :"
    )

    print(
        PROJECT_DIR
    )

    print()
    print(
        "Répertoire modèles :"
    )

    print(
        MODELS_DIR
    )

    print()
    print(
        "Artefacts produits :"
    )

    print(
        " -",
        MODEL_FILES["M1"]
    )

    print(
        " -",
        MODEL_FILES["M2"]
    )

    print(
        " -",
        MODEL_FILES["M3"]
    )

    print(
        " -",
        MODEL_FILES["M4"]
    )

    # ========================================================
    # M1
    # ========================================================

    train_m1()

    print()
    print(
        "M1 terminé."
    )

    gc.collect()

    # ========================================================
    # M2
    # ========================================================

    train_m2()

    print()
    print(
        "M2 terminé."
    )

    gc.collect()

    # ========================================================
    # M3
    # ========================================================

    train_m3()

    print()
    print(
        "M3 terminé."
    )

    gc.collect()

    # ========================================================
    # M4
    # ========================================================

    train_m4()

    print()
    print(
        "M4 terminé."
    )

    gc.collect()

    # ========================================================
    # VALIDATION FINALE
    # ========================================================

    validate_all_artifacts()

    # ========================================================
    # FIN
    # ========================================================

    print()
    print(
        "=" * 100
    )

    print(
        "LES 4 MODELES SONT TERMINES"
    )

    print(
        "M1 ET M2 : MODELES HGB"
    )

    print(
        "M3 ET M4 : BASELINES HISTORIQUES"
    )

    print(
        "=" * 100
    )

    print()
    print(
        "Artefacts produits :"
    )

    print(
        MODELS_DIR
        / MODEL_FILES["M1"]
    )

    print(
        MODELS_DIR
        / MODEL_FILES["M2"]
    )

    print(
        MODELS_DIR
        / MODEL_FILES["M3"]
    )

    print(
        MODELS_DIR
        / MODEL_FILES["M4"]
    )

    print()
    print(
        "=" * 100
    )
