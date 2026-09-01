from datetime import datetime
import json

import requests
from sqlalchemy import create_engine, text

from app.config import DATABASE_URL


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8000"

engine = create_engine(
    DATABASE_URL
)


# ============================================================
# INSERTION HISTORIQUE
# ============================================================

INSERT_SQL = text("""
    INSERT INTO public.ml_prediction_history (

        prediction_timestamp,
        model_name,
        model_version,
        perimeter,
        request_profile,
        prediction,
        prediction_total,
        api_status,
        error_message

    )

    VALUES (

        :prediction_timestamp,
        :model_name,
        :model_version,
        :perimeter,
        CAST(:request_profile AS JSONB),
        CAST(:prediction AS JSONB),
        :prediction_total,
        :api_status,
        :error_message
    )
""")


def save_prediction(
    model_name,
    model_version,
    perimeter,
    request_profile,
    prediction,
    prediction_total,
    status="success",
    error_message=None,
):

    with engine.begin() as connection:

        connection.execute(
            INSERT_SQL,
            {
                "prediction_timestamp":
                    datetime.now(),

                "model_name":
                    model_name,

                "model_version":
                    model_version,

                "perimeter":
                    perimeter,

                "request_profile":
                    json.dumps(
                        request_profile
                    ),

                "prediction":
                    json.dumps(
                        prediction
                    ),

                "prediction_total":
                    prediction_total,

                "api_status":
                    status,

                "error_message":
                    error_message,
            }
        )


# ============================================================
# PREDICTION M3
# ============================================================

def collect_m3():

    profile = {
        "code_arret":
            "810_801_594 ",

        "id_zdc":
            "71590",

        "cat_jour":
            "JOHV",
    }

    try:

        response = requests.post(
            f"{API_URL}/predict/m3",
            json={
                "profile":
                    profile
            },
            timeout=30,
        )

        response.raise_for_status()

        result = response.json()

        save_prediction(
            model_name="M3",
            model_version="production",
            perimeter="Ferre",
            request_profile=profile,
            prediction=result["profile"],
            prediction_total=result["total_percent"],
        )

        print(
            "M3 : prévision historisée"
        )

    except Exception as exc:

        save_prediction(
            model_name="M3",
            model_version="production",
            perimeter="Ferre",
            request_profile=profile,
            prediction=[],
            prediction_total=None,
            status="error",
            error_message=str(exc),
        )

        print(
            "M3 : erreur :",
            exc
        )


# ============================================================
# PREDICTION M4
# ============================================================

def collect_m4():

    profile = {
        "code_ligne":
            "534_534_524",

        "id_groupofligne":
            "A00553",

        "cat_jour":
            "JOHV",
    }

    try:

        response = requests.post(
            f"{API_URL}/predict/m4",
            json={
                "profile":
                    profile
            },
            timeout=30,
        )

        response.raise_for_status()

        result = response.json()

        save_prediction(
            model_name="M4",
            model_version="production",
            perimeter="Surface",
            request_profile=profile,
            prediction=result["profile"],
            prediction_total=result["total_percent"],
        )

        print(
            "M4 : prévision historisée"
        )

    except Exception as exc:

        save_prediction(
            model_name="M4",
            model_version="production",
            perimeter="Surface",
            request_profile=profile,
            prediction=[],
            prediction_total=None,
            status="error",
            error_message=str(exc),
        )

        print(
            "M4 : erreur :",
            exc
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 70
    )

    print(
        "HISTORISATION DES PREDICTIONS"
    )

    print(
        "=" * 70
    )

    collect_m3()
    collect_m4()

    print(
        "=" * 70
    )

    print(
        "COLLECTE TERMINEE"
    )

    print(
        "=" * 70
    )