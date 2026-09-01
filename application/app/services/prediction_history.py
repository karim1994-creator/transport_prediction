from datetime import datetime
import json

from sqlalchemy import create_engine, text


class PredictionHistory:

    def __init__(self, database_url):

        self.engine = create_engine(
            database_url
        )

    # ========================================================
    # ENREGISTREMENT
    # ========================================================

    def save_prediction(
        self,
        model_name,
        model_version,
        perimeter,
        request_profile,
        prediction,
        prediction_total,
        api_status="success",
        error_message=None,
    ):

        query = text("""
            INSERT INTO prediction.ml_prediction_history (

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

        with self.engine.begin() as connection:

            connection.execute(
                query,
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
                        api_status,

                    "error_message":
                        error_message,
                }
            )