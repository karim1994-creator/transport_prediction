import numpy as np
import pandas as pd

from sqlalchemy import create_engine, text

from app.config import DATABASE_URL

from app.services.ml_features import (
    FEATURES_M1,
    FEATURES_M2,
    prepare_m1_features,
    prepare_m2_features,
)


class PredictionEngine:

    def __init__(self, loader):

        self.loader = loader

        self.engine = create_engine(
            DATABASE_URL
        )

    # ========================================================
    # M1 — CHARGER L'HISTORIQUE
    # ========================================================

    def _load_history_m1(
        self,
        code_arret,
        id_zdc,
        date_prediction,
    ):

        query = text("""
            SELECT

                jour,

                code_arret,

                id_zdc,

                MAX(cat_jour) AS cat_jour,

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

            WHERE code_arret = :code_arret

              AND id_zdc = :id_zdc

              AND jour < :date_prediction

            GROUP BY

                jour,
                code_arret,
                id_zdc

            ORDER BY
                jour
        """)

        return pd.read_sql(
            query,
            self.engine,
            params={
                "code_arret":
                    code_arret,

                "id_zdc":
                    id_zdc,

                "date_prediction":
                    pd.Timestamp(
                        date_prediction
                    ),
            },
        )

    # ========================================================
    # M2 — CHARGER L'HISTORIQUE
    # ========================================================

    def _load_history_m2(
        self,
        code_ligne,
        id_groupoflines,
        date_prediction,
    ):

        query = text("""
            SELECT

                jour,

                code_ligne,

                id_groupoflines,

                MAX(cat_jour) AS cat_jour,

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

            WHERE code_ligne = :code_ligne

              AND id_groupoflines =
                  :id_groupoflines

              AND jour < :date_prediction

            GROUP BY

                jour,
                code_ligne,
                id_groupoflines

            ORDER BY
                jour
        """)

        return pd.read_sql(
            query,
            self.engine,
            params={
                "code_ligne":
                    code_ligne,

                "id_groupoflines":
                    id_groupoflines,

                "date_prediction":
                    pd.Timestamp(
                        date_prediction
                    ),
            },
        )

    # ========================================================
    # PREPARATION M1
    # ========================================================

    def _prepare_m1_prediction(
        self,
        row,
    ):

        target = pd.DataFrame(
            [row]
        )

        target["jour"] = pd.to_datetime(
            target["jour"]
        )

        target["nb_vald"] = np.nan

        history = self._load_history_m1(

            code_arret=row[
                "code_arret"
            ],

            id_zdc=row[
                "id_zdc"
            ],

            date_prediction=row[
                "jour"
            ],
        )

        if history.empty:

            raise ValueError(
                "M1 : aucun historique disponible "
                "pour calculer les variables V4."
            )

        combined = pd.concat(
            [
                history,
                target,
            ],
            ignore_index=True,
            sort=False,
        )

        combined = prepare_m1_features(
            combined
        )

        prediction_row = (

            combined[
                combined["jour"]
                == target.iloc[0]["jour"]
            ]

            .tail(1)
        )

        if prediction_row.empty:

            raise ValueError(
                "M1 : impossible de construire "
                "les variables de prédiction."
            )

        return prediction_row

    # ========================================================
    # PREPARATION M2
    # ========================================================

    def _prepare_m2_prediction(
        self,
        row,
    ):

        target = pd.DataFrame(
            [row]
        )

        target["jour"] = pd.to_datetime(
            target["jour"]
        )

        target["nb_vald"] = np.nan

        history = self._load_history_m2(

            code_ligne=row[
                "code_ligne"
            ],

            id_groupoflines=row[
                "id_groupoflines"
            ],

            date_prediction=row[
                "jour"
            ],
        )

        if history.empty:

            raise ValueError(
                "M2 : aucun historique disponible "
                "pour calculer les variables V4."
            )

        combined = pd.concat(
            [
                history,
                target,
            ],
            ignore_index=True,
            sort=False,
        )

        combined = prepare_m2_features(
            combined
        )

        prediction_row = (

            combined[
                combined["jour"]
                == target.iloc[0]["jour"]
            ]

            .tail(1)
        )

        if prediction_row.empty:

            raise ValueError(
                "M2 : impossible de construire "
                "les variables de prédiction."
            )

        return prediction_row


    @staticmethod
    def _json_safe(value):
        if isinstance(value, pd.Timestamp):
            return value.isoformat()

        if isinstance(value, np.generic):
            value = value.item()

        if isinstance(value, float):
            if not np.isfinite(value):
                return None

        return value

    # ========================================================
    # PREDICTION M1 / M2
    # ========================================================

    def predict_ml(
        self,
        model_name,
        rows,
    ):

        artifact = self.loader.get(
            model_name
        )

        if (
            "modele" not in artifact
            or
            "features" not in artifact
        ):

            raise ValueError(
                f"{model_name} n'est pas un "
                "artefact ML M1/M2 compatible."
            )

        if model_name not in {
            "M1",
            "M2",
        }:

            raise ValueError(
                "predict_ml accepte uniquement M1 ou M2."
            )

        if not rows:

            raise ValueError(
                "Aucune ligne à prédire."
            )

        prediction_rows = []
        feature_rows = []

        # ----------------------------------------------------
        # Construction des features
        # ----------------------------------------------------

        for row in rows:

            if model_name == "M1":

                prepared = (
                    self._prepare_m1_prediction(
                        row
                    )
                )

                expected_features = FEATURES_M1

            else:

                prepared = (
                    self._prepare_m2_prediction(
                        row
                    )
                )

                expected_features = FEATURES_M2

            features = list(
                artifact["features"]
            )

            if set(features) != set(
                expected_features
            ):

                raise ValueError(
                    f"{model_name} : les features "
                    "de l'artefact ne correspondent "
                    "pas au feature engineering V4."
                )

            missing = [
                col
                for col in features
                if col not in prepared.columns
            ]

            if missing:

                raise ValueError(
                    f"{model_name} : variables "
                    f"manquantes : {missing}"
                )

            # Données complètes utilisées pour l'historisation
            full_feature_row = dict(row)

            full_feature_row.update(
                prepared.iloc[0].to_dict()
            )

            safe_feature_row = {
                key: self._json_safe(value)
                for key, value in full_feature_row.items()
            }

            feature_rows.append(
                safe_feature_row
            )

            # Uniquement les 24 variables envoyées au modèle
            prediction_rows.append(
                prepared[
                    features
                ].iloc[0]
            )

        # ----------------------------------------------------
        # DataFrame final
        # ----------------------------------------------------

        X = pd.DataFrame(
            prediction_rows
        )

        predictions = artifact[
            "modele"
        ].predict(
            X
        )

        predictions = np.clip(
            np.asarray(
                predictions,
                dtype=float
            ),
            0,
            None,
        )

        return {
            "model": model_name,
            "target": artifact.get("cible"),
            "version": artifact.get(
                "version_optimisation",
                "production"
            ),
            "n_rows": len(X),
            "predictions": predictions.tolist(),
            "features": features,
            "feature_rows": feature_rows,
        }
    # ========================================================
    # M3 / M4
    # ========================================================

    def predict_full_profile(
        self,
        model_name,
        profile,
    ):

        artifact = self.loader.get(
            model_name
        )

        if (
            "baseline" not in artifact
            or
            "profil_keys" not in artifact
        ):

            raise ValueError(
                f"{model_name} n'est pas un "
                "artefact baseline M3/M4 compatible."
            )

        keys = list(
            artifact[
                "profil_keys"
            ]
        )

        missing = [
            k
            for k in keys
            if k not in profile
        ]

        if missing:

            raise ValueError(
                "Variables profil manquantes : "
                f"{missing}"
            )

        baseline = artifact[
            "baseline"
        ]

        rows = [

            {
                **{
                    k: profile[k]
                    for k in keys
                },

                "heure_debut":
                    hour,
            }

            for hour in range(24)
        ]

        query = pd.DataFrame(
            rows
        )

        merge_keys = (
            keys
            + [
                "heure_debut"
            ]
        )

        out = query.merge(

            baseline[
                merge_keys
                + [
                    "pourc_validations"
                ]
            ],

            on=merge_keys,

            how="left",
        )

        out["prediction"] = (

            out[
                "pourc_validations"
            ]

            .fillna(
                100.0 / 24.0
            )

            .astype(float)
        )

        out = out.drop(
            columns=[
                "pourc_validations"
            ]
        )

        total = (
            out[
                "prediction"
            ].sum()
        )

        if total > 0:

            out["prediction"] = (

                out["prediction"]
                / total
                * 100.0
            )

        else:

            out["prediction"] = (
                100.0 / 24.0
            )

        return {

            "model":
                model_name,

            "target":
                artifact.get(
                    "cible"
                ),

            "strategy":
                artifact.get(
                    "strategie"
                ),

            "profile_keys":
                keys,

            "profile": [

                {
                    "heure_debut":
                        int(
                            r.heure_debut
                        ),

                    "prediction_percent":
                        float(
                            r.prediction
                        ),
                }

                for r in out.itertuples()
            ],

            "total_percent":
                float(
                    out[
                        "prediction"
                    ].sum()
                ),
        }