from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from app.config import DATABASE_URL
from app.services.ml_features import prepare_m1_features, prepare_m2_features


class DayPredictionService:
    """Predictions massives M1/M2 et profils M3/M4 pour une journée."""

    def __init__(self, loader):
        self.loader = loader
        self.engine = create_engine(DATABASE_URL)

    def _get_meteo_day(self, day: pd.Timestamp) -> Dict[str, Any]:
        query = text("""
            SELECT
                date,
                temperature_moyenne,
                pluie_totale,
                probabilite_pluie_moyenne,
                vitesse_vent_moyenne,
                code_meteo
            FROM transport.dm_meteo_journalier
            WHERE date = :day
        """)

        with self.engine.begin() as conn:
            row = conn.execute(
                query,
                {"day": day.date()}
            ).mappings().first()

        return dict(row) if row else {}

    def _get_jour_exploitation(self, day: pd.Timestamp) -> Dict[str, Any]:
        query = text("""
            SELECT
                date,
                nom_jour,
                num_mois,
                nom_mois,
                annee,
                trimestre,
                semestre,
                vacance_scolaire,
                jour_ferie_et_pont,
                cat_jour
            FROM transport.dm_jour_exploitation
            WHERE date = :day
        """)

        with self.engine.begin() as conn:
            row = conn.execute(
                query,
                {"day": day.date()}
            ).mappings().first()

        return dict(row) if row else {}

    def _infer_cat_jour(self, table: str, target_date: pd.Timestamp) -> Any:
        query = text(f"""
            SELECT cat_jour
            FROM {table}
            WHERE jour < :day
            GROUP BY cat_jour
            ORDER BY COUNT(*) DESC
            LIMIT 1
        """)
        with self.engine.begin() as conn:
            return conn.execute(query, {"day": target_date}).scalar()

    def _history_m1(self, day: pd.Timestamp) -> pd.DataFrame:
        return pd.read_sql(text("""
            SELECT jour, code_arret, id_zdc,
                   MAX(cat_jour) AS cat_jour,
                   MAX(temperature_moyenne) AS temperature_moyenne,
                   MAX(pluie_totale) AS pluie_totale,
                   MAX(vitesse_vent_moyenne) AS vitesse_vent_moyenne,
                   MAX(code_meteo) AS code_meteo,
                   SUM(nb_vald) AS nb_vald
            FROM vues_metier.ferre_analyse
            WHERE jour < :day
            GROUP BY jour, code_arret, id_zdc
            ORDER BY code_arret, id_zdc, jour
        """), self.engine, params={"day": day})

    def _history_m2(self, day: pd.Timestamp) -> pd.DataFrame:
        return pd.read_sql(text("""
            SELECT jour, code_ligne, id_groupoflines,
                   MAX(cat_jour) AS cat_jour,
                   MAX(temperature_moyenne) AS temperature_moyenne,
                   MAX(pluie_totale) AS pluie_totale,
                   MAX(vitesse_vent_moyenne) AS vitesse_vent_moyenne,
                   MAX(code_meteo) AS code_meteo,
                   SUM(nb_vald) AS nb_vald
            FROM vues_metier.surface_analyse
            WHERE jour < :day
            GROUP BY jour, code_ligne, id_groupoflines
            ORDER BY code_ligne, id_groupoflines, jour
        """), self.engine, params={"day": day})


    @staticmethod
    def _build_targets(
        history: pd.DataFrame,
        day: pd.Timestamp,
        entity_cols: List[str],
        cat_jour: Any,
        meteo: Dict[str, Any],
    ) -> pd.DataFrame:

        last = (
            history
            .sort_values("jour")
            .groupby(entity_cols, as_index=False)
            .tail(1)
            .copy()
        )

        last["jour"] = day
        last["nb_vald"] = np.nan

        # Catégorie du jour
        last["cat_jour"] = cat_jour

        # Météo du jour
        last["temperature_moyenne"] = meteo["temperature_moyenne"]
        last["pluie_totale"] = meteo["pluie_totale"]
        last["vitesse_vent_moyenne"] = meteo["vitesse_vent_moyenne"]
        last["code_meteo"] = meteo["code_meteo"]

        return last

    def predict_ml_day(
        self,
        model_name: str,
        day: Any,
        meteo: Dict[str, Any] | None = None,
        jour_exploitation: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:

        day = pd.Timestamp(day).normalize()

        # --------------------------------------------------------
        # Récupération automatique depuis les tables DM
        # --------------------------------------------------------

        if meteo is None:
            meteo = self._get_meteo_day(day)

        if jour_exploitation is None:
            jour_exploitation = self._get_jour_exploitation(day)

        # --------------------------------------------------------
        # Vérifications
        # --------------------------------------------------------

        if not meteo:
            raise ValueError(
                f"Données météo non présentes dans la table DM "
                f"pour le {day.date()}."
            )

        if not jour_exploitation:
            raise ValueError(
                f"Informations de la journée non présentes dans la table DM "
                f"pour le {day.date()}."
            )

        cat = jour_exploitation.get("cat_jour")

        if not cat:
            raise ValueError(
                f"cat_jour absent dans dm_jour_exploitation "
                f"pour le {day.date()}."
            )

        # --------------------------------------------------------
        # M1
        # --------------------------------------------------------

        if model_name == "M1":

            history = self._history_m1(day)

            if history.empty:
                raise ValueError(
                    "M1 : aucun historique disponible avant la date demandée."
                )

            target = self._build_targets(
                history,
                day,
                ["code_arret", "id_zdc"],
                cat,
                meteo,
            )

            combined = pd.concat(
                [history, target],
                ignore_index=True
            )

            prepared = prepare_m1_features(combined)

        # --------------------------------------------------------
        # M2
        # --------------------------------------------------------

        elif model_name == "M2":

            history = self._history_m2(day)

            if history.empty:
                raise ValueError(
                    "M2 : aucun historique disponible avant la date demandée."
                )

            target = self._build_targets(
                history,
                day,
                ["code_ligne", "id_groupoflines"],
                cat,
                meteo,
            )

            combined = pd.concat(
                [history, target],
                ignore_index=True
            )

            prepared = prepare_m2_features(combined)

        else:
            raise ValueError(
                "model_name doit être M1 ou M2"
            )

        # --------------------------------------------------------
        # Prédiction
        # --------------------------------------------------------

        targets = prepared[
            prepared["jour"] == day
        ].copy()

        artifact = self.loader.get(model_name)

        features = list(
            artifact["features"]
        )

        missing_features = [
            col
            for col in features
            if col not in targets.columns
        ]

        if missing_features:
            raise ValueError(
                f"{model_name} : variables manquantes : "
                f"{missing_features}"
            )

        X = targets[features]

        predictions = np.clip(
            np.asarray(
                artifact["modele"].predict(X),
                dtype=float
            ),
            0,
            None,
        )

        return {
            "model": model_name,
            "version": artifact.get(
                "version_optimisation",
                "V4"
            ),
            "day": str(day.date()),
            "n_rows": len(targets),
            "features": features,
            "feature_rows": targets.to_dict(
                orient="records"
            ),
            "predictions": predictions.tolist(),
            "cat_jour": cat,
            "meteo": meteo,
            "jour_exploitation": jour_exploitation,
        }

    def predict_profile_day(
        self,
        model_name: str,
        day: Any,
        meteo: Dict[str, Any] | None = None,
        jour_exploitation: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Génère les profils horaires complets pour M3 ou M4.

        M3 :
            code_arret + id_zdc + cat_jour + heure_debut

        M4 :
            code_ligne + id_groupofligne + cat_jour + heure_debut

        Les prédictions proviennent directement des baselines
        historiques stockées dans les fichiers .joblib.
        """

        model_name = model_name.upper()

        if model_name not in {"M3", "M4"}:
            raise ValueError(
                f"predict_profile_day() attend M3 ou M4, reçu : {model_name}"
            )

        day = pd.Timestamp(day).normalize()

        # =========================================================
        # 1. Récupération des informations de la journée
        # =========================================================

        if jour_exploitation is None:
            jour_exploitation = self._get_jour_exploitation(day)

        if not jour_exploitation:
            raise ValueError(
                f"Informations de la journée non présentes dans "
                f"transport.dm_jour_exploitation pour le {day.date()}."
            )

        cat_jour = jour_exploitation.get("cat_jour")

        if not cat_jour:
            raise ValueError(
                f"cat_jour absent dans transport.dm_jour_exploitation "
                f"pour le {day.date()}."
            )

        # =========================================================
        # 2. Chargement de la baseline M3 ou M4
        # =========================================================

        model_obj = self.loader.get(model_name)

        if not isinstance(model_obj, dict):
            raise ValueError(
                f"{model_name} : l'artefact chargé n'est pas un dictionnaire."
            )

        if "baseline" not in model_obj:
            raise ValueError(
                f"{model_name} : clé 'baseline' absente de l'artefact."
            )

        baseline = model_obj["baseline"].copy()

        # =========================================================
        # 3. Définition des clés
        # =========================================================

        if model_name == "M3":

            entity_cols = [
                "code_arret",
                "id_zdc",
            ]

            profile_keys = [
                "code_arret",
                "id_zdc",
                "cat_jour",
                "heure_debut",
            ]

        else:

            entity_cols = [
                "code_ligne",
                "id_groupofligne",
            ]

            profile_keys = [
                "code_ligne",
                "id_groupofligne",
                "cat_jour",
                "heure_debut",
            ]

        # =========================================================
        # 4. Vérification des colonnes
        # =========================================================

        required_columns = profile_keys + [
            "pourc_validations"
        ]

        missing_columns = [
            col
            for col in required_columns
            if col not in baseline.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{model_name} : colonnes manquantes dans la baseline : "
                f"{missing_columns}"
            )

        # =========================================================
        # 5. Filtre sur cat_jour
        # =========================================================

        baseline_day = baseline[
            baseline["cat_jour"].astype(str) == str(cat_jour)
        ].copy()

        if baseline_day.empty:
            raise ValueError(
                f"{model_name} : aucun profil historique disponible "
                f"pour cat_jour={cat_jour}."
            )

        # =========================================================
        # 6. Toutes les entités
        # =========================================================

        entities = (
            baseline_day[entity_cols]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        if entities.empty:
            raise ValueError(
                f"{model_name} : aucune entité disponible "
                f"pour cat_jour={cat_jour}."
            )

        # =========================================================
        # 7. Création des 24 heures
        # =========================================================

        hours = pd.DataFrame({
            "heure_debut": range(24)
        })

        entities["_tmp_key"] = 1
        hours["_tmp_key"] = 1

        profile_rows = (
            entities
            .merge(hours, on="_tmp_key")
            .drop(columns="_tmp_key")
        )

        # =========================================================
        # 8. Ajout de cat_jour
        # =========================================================

        profile_rows["cat_jour"] = cat_jour

        # =========================================================
        # 9. Jointure avec la baseline
        # =========================================================

        profile_rows = profile_rows.merge(
            baseline_day[
                profile_keys + ["pourc_validations"]
            ],
            on=profile_keys,
            how="left",
        )

        # =========================================================
        # 10. Vérification des valeurs manquantes
        # =========================================================

        missing_predictions = profile_rows[
            profile_rows["pourc_validations"].isna()
        ]

        if not missing_predictions.empty:

            nb_missing = len(missing_predictions)

            raise ValueError(
                f"{model_name} : {nb_missing} profils horaires "
                f"sont absents de la baseline pour "
                f"cat_jour={cat_jour}."
            )

        # =========================================================
        # 11. Construction du résultat
        # =========================================================

        profile_rows["jour"] = day.date()

        profile_rows["pourc_validations_predit"] = (
            profile_rows["pourc_validations"]
            .astype(float)
        )

        profile_rows["modele"] = model_name

        profile_rows["version_modele"] = model_obj.get(
            "nom_modele",
            "baseline_historique"
        )

        # =========================================================
        # 12. Colonnes finales
        # =========================================================

        result_columns = (
            ["jour"]
            + entity_cols
            + [
                "cat_jour",
                "heure_debut",
                "pourc_validations_predit",
                "modele",
                "version_modele",
            ]
        )

        profile_rows = profile_rows[
            result_columns
        ].copy()

        # =========================================================
        # 13. Tri
        # =========================================================

        profile_rows = profile_rows.sort_values(
            entity_cols + ["heure_debut"]
        ).reset_index(drop=True)

        # =========================================================
        # 14. Résultat
        # =========================================================

        return {
            "day": day,
            "model_name": model_name,
            "cat_jour": cat_jour,
            "jour_exploitation": jour_exploitation,
            "n_rows": len(profile_rows),
            "n_entities": len(entities),
            "profile_rows": profile_rows,
            "version": model_obj.get(
                "nom_modele",
                "baseline_historique"
            ),
        }
