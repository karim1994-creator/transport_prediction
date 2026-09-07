from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from sqlalchemy import create_engine, text


class PredictionHistory:
    """Historisation des prédictions M1/M2/M3/M4 dans les 4 tables métier."""

    def __init__(self, database_url: str):
        self.engine = create_engine(database_url)

    @staticmethod
    def _value(row: Dict[str, Any], key: str) -> Any:
        value = row.get(key)
        if pd.isna(value):
            return None
        return value

    def save_m1_predictions(
        self,
        feature_rows: List[Dict[str, Any]],
        predictions: Iterable[float],
        version: str,
    ) -> int:
        sql = text("""
            INSERT INTO prediction.frequentation_journaliere_ferre (
                jour, code_arret, id_zdc, mois, trimestre, semaine,
                jour_semaine, jour_du_mois, jour_annee, est_debut_mois,
                est_fin_mois, temperature_moyenne, pluie_totale,
                vitesse_vent_moyenne, cat_jour, code_meteo,
                moyenne_historique_causale, mediane_historique_causale,
                moyenne_historique_cat_causale, mediane_historique_cat_causale,
                lag_1, lag_7, lag_14, moyenne_mobile_7, moyenne_mobile_28,
                nb_vald_predit, modele, version_modele
            ) VALUES (
                :jour, :code_arret, :id_zdc, :mois, :trimestre, :semaine,
                :jour_semaine, :jour_du_mois, :jour_annee, :est_debut_mois,
                :est_fin_mois, :temperature_moyenne, :pluie_totale,
                :vitesse_vent_moyenne, :cat_jour, :code_meteo,
                :moyenne_historique_causale, :mediane_historique_causale,
                :moyenne_historique_cat_causale, :mediane_historique_cat_causale,
                :lag_1, :lag_7, :lag_14, :moyenne_mobile_7, :moyenne_mobile_28,
                :prediction, 'M1', :version
            )
        """)
        payload = []
        for row, prediction in zip(feature_rows, predictions):
            payload.append({
                "jour": pd.Timestamp(row["jour"]).date(),
                "code_arret": str(row["code_arret"]),
                "id_zdc": str(row["id_zdc"]),
                **{k: self._value(row, k) for k in [
                    "mois", "trimestre", "semaine", "jour_semaine",
                    "jour_du_mois", "jour_annee", "est_debut_mois",
                    "est_fin_mois", "temperature_moyenne", "pluie_totale",
                    "vitesse_vent_moyenne", "cat_jour", "code_meteo",
                    "moyenne_historique_causale", "mediane_historique_causale",
                    "moyenne_historique_cat_causale", "mediane_historique_cat_causale",
                    "lag_1", "lag_7", "lag_14", "moyenne_mobile_7",
                    "moyenne_mobile_28",
                ]},
                "prediction": float(prediction),
                "version": version,
            })
        if payload:
            with self.engine.begin() as conn:
                conn.execute(sql, payload)
        return len(payload)

    def save_m2_predictions(
        self,
        feature_rows: List[Dict[str, Any]],
        predictions: Iterable[float],
        version: str,
    ) -> int:
        sql = text("""
            INSERT INTO prediction.frequentation_journaliere_surface (
                jour, code_ligne, id_groupoflines, mois, trimestre, semaine,
                jour_semaine, jour_du_mois, jour_annee, est_debut_mois,
                est_fin_mois, temperature_moyenne, pluie_totale,
                vitesse_vent_moyenne, cat_jour, code_meteo,
                moyenne_historique_causale, mediane_historique_causale,
                moyenne_historique_cat_causale, mediane_historique_cat_causale,
                lag_1, lag_7, lag_14, moyenne_mobile_7, moyenne_mobile_28,
                nb_vald_predit, modele, version_modele
            ) VALUES (
                :jour, :code_ligne, :id_groupoflines, :mois, :trimestre, :semaine,
                :jour_semaine, :jour_du_mois, :jour_annee, :est_debut_mois,
                :est_fin_mois, :temperature_moyenne, :pluie_totale,
                :vitesse_vent_moyenne, :cat_jour, :code_meteo,
                :moyenne_historique_causale, :mediane_historique_causale,
                :moyenne_historique_cat_causale, :mediane_historique_cat_causale,
                :lag_1, :lag_7, :lag_14, :moyenne_mobile_7, :moyenne_mobile_28,
                :prediction, 'M2', :version
            )
        """)
        payload = []
        for row, prediction in zip(feature_rows, predictions):
            payload.append({
                "jour": pd.Timestamp(row["jour"]).date(),
                "code_ligne": str(row["code_ligne"]),
                "id_groupoflines": str(row["id_groupoflines"]),
                **{k: self._value(row, k) for k in [
                    "mois", "trimestre", "semaine", "jour_semaine",
                    "jour_du_mois", "jour_annee", "est_debut_mois",
                    "est_fin_mois", "temperature_moyenne", "pluie_totale",
                    "vitesse_vent_moyenne", "cat_jour", "code_meteo",
                    "moyenne_historique_causale", "mediane_historique_causale",
                    "moyenne_historique_cat_causale", "mediane_historique_cat_causale",
                    "lag_1", "lag_7", "lag_14", "moyenne_mobile_7",
                    "moyenne_mobile_28",
                ]},
                "prediction": float(prediction),
                "version": version,
            })
        if payload:
            with self.engine.begin() as conn:
                conn.execute(sql, payload)
        return len(payload)

    def save_profile_prediction(
        self,
        model_name: str,
        day: Any,
        profile: List[Dict[str, Any]],
        version: str,
        profile_meta: Optional[Dict[str, Any]] = None,
    ) -> int:

        profile_meta = profile_meta or {}

        if model_name == "M3":

            code_arret = profile_meta.get("code_arret")
            id_zdc = profile_meta.get("id_zdc")
            cat_jour = profile_meta.get("cat_jour")

            if not code_arret or not id_zdc or not cat_jour:
                raise ValueError(
                    "M3 : métadonnées de profil incomplètes "
                    "(code_arret, id_zdc, cat_jour)."
                )

            sql = text("""
                INSERT INTO prediction.profils_horaires_ferre (
                    jour,
                    code_arret,
                    id_zdc,
                    cat_jour,
                    heure_debut,
                    pourc_validations_predit,
                    modele,
                    version_modele
                )
                VALUES (
                    :jour,
                    :code_arret,
                    :id_zdc,
                    :cat_jour,
                    :heure_debut,
                    :prediction,
                    'M3',
                    :version
                )
            """)

            payload = []

            for item in profile:
                payload.append({
                    "jour": pd.Timestamp(day).date(),
                    "code_arret": str(code_arret),
                    "id_zdc": str(id_zdc),
                    "cat_jour": str(cat_jour),
                    "heure_debut": int(item["heure_debut"]),
                    "prediction": float(item["prediction_percent"]),
                    "version": version,
                })

        elif model_name == "M4":

            code_ligne = profile_meta.get("code_ligne")

            # L'UI/API peut utiliser id_groupoflines,
            # mais la table historique utilise id_groupofligne.
            id_group = (
                profile_meta.get("id_groupofligne")
                or profile_meta.get("id_groupoflines")
            )

            cat_jour = profile_meta.get("cat_jour")

            if not code_ligne or not id_group or not cat_jour:
                raise ValueError(
                    "M4 : métadonnées de profil incomplètes "
                    "(code_ligne, id_groupofligne, cat_jour)."
                )

            sql = text("""
                INSERT INTO prediction.profils_horaires_surface (
                    jour,
                    code_ligne,
                    id_groupofligne,
                    cat_jour,
                    heure_debut,
                    pourc_validations_predit,
                    modele,
                    version_modele
                )
                VALUES (
                    :jour,
                    :code_ligne,
                    :id_groupofligne,
                    :cat_jour,
                    :heure_debut,
                    :prediction,
                    'M4',
                    :version
                )
            """)

            payload = []

            for item in profile:
                payload.append({
                    "jour": pd.Timestamp(day).date(),
                    "code_ligne": str(code_ligne),
                    "id_groupofligne": str(id_group),
                    "cat_jour": str(cat_jour),
                    "heure_debut": int(item["heure_debut"]),
                    "prediction": float(item["prediction_percent"]),
                    "version": version,
                })

        else:
            raise ValueError("model_name doit être M3 ou M4")

        if payload:
            with self.engine.begin() as conn:
                conn.execute(sql, payload)

        return len(payload)



    def save_m3_full_day(
        self,
        day: Any,
        profile_rows: pd.DataFrame,
        version: str,
    ) -> int:
        """
        Historise le profil complet M3 (24h × tous les arrêts/ZDC).
        """

        if profile_rows is None or profile_rows.empty:
            return 0

        sql = text("""
            INSERT INTO prediction.profils_horaires_ferre (
                jour,
                code_arret,
                id_zdc,
                cat_jour,
                heure_debut,
                pourc_validations_predit,
                modele,
                version_modele
            )
            VALUES (
                :jour,
                :code_arret,
                :id_zdc,
                :cat_jour,
                :heure_debut,
                :prediction,
                'M3',
                :version
            )
        """)

        payload = []

        for _, row in profile_rows.iterrows():
            payload.append({
                "jour": pd.Timestamp(day).date(),
                "code_arret": str(row["code_arret"]),
                "id_zdc": str(row["id_zdc"]),
                "cat_jour": str(row["cat_jour"]),
                "heure_debut": int(row["heure_debut"]),
                "prediction": float(
                    row["pourc_validations_predit"]
                ),
                "version": version,
            })

        if payload:
            with self.engine.begin() as conn:
                conn.execute(sql, payload)

        return len(payload)


    def save_m4_full_day(
        self,
        day: Any,
        profile_rows: pd.DataFrame,
        version: str,
    ) -> int:
        """
        Historise le profil complet M4 (24h × toutes les lignes/groupes).
        """

        if profile_rows is None or profile_rows.empty:
            return 0

        sql = text("""
            INSERT INTO prediction.profils_horaires_surface (
                jour,
                code_ligne,
                id_groupofligne,
                cat_jour,
                heure_debut,
                pourc_validations_predit,
                modele,
                version_modele
            )
            VALUES (
                :jour,
                :code_ligne,
                :id_groupofligne,
                :cat_jour,
                :heure_debut,
                :prediction,
                'M4',
                :version
            )
        """)

        payload = []

        for _, row in profile_rows.iterrows():
            payload.append({
                "jour": pd.Timestamp(day).date(),
                "code_ligne": str(row["code_ligne"]),
                "id_groupofligne": str(
                    row["id_groupofligne"]
                ),
                "cat_jour": str(row["cat_jour"]),
                "heure_debut": int(row["heure_debut"]),
                "prediction": float(
                    row["pourc_validations_predit"]
                ),
                "version": version,
            })

        if payload:
            with self.engine.begin() as conn:
                conn.execute(sql, payload)

        return len(payload)



    def latest_day_summary(self) -> Optional[Dict[str, Any]]:
        queries = {
            "M1": "SELECT MAX(jour) FROM prediction.frequentation_journaliere_ferre",
            "M2": "SELECT MAX(jour) FROM prediction.frequentation_journaliere_surface",
            "M3": "SELECT MAX(jour) FROM prediction.profils_horaires_ferre",
            "M4": "SELECT MAX(jour) FROM prediction.profils_horaires_surface",
        }
        with self.engine.begin() as conn:
            dates = {name: conn.execute(text(sql)).scalar() for name, sql in queries.items()}
        valid = [d for d in dates.values() if d is not None]
        if not valid:
            return None
        latest = max(valid)
        result = {"jour": latest, "modeles": [k for k, v in dates.items() if v == latest]}
        return result



    def dashboard(self):
        """
        Retourne les informations nécessaires au dashboard
        de la dernière journée prédite.
        """

        latest = self.latest_day_summary()

        if latest is None:
            return None

        day = latest["jour"]
        modeles = latest["modeles"]

        result = {
            "jour": str(day),
            "modeles": modeles,
            "nb_lignes": 0,
            "totaux": {},
            "top_heures": [],
            "analyse": "",
        }

        with self.engine.begin() as conn:

            # --------------------------------------------------
            # Nombre de prédictions journalières
            # --------------------------------------------------

            if "M1" in modeles:
                row = conn.execute(text("""
                    SELECT
                        COUNT(*) AS n,
                        COALESCE(SUM(nb_vald_predit), 0) AS total
                    FROM prediction.frequentation_journaliere_ferre
                    WHERE jour = :day
                """), {"day": day}).mappings().one()

                result["nb_lignes"] += int(row["n"])
                result["totaux"]["M1"] = float(row["total"])

            if "M2" in modeles:
                row = conn.execute(text("""
                    SELECT
                        COUNT(*) AS n,
                        COALESCE(SUM(nb_vald_predit), 0) AS total
                    FROM prediction.frequentation_journaliere_surface
                    WHERE jour = :day
                """), {"day": day}).mappings().one()

                result["nb_lignes"] += int(row["n"])
                result["totaux"]["M2"] = float(row["total"])

            # --------------------------------------------------
            # Heure de pointe M3
            # --------------------------------------------------

            if "M3" in modeles:
                row = conn.execute(text("""
                    SELECT
                        heure_debut,
                        AVG(pourc_validations_predit) AS prediction
                    FROM prediction.profils_horaires_ferre
                    WHERE jour = :day
                    GROUP BY heure_debut
                    ORDER BY prediction DESC
                    LIMIT 1
                """), {"day": day}).mappings().first()

                if row:
                    result["top_heures"].append({
                        "modele": "M3",
                        "heure": int(row["heure_debut"]),
                        "prediction_percent": round(
                            float(row["prediction"]), 2
                        ),
                    })

            # --------------------------------------------------
            # Heure de pointe M4
            # --------------------------------------------------

            if "M4" in modeles:
                row = conn.execute(text("""
                    SELECT
                        heure_debut,
                        AVG(pourc_validations_predit) AS prediction
                    FROM prediction.profils_horaires_surface
                    WHERE jour = :day
                    GROUP BY heure_debut
                    ORDER BY prediction DESC
                    LIMIT 1
                """), {"day": day}).mappings().first()

                if row:
                    result["top_heures"].append({
                        "modele": "M4",
                        "heure": int(row["heure_debut"]),
                        "prediction_percent": round(
                            float(row["prediction"]), 2
                        ),
                    })

        # ------------------------------------------------------
        # Petite analyse automatique
        # ------------------------------------------------------

        analyses = []

        if "M1" in result["totaux"]:
            analyses.append(
                f"M1 prévoit environ "
                f"{result['totaux']['M1']:,.0f} validations "
                f"sur le périmètre ferré."
            )

        if "M2" in result["totaux"]:
            analyses.append(
                f"M2 prévoit environ "
                f"{result['totaux']['M2']:,.0f} validations "
                f"sur le périmètre surface."
            )

        if result["top_heures"]:
            heures = [
                f"{x['modele']} : {x['heure']:02d}h"
                for x in result["top_heures"]
            ]
            analyses.append(
                "Heures de pointe : "
                + " ; ".join(heures)
                + "."
            )

        result["analyse"] = " ".join(analyses)

        return result
