import pandas as pd

from fastapi import APIRouter, HTTPException, Request

from prometheus_client import Counter

from app.schemas.prediction import (
    MLPredictionRequest,
    ProfilePredictionRequest,
)


router = APIRouter()


# ============================================================
# METRIQUES ML PROMETHEUS
# ============================================================

PREDICTION_ERRORS_TOTAL = Counter(
    "ml_prediction_errors_total",
    "Nombre total d'erreurs de prediction par modele",
    ["model"],
)


# ============================================================
# ENTITES PREDITES ET LANCEMENTS DES MODELES
# ============================================================

PREDICTION_ENTITIES_TOTAL = Counter(
    "ml_prediction_entities_total",
    "Nombre total d'entites effectivement predites",
    ["model", "mode"],
)

MODEL_RUNS_TOTAL = Counter(
    "ml_model_runs_total",
    "Nombre total de lancements des modeles",
    ["model", "mode"],
)


# ============================================================
# HEALTH
# ============================================================

@router.get("/health")
def health(request: Request):

    loaded = len(
        request.app.state.model_loader.models
    )

    return {
        "status": "ok" if loaded == 4 else "degraded",
        "models_loaded": loaded,
    }


# ============================================================
# MODELS
# ============================================================

@router.get("/models")
def models(request: Request):

    return request.app.state.model_loader.status()


# ============================================================
# M1
# ============================================================

@router.post("/predict/m1")
def predict_m1(
    body: MLPredictionRequest,
    request: Request,
):

    try:

        MODEL_RUNS_TOTAL.labels(
            model="M1",
            mode="single",
        ).inc()

        result = request.app.state.prediction_engine.predict_ml(
            "M1",
            body.rows,
        )

        request.app.state.prediction_history.save_m1_predictions(
            feature_rows=result["feature_rows"],
            predictions=result["predictions"],
            version=result.get("version", "V4"),
        )

        PREDICTION_ENTITIES_TOTAL.labels(
            model="M1",
            mode="single",
        ).inc(
            len(result.get("predictions", []))
        )

        return result

    except Exception as exc:

        PREDICTION_ERRORS_TOTAL.labels(
            model="M1"
        ).inc()

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


# ============================================================
# M2
# ============================================================

@router.post("/predict/m2")
def predict_m2(
    body: MLPredictionRequest,
    request: Request,
):

    try:

        MODEL_RUNS_TOTAL.labels(
            model="M2",
            mode="single",
        ).inc()

        result = request.app.state.prediction_engine.predict_ml(
            "M2",
            body.rows,
        )

        request.app.state.prediction_history.save_m2_predictions(
            feature_rows=result["feature_rows"],
            predictions=result["predictions"],
            version=result.get("version", "V4"),
        )

        PREDICTION_ENTITIES_TOTAL.labels(
            model="M2",
            mode="single",
        ).inc(
            len(result.get("predictions", []))
        )

        return result

    except Exception as exc:

        PREDICTION_ERRORS_TOTAL.labels(
            model="M2"
        ).inc()

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


# ============================================================
# M3
# PROFIL HORAIRE - FERRE
# ============================================================

@router.post("/predict/m3")
def predict_m3(
    body: ProfilePredictionRequest,
    request: Request,
):

    try:

        MODEL_RUNS_TOTAL.labels(
            model="M3",
            mode="single",
        ).inc()

        result = request.app.state.prediction_engine.predict_full_profile(
            "M3",
            body.profile,
        )

        # ----------------------------------------------------
        # MONITORING PSI M3
        # ----------------------------------------------------

        request.app.state.drift_monitor.update(
            "M3",
            result["profile"],
        )

        # ----------------------------------------------------
        # VALIDATION DES METADONNEES
        # ----------------------------------------------------

        profile_request = dict(
            body.profile or {}
        )

        day = profile_request.get("jour")
        code_arret = profile_request.get("code_arret")
        id_zdc = profile_request.get("id_zdc")
        cat_jour = profile_request.get("cat_jour")

        if not day:
            raise ValueError(
                "M3 : le champ 'jour' est obligatoire pour l'historisation."
            )

        if not code_arret or not id_zdc:
            raise ValueError(
                "M3 : code_arret et id_zdc sont obligatoires "
                "pour l'historisation."
            )

        if not cat_jour:
            raise ValueError(
                "M3 : cat_jour est obligatoire "
                "pour l'historisation du profil horaire."
            )

        # ----------------------------------------------------
        # HISTORISATION
        # ----------------------------------------------------

        request.app.state.prediction_history.save_profile_prediction(
            model_name="M3",
            day=day,
            profile=result["profile"],
            version=result.get(
                "version",
                "production",
            ),
            profile_meta={
                "code_arret": code_arret,
                "id_zdc": id_zdc,
                "cat_jour": cat_jour,
            },
        )

        # ----------------------------------------------------
        # PROMETHEUS
        # ----------------------------------------------------

        PREDICTION_ENTITIES_TOTAL.labels(
            model="M3",
            mode="single",
        ).inc()

        return result

    except Exception as exc:

        PREDICTION_ERRORS_TOTAL.labels(
            model="M3"
        ).inc()

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


# ============================================================
# M4
# PROFIL HORAIRE - SURFACE
# ============================================================

@router.post("/predict/m4")
def predict_m4(
    body: ProfilePredictionRequest,
    request: Request,
):

    try:

        MODEL_RUNS_TOTAL.labels(
            model="M4",
            mode="single",
        ).inc()

        result = request.app.state.prediction_engine.predict_full_profile(
            "M4",
            body.profile,
        )

        # ----------------------------------------------------
        # MONITORING PSI M4
        # ----------------------------------------------------

        request.app.state.drift_monitor.update(
            "M4",
            result["profile"],
        )

        # ----------------------------------------------------
        # VALIDATION DES METADONNEES
        # ----------------------------------------------------

        profile_request = dict(
            body.profile or {}
        )

        day = profile_request.get("jour")
        code_ligne = profile_request.get("code_ligne")

        id_groupofligne = (
            profile_request.get("id_groupofligne")
            or profile_request.get("id_groupoflines")
        )

        cat_jour = profile_request.get("cat_jour")

        if not day:
            raise ValueError(
                "M4 : le champ 'jour' est obligatoire pour l'historisation."
            )

        if not code_ligne or not id_groupofligne:
            raise ValueError(
                "M4 : code_ligne et id_groupofligne sont obligatoires "
                "pour l'historisation."
            )

        if not cat_jour:
            raise ValueError(
                "M4 : cat_jour est obligatoire "
                "pour l'historisation du profil horaire."
            )

        # ----------------------------------------------------
        # HISTORISATION
        # ----------------------------------------------------

        request.app.state.prediction_history.save_profile_prediction(
            model_name="M4",
            day=day,
            profile=result["profile"],
            version=result.get(
                "version",
                "production",
            ),
            profile_meta={
                "code_ligne": code_ligne,
                "id_groupofligne": id_groupofligne,
                "cat_jour": cat_jour,
            },
        )

        # ----------------------------------------------------
        # PROMETHEUS
        # ----------------------------------------------------

        PREDICTION_ENTITIES_TOTAL.labels(
            model="M4",
            mode="single",
        ).inc()

        return result

    except Exception as exc:

        PREDICTION_ERRORS_TOTAL.labels(
            model="M4"
        ).inc()

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


# ============================================================
# MONITORING DERIVE - PSI
# ============================================================

@router.get("/monitoring/drift")
def drift_status(request: Request):

    return request.app.state.drift_monitor.status()


# ============================================================
# PREDICTION JOURNALIERE M1
# ============================================================

@router.post("/predict/day/m1")
def predict_day_m1(
    body: dict,
    request: Request,
):

    try:

        MODEL_RUNS_TOTAL.labels(
            model="M1",
            mode="day",
        ).inc()

        result = request.app.state.day_prediction_service.predict_ml_day(
            "M1",
            body["jour"],
        )

        # ====================================================
        # MONITORING PSI M1
        #
        # M1 est un modele journalier.
        # On compare donc la distribution des predictions
        # du jour avec la reference M1 du fichier
        # drift_reference.json.
        #
        # Aucun champ heure n'est ajoute.
        # ====================================================

        request.app.state.drift_monitor.update_from_predictions(
            "M1",
            result["predictions"],
        )

        # ----------------------------------------------------
        # HISTORISATION
        # ----------------------------------------------------

        request.app.state.prediction_history.save_m1_predictions(
            feature_rows=result["feature_rows"],
            predictions=result["predictions"],
            version=result["version"],
        )

        # ----------------------------------------------------
        # PROMETHEUS
        # ----------------------------------------------------

        PREDICTION_ENTITIES_TOTAL.labels(
            model="M1",
            mode="day",
        ).inc(
            result["n_rows"]
        )

        return {
            "model": "M1",
            "jour": result["day"],
            "n_predictions": result["n_rows"],
            "total_predictions": float(
                sum(result["predictions"])
            ),
            "version": result["version"],
        }

    except Exception as exc:

        PREDICTION_ERRORS_TOTAL.labels(
            model="M1"
        ).inc()

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


# ============================================================
# PREDICTION JOURNALIERE M2
# ============================================================

@router.post("/predict/day/m2")
def predict_day_m2(
    body: dict,
    request: Request,
):

    try:

        MODEL_RUNS_TOTAL.labels(
            model="M2",
            mode="day",
        ).inc()

        result = request.app.state.day_prediction_service.predict_ml_day(
            "M2",
            body["jour"],
        )

        # ====================================================
        # MONITORING PSI M2
        #
        # M2 est un modele journalier.
        # On compare donc la distribution des predictions
        # du jour avec la reference M2 du fichier
        # drift_reference.json.
        # ====================================================

        request.app.state.drift_monitor.update_from_predictions(
            "M2",
            result["predictions"],
        )

        # ----------------------------------------------------
        # HISTORISATION
        # ----------------------------------------------------

        request.app.state.prediction_history.save_m2_predictions(
            feature_rows=result["feature_rows"],
            predictions=result["predictions"],
            version=result["version"],
        )

        # ----------------------------------------------------
        # PROMETHEUS
        # ----------------------------------------------------

        PREDICTION_ENTITIES_TOTAL.labels(
            model="M2",
            mode="day",
        ).inc(
            result["n_rows"]
        )

        return {
            "model": "M2",
            "jour": result["day"],
            "n_predictions": result["n_rows"],
            "total_predictions": float(
                sum(result["predictions"])
            ),
            "version": result["version"],
        }

    except Exception as exc:

        PREDICTION_ERRORS_TOTAL.labels(
            model="M2"
        ).inc()

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


# ============================================================
# PREDICTION JOURNALIERE M3
# ============================================================

@router.post("/predict/day/m3")
def predict_day_m3(
    body: dict,
    request: Request,
):

    try:

        MODEL_RUNS_TOTAL.labels(
            model="M3",
            mode="day",
        ).inc()

        result = request.app.state.day_prediction_service.predict_profile_day(
            "M3",
            body["jour"],
        )

        request.app.state.prediction_history.save_m3_full_day(
            result["day"],
            result["profile_rows"],
            result["version"],
        )

        # ----------------------------------------------------
        # MONITORING PSI M3
        # ----------------------------------------------------

        profile_rows = result["profile_rows"]

        if (
            profile_rows is not None
            and not profile_rows.empty
        ):

            profile_for_drift = []

            grouped = (
                profile_rows
                .groupby("heure_debut")[
                    "pourc_validations_predit"
                ]
                .mean()
                .reset_index()
            )

            for _, row in grouped.iterrows():

                profile_for_drift.append(
                    {
                        "heure_debut": int(
                            row["heure_debut"]
                        ),
                        "prediction_percent": float(
                            row[
                                "pourc_validations_predit"
                            ]
                        ),
                    }
                )

            request.app.state.drift_monitor.update(
                "M3",
                profile_for_drift,
            )

        # ----------------------------------------------------
        # PROMETHEUS
        # ----------------------------------------------------

        PREDICTION_ENTITIES_TOTAL.labels(
            model="M3",
            mode="day",
        ).inc(
            result["n_entities"]
        )

        return {
            "model": "M3",
            "jour": result["day"],
            "n_predictions": result["n_rows"],
            "cat_jour": result["cat_jour"],
            "version": result["version"],
        }

    except Exception as exc:

        PREDICTION_ERRORS_TOTAL.labels(
            model="M3"
        ).inc()

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


# ============================================================
# PREDICTION JOURNALIERE M4
# ============================================================

@router.post("/predict/day/m4")
def predict_day_m4(
    body: dict,
    request: Request,
):

    try:

        MODEL_RUNS_TOTAL.labels(
            model="M4",
            mode="day",
        ).inc()

        result = request.app.state.day_prediction_service.predict_profile_day(
            "M4",
            body["jour"],
        )

        request.app.state.prediction_history.save_m4_full_day(
            result["day"],
            result["profile_rows"],
            result["version"],
        )

        # ----------------------------------------------------
        # MONITORING PSI M4
        # ----------------------------------------------------

        profile_rows = result["profile_rows"]

        if (
            profile_rows is not None
            and not profile_rows.empty
        ):

            profile_for_drift = []

            grouped = (
                profile_rows
                .groupby("heure_debut")[
                    "pourc_validations_predit"
                ]
                .mean()
                .reset_index()
            )

            for _, row in grouped.iterrows():

                profile_for_drift.append(
                    {
                        "heure_debut": int(
                            row["heure_debut"]
                        ),
                        "prediction_percent": float(
                            row[
                                "pourc_validations_predit"
                            ]
                        ),
                    }
                )

            request.app.state.drift_monitor.update(
                "M4",
                profile_for_drift,
            )

        # ----------------------------------------------------
        # PROMETHEUS
        # ----------------------------------------------------

        PREDICTION_ENTITIES_TOTAL.labels(
            model="M4",
            mode="day",
        ).inc(
            result["n_entities"]
        )

        return {
            "model": "M4",
            "jour": result["day"],
            "n_predictions": result["n_rows"],
            "cat_jour": result["cat_jour"],
            "version": result["version"],
        }

    except Exception as exc:

        PREDICTION_ERRORS_TOTAL.labels(
            model="M4"
        ).inc()

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


# ============================================================
# DASHBOARD DRIFT
# ============================================================

@router.get("/prediction/dashboard/latest")
def prediction_dashboard_latest(
    request: Request,
):

    return (
        request.app.state.prediction_history.dashboard()
        or {
            "jour": None,
            "modeles": [],
            "nb_lignes": 0,
            "totaux": {},
            "top_heures": [],
            "analyse": "Aucune prédiction enregistrée.",
        }
    )


# ============================================================
# PREDICTION COMPLETE D'UNE JOURNEE
# M1 + M2 + M3 + M4
# ============================================================

@router.post("/predict/day")
def predict_full_day(
    body: dict,
    request: Request,
):
    """
    Lance les prédictions M1, M2, M3 et M4 pour une journée.

    Monitoring PSI :

    M1 :
        distribution des predictions journalieres.

    M2 :
        distribution des predictions journalieres.

    M3 :
        profil horaire de 24 valeurs.

    M4 :
        profil horaire de 24 valeurs.
    """

    try:

        # =====================================================
        # 1. DATE
        # =====================================================

        jour = body.get("jour")

        if not jour:

            raise HTTPException(
                status_code=400,
                detail="Le champ 'jour' est obligatoire.",
            )

        try:

            jour_ts = pd.Timestamp(
                jour
            ).normalize()

            jour_str = str(
                jour_ts.date()
            )

        except Exception:

            raise HTTPException(
                status_code=400,
                detail=f"Date invalide : {jour}",
            )

        # =====================================================
        # 2. SERVICES
        # =====================================================

        service = (
            request
            .app
            .state
            .day_prediction_service
        )

        history = (
            request
            .app
            .state
            .prediction_history
        )

        drift_monitor = (
            request
            .app
            .state
            .drift_monitor
        )

        # =====================================================
        # 3. METEO
        # =====================================================

        meteo = body.get(
            "meteo"
        )

        if meteo is None:

            meteo = service._get_meteo_day(
                jour_ts
            )

        meteo_absente = not bool(
            meteo
        )

        # =====================================================
        # 4. JOUR D'EXPLOITATION
        # =====================================================

        jour_exploitation = (
            body.get(
                "jour_exploitation"
            )
        )

        if jour_exploitation is None:

            jour_exploitation = (
                service._get_jour_exploitation(
                    jour_ts
                )
            )

        jour_exploitation_absent = not bool(
            jour_exploitation
        )

        # =====================================================
        # 5. CAT_JOUR
        # =====================================================

        cat_jour_absent = False

        if jour_exploitation:

            cat_jour = (
                jour_exploitation.get(
                    "cat_jour"
                )
            )

            if not cat_jour:

                cat_jour_absent = True

        # =====================================================
        # 6. DONNEES MANQUANTES
        # =====================================================

        if (
            meteo_absente
            or jour_exploitation_absent
            or cat_jour_absent
        ):

            return {

                "statut": "donnees_manquantes",

                "jour": jour_str,

                "donnees": {

                    "meteo": (
                        meteo
                        if meteo
                        else None
                    ),

                    "jour_exploitation": (
                        jour_exploitation
                        if jour_exploitation
                        else None
                    ),
                },

                "manquantes": {

                    "meteo_absente": (
                        meteo_absente
                    ),

                    "jour_exploitation_absent": (
                        jour_exploitation_absent
                    ),

                    "cat_jour_absent": (
                        cat_jour_absent
                    ),
                },

                "message": (
                    "Certaines données nécessaires "
                    "à la prédiction sont absentes "
                    "des tables DM. "
                    "Veuillez les renseigner manuellement."
                ),
            }

        # =====================================================
        # 7. RESULTAT GLOBAL
        # =====================================================

        result = {

            "statut": "ok",

            "jour": jour_str,

            "modeles": {},

            "nb_predictions_total": 0,

            "erreurs": [],
        }

        # =====================================================
        # 8. M1
        # =====================================================

        try:

            MODEL_RUNS_TOTAL.labels(
                model="M1",
                mode="day",
            ).inc()

            m1 = service.predict_ml_day(
                "M1",
                jour_ts,
                meteo=meteo,
                jour_exploitation=jour_exploitation,
            )

            # =================================================
            # MONITORING PSI M1
            # =================================================
            #
            # IMPORTANT :
            # M1 n'a PAS de dimension horaire.
            #
            # On envoie directement toutes les predictions
            # journalieres au DriftMonitor.
            #
            # Le DriftMonitor utilise les bins M1 du
            # drift_reference.json.
            # =================================================

            drift_monitor.update_from_predictions(
                "M1",
                m1["predictions"],
            )

            # -------------------------------------------------
            # HISTORISATION
            # -------------------------------------------------

            n_saved = (
                history.save_m1_predictions(
                    feature_rows=m1[
                        "feature_rows"
                    ],
                    predictions=m1[
                        "predictions"
                    ],
                    version=m1[
                        "version"
                    ],
                )
            )

            # -------------------------------------------------
            # PROMETHEUS
            # -------------------------------------------------

            PREDICTION_ENTITIES_TOTAL.labels(
                model="M1",
                mode="day",
            ).inc(
                m1["n_rows"]
            )

            result[
                "modeles"
            ][
                "M1"
            ] = {

                "statut": "ok",

                "n_predictions": (
                    m1["n_rows"]
                ),

                "n_enregistrees": (
                    n_saved
                ),

                "total_predictions": float(
                    sum(
                        m1[
                            "predictions"
                        ]
                    )
                ),

                "version": (
                    m1["version"]
                ),

                "cat_jour": (
                    m1["cat_jour"]
                ),
            }

            result[
                "nb_predictions_total"
            ] += n_saved

        except Exception as exc:

            PREDICTION_ERRORS_TOTAL.labels(
                model="M1"
            ).inc()

            result[
                "modeles"
            ][
                "M1"
            ] = {

                "statut": "erreur",

                "erreur": str(
                    exc
                ),
            }

            result[
                "erreurs"
            ].append(
                f"M1 : {str(exc)}"
            )

        # =====================================================
        # 9. M2
        # =====================================================

        try:

            MODEL_RUNS_TOTAL.labels(
                model="M2",
                mode="day",
            ).inc()

            m2 = service.predict_ml_day(
                "M2",
                jour_ts,
                meteo=meteo,
                jour_exploitation=jour_exploitation,
            )

            # =================================================
            # MONITORING PSI M2
            # =================================================
            #
            # M2 n'a PAS de dimension horaire.
            #
            # On envoie directement toutes les predictions
            # journalieres au DriftMonitor.
            #
            # Le DriftMonitor utilise les bins M2 du
            # drift_reference.json.
            # =================================================

            drift_monitor.update_from_predictions(
                "M2",
                m2["predictions"],
            )

            # -------------------------------------------------
            # HISTORISATION
            # -------------------------------------------------

            n_saved = (
                history.save_m2_predictions(
                    feature_rows=m2[
                        "feature_rows"
                    ],
                    predictions=m2[
                        "predictions"
                    ],
                    version=m2[
                        "version"
                    ],
                )
            )

            # -------------------------------------------------
            # PROMETHEUS
            # -------------------------------------------------

            PREDICTION_ENTITIES_TOTAL.labels(
                model="M2",
                mode="day",
            ).inc(
                m2["n_rows"]
            )

            result[
                "modeles"
            ][
                "M2"
            ] = {

                "statut": "ok",

                "n_predictions": (
                    m2["n_rows"]
                ),

                "n_enregistrees": (
                    n_saved
                ),

                "total_predictions": float(
                    sum(
                        m2[
                            "predictions"
                        ]
                    )
                ),

                "version": (
                    m2["version"]
                ),

                "cat_jour": (
                    m2["cat_jour"]
                ),
            }

            result[
                "nb_predictions_total"
            ] += n_saved

        except Exception as exc:

            PREDICTION_ERRORS_TOTAL.labels(
                model="M2"
            ).inc()

            result[
                "modeles"
            ][
                "M2"
            ] = {

                "statut": "erreur",

                "erreur": str(
                    exc
                ),
            }

            result[
                "erreurs"
            ].append(
                f"M2 : {str(exc)}"
            )

        # =====================================================
        # 10. M3
        # =====================================================

        try:

            MODEL_RUNS_TOTAL.labels(
                model="M3",
                mode="day",
            ).inc()

            m3 = service.predict_profile_day(
                "M3",
                jour_ts,
                jour_exploitation=jour_exploitation,
            )

            # -------------------------------------------------
            # HISTORISATION
            # -------------------------------------------------

            n_saved = (
                history.save_m3_full_day(
                    m3["day"],
                    m3["profile_rows"],
                    m3["version"],
                )
            )

            # -------------------------------------------------
            # MONITORING PSI M3
            # -------------------------------------------------

            profile_rows = (
                m3["profile_rows"]
            )

            if (
                profile_rows is not None
                and not profile_rows.empty
            ):

                profile_for_drift = []

                grouped = (
                    profile_rows
                    .groupby(
                        "heure_debut"
                    )[
                        "pourc_validations_predit"
                    ]
                    .mean()
                    .reset_index()
                )

                for _, row in grouped.iterrows():

                    profile_for_drift.append(
                        {
                            "heure_debut": int(
                                row[
                                    "heure_debut"
                                ]
                            ),
                            "prediction_percent": float(
                                row[
                                    "pourc_validations_predit"
                                ]
                            ),
                        }
                    )

                drift_monitor.update(
                    "M3",
                    profile_for_drift,
                )

            # -------------------------------------------------
            # PROMETHEUS
            # -------------------------------------------------

            PREDICTION_ENTITIES_TOTAL.labels(
                model="M3",
                mode="day",
            ).inc(
                m3["n_entities"]
            )

            result[
                "modeles"
            ][
                "M3"
            ] = {

                "statut": "ok",

                "n_predictions": (
                    m3["n_rows"]
                ),

                "n_enregistrees": (
                    n_saved
                ),

                "n_entities": (
                    m3["n_entities"]
                ),

                "cat_jour": (
                    m3["cat_jour"]
                ),

                "version": (
                    m3["version"]
                ),
            }

            result[
                "nb_predictions_total"
            ] += n_saved

        except Exception as exc:

            PREDICTION_ERRORS_TOTAL.labels(
                model="M3"
            ).inc()

            result[
                "modeles"
            ][
                "M3"
            ] = {

                "statut": "erreur",

                "erreur": str(
                    exc
                ),
            }

            result[
                "erreurs"
            ].append(
                f"M3 : {str(exc)}"
            )

        # =====================================================
        # 11. M4
        # =====================================================

        try:

            MODEL_RUNS_TOTAL.labels(
                model="M4",
                mode="day",
            ).inc()

            m4 = service.predict_profile_day(
                "M4",
                jour_ts,
                jour_exploitation=jour_exploitation,
            )

            # -------------------------------------------------
            # HISTORISATION
            # -------------------------------------------------

            n_saved = (
                history.save_m4_full_day(
                    m4["day"],
                    m4["profile_rows"],
                    m4["version"],
                )
            )

            # -------------------------------------------------
            # MONITORING PSI M4
            # -------------------------------------------------

            profile_rows = (
                m4["profile_rows"]
            )

            if (
                profile_rows is not None
                and not profile_rows.empty
            ):

                profile_for_drift = []

                grouped = (
                    profile_rows
                    .groupby(
                        "heure_debut"
                    )[
                        "pourc_validations_predit"
                    ]
                    .mean()
                    .reset_index()
                )

                for _, row in grouped.iterrows():

                    profile_for_drift.append(
                        {
                            "heure_debut": int(
                                row[
                                    "heure_debut"
                                ]
                            ),
                            "prediction_percent": float(
                                row[
                                    "pourc_validations_predit"
                                ]
                            ),
                        }
                    )

                drift_monitor.update(
                    "M4",
                    profile_for_drift,
                )

            # -------------------------------------------------
            # PROMETHEUS
            # -------------------------------------------------

            PREDICTION_ENTITIES_TOTAL.labels(
                model="M4",
                mode="day",
            ).inc(
                m4["n_entities"]
            )

            result[
                "modeles"
            ][
                "M4"
            ] = {

                "statut": "ok",

                "n_predictions": (
                    m4["n_rows"]
                ),

                "n_enregistrees": (
                    n_saved
                ),

                "n_entities": (
                    m4["n_entities"]
                ),

                "cat_jour": (
                    m4["cat_jour"]
                ),

                "version": (
                    m4["version"]
                ),
            }

            result[
                "nb_predictions_total"
            ] += n_saved

        except Exception as exc:

            PREDICTION_ERRORS_TOTAL.labels(
                model="M4"
            ).inc()

            result[
                "modeles"
            ][
                "M4"
            ] = {

                "statut": "erreur",

                "erreur": str(
                    exc
                ),
            }

            result[
                "erreurs"
            ].append(
                f"M4 : {str(exc)}"
            )

        # =====================================================
        # 12. STATUT GLOBAL
        # =====================================================

        if not result[
            "erreurs"
        ]:

            result[
                "statut"
            ] = "ok"

        elif len(
            result["erreurs"]
        ) < 4:

            result[
                "statut"
            ] = "partiel"

        else:

            result[
                "statut"
            ] = "erreur"

        return result

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )
