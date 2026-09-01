from fastapi import APIRouter, HTTPException, Request

from prometheus_client import Counter

from app.schemas.prediction import (
    MLPredictionRequest,
    ProfilePredictionRequest,
)


router = APIRouter()


# ============================================================
# METRIQUES ML
# ============================================================

PREDICTIONS_TOTAL = Counter(
    "ml_predictions_total",
    "Nombre total de predictions par modele",
    ["model"],
)


PREDICTION_ERRORS_TOTAL = Counter(
    "ml_prediction_errors_total",
    "Nombre total d'erreurs de prediction par modele",
    ["model"],
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

        result = request.app.state.prediction_engine.predict_ml(
            "M1",
            body.rows,
        )

        PREDICTIONS_TOTAL.labels(
            model="M1"
        ).inc()

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

        result = request.app.state.prediction_engine.predict_ml(
            "M2",
            body.rows,
        )

        PREDICTIONS_TOTAL.labels(
            model="M2"
        ).inc()

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

        # ====================================================
        # PREDICTION
        # ====================================================

        result = request.app.state.prediction_engine.predict_full_profile(
            "M3",
            body.profile,
        )

        # ====================================================
        # MONITORING DE DERIVE - PSI
        # ====================================================

        request.app.state.drift_monitor.update(
            "M3",
            result["profile"],
        )

        # ====================================================
        # HISTORISATION DE LA PREDICTION
        # ====================================================

        request.app.state.prediction_history.save_prediction(
            model_name="M3",
            model_version="production",
            perimeter="Ferre",
            request_profile=body.profile,
            prediction=result["profile"],
            prediction_total=result["total_percent"],
        )

        # ====================================================
        # METRIQUE PROMETHEUS
        # ====================================================

        PREDICTIONS_TOTAL.labels(
            model="M3"
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

        # ====================================================
        # PREDICTION
        # ====================================================

        result = request.app.state.prediction_engine.predict_full_profile(
            "M4",
            body.profile,
        )

        # ====================================================
        # MONITORING DE DERIVE - PSI
        # ====================================================

        request.app.state.drift_monitor.update(
            "M4",
            result["profile"],
        )

        # ====================================================
        # HISTORISATION DE LA PREDICTION
        # ====================================================

        request.app.state.prediction_history.save_prediction(
            model_name="M4",
            model_version="production",
            perimeter="Surface",
            request_profile=body.profile,
            prediction=result["profile"],
            prediction_total=result["total_percent"],
        )

        # ====================================================
        # METRIQUE PROMETHEUS
        # ====================================================

        PREDICTIONS_TOTAL.labels(
            model="M4"
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