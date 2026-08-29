from fastapi import APIRouter, HTTPException, Request
from app.schemas.prediction import MLPredictionRequest, ProfilePredictionRequest

router = APIRouter()

@router.get("/health")
def health(request: Request):
    loaded = len(request.app.state.model_loader.models)
    return {"status": "ok" if loaded == 4 else "degraded", "models_loaded": loaded}

@router.get("/models")
def models(request: Request):
    return request.app.state.model_loader.status()

@router.post("/predict/m1")
def predict_m1(body: MLPredictionRequest, request: Request):
    try:
        return request.app.state.prediction_engine.predict_ml("M1", body.rows)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@router.post("/predict/m2")
def predict_m2(body: MLPredictionRequest, request: Request):
    try:
        return request.app.state.prediction_engine.predict_ml("M2", body.rows)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@router.post("/predict/m3")
def predict_m3(body: ProfilePredictionRequest, request: Request):
    try:
        return request.app.state.prediction_engine.predict_full_profile("M3", body.profile)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@router.post("/predict/m4")
def predict_m4(body: ProfilePredictionRequest, request: Request):
    try:
        return request.app.state.prediction_engine.predict_full_profile("M4", body.profile)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
