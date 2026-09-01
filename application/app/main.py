from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from app.api.routes import router
from app.config import DATABASE_URL
from app.services.model_loader import ModelLoader
from app.services.prediction_engine import PredictionEngine
from app.services.drift_monitor import DriftMonitor
from app.services.prediction_history import PredictionHistory


# ============================================================
# LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    # ========================================================
    # CHARGEMENT DES MODELES
    # ========================================================

    loader = ModelLoader()

    loader.load_all()

    app.state.model_loader = loader

    # ========================================================
    # MOTEUR DE PREDICTION
    # ========================================================

    app.state.prediction_engine = PredictionEngine(
        loader
    )

    # ========================================================
    # MONITORING DE DERIVE - PSI
    # ========================================================

    reference_file = (
        Path(__file__).resolve().parent
        / "services"
        / "drift_reference.json"
    )

    app.state.drift_monitor = DriftMonitor(
        reference_file
    )

    # ========================================================
    # HISTORISATION DES PREDICTIONS
    # ========================================================

    app.state.prediction_history = PredictionHistory(
        DATABASE_URL
    )

    # ========================================================
    # APPLICATION ACTIVE
    # ========================================================

    yield

    # ========================================================
    # NETTOYAGE
    # ========================================================

    app.state.model_loader = None
    app.state.prediction_engine = None
    app.state.drift_monitor = None
    app.state.prediction_history = None


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Transport Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# ROUTES API
# ============================================================

app.include_router(
    router
)


# ============================================================
# PROMETHEUS
# ============================================================

Instrumentator().instrument(
    app
).expose(
    app,
    endpoint="/metrics"
)