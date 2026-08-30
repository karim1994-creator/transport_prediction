from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from app.api.routes import router
from app.services.model_loader import ModelLoader
from app.services.prediction_engine import PredictionEngine
from app.services.drift_monitor import DriftMonitor


@asynccontextmanager
async def lifespan(app):

    loader = ModelLoader()

    loader.load_all()

    app.state.model_loader = loader

    app.state.prediction_engine = (
        PredictionEngine(loader)
    )

    reference_file = (
        Path(__file__).resolve().parent
        / "services"
        / "drift_reference.json"
    )

    app.state.drift_monitor = DriftMonitor(
        reference_file
    )

    yield

    app.state.model_loader = None

    app.state.prediction_engine = None

    app.state.drift_monitor = None


app = FastAPI(
    title="Transport Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)

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