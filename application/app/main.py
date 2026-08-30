from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from app.api.routes import router
from app.services.model_loader import ModelLoader
from app.services.prediction_engine import PredictionEngine


@asynccontextmanager
async def lifespan(app):
    loader = ModelLoader()
    loader.load_all()

    app.state.model_loader = loader
    app.state.prediction_engine = PredictionEngine(loader)

    yield

    app.state.model_loader = None
    app.state.prediction_engine = None


app = FastAPI(
    title="Transport Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)


# ============================================================
# MONITORING PROMETHEUS
# ============================================================

Instrumentator().instrument(app).expose(
    app,
    endpoint="/metrics"
)