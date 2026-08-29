# ============================================================
# TEST HEALTH API
# ============================================================

from fastapi.testclient import TestClient

from app.main import app


def test_health():

    # Le contexte lifespan de FastAPI doit être exécuté
    # pour initialiser le model_loader.
    with TestClient(app) as client:

        response = client.get(
            "/health"
        )

        assert response.status_code == 200

        data = response.json()

        assert data["status"] == "ok"

        assert data["models_loaded"] == 4