# ============================================================
# TESTS DES ENDPOINTS M3 / M4
# ============================================================

from fastapi.testclient import TestClient

from app.main import app


# ============================================================
# M3
# ============================================================

def test_predict_m3():

    payload = {
        "profile": {
            "code_arret": "810_801_594",
            "id_zdc": "71590",
            "cat_jour": "JOHV",
        }
    }

    with TestClient(app) as client:

        response = client.post(
            "/predict/m3",
            json=payload
        )

    assert response.status_code == 200

    data = response.json()

    assert data["model"] == "M3"

    assert data["target"] == "pourc_validations"

    assert data["strategy"] == "baseline_historique"

    assert data["profile_keys"] == [
        "code_arret",
        "id_zdc",
        "cat_jour",
    ]

    assert len(data["profile"]) == 24

    total = data["total_percent"]

    assert abs(
        total - 100.0
    ) < 1e-6


# ============================================================
# M4
# ============================================================

def test_predict_m4():

    payload = {
        "profile": {
            "code_ligne": "534_534_524",
            "id_groupofligne": "A00553",
            "cat_jour": "JOHV",
        }
    }

    with TestClient(app) as client:

        response = client.post(
            "/predict/m4",
            json=payload
        )

    assert response.status_code == 200

    data = response.json()

    assert data["model"] == "M4"

    assert data["target"] == "pourc_validations"

    assert data["strategy"] == "baseline_historique"

    assert data["profile_keys"] == [
        "code_ligne",
        "id_groupofligne",
        "cat_jour",
    ]

    assert len(data["profile"]) == 24

    total = data["total_percent"]

    assert abs(
        total - 100.0
    ) < 1e-6


# ============================================================
# STRUCTURE M3
# ============================================================

def test_m3_profile_structure():

    payload = {
        "profile": {
            "code_arret": "810_801_594",
            "id_zdc": "71590",
            "cat_jour": "JOHV",
        }
    }

    with TestClient(app) as client:

        response = client.post(
            "/predict/m3",
            json=payload
        )

    assert response.status_code == 200

    data = response.json()

    assert len(data["profile"]) == 24

    for item in data["profile"]:

        assert "heure_debut" in item

        assert "prediction_percent" in item

        assert (
            0
            <= item["heure_debut"]
            <= 23
        )

        assert (
            item["prediction_percent"]
            >= 0
        )


# ============================================================
# STRUCTURE M4
# ============================================================

def test_m4_profile_structure():

    payload = {
        "profile": {
            "code_ligne": "534_534_524",
            "id_groupofligne": "A00553",
            "cat_jour": "JOHV",
        }
    }

    with TestClient(app) as client:

        response = client.post(
            "/predict/m4",
            json=payload
        )

    assert response.status_code == 200

    data = response.json()

    assert len(data["profile"]) == 24

    for item in data["profile"]:

        assert "heure_debut" in item

        assert "prediction_percent" in item

        assert (
            0
            <= item["heure_debut"]
            <= 23
        )

        assert (
            item["prediction_percent"]
            >= 0
        )