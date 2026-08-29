# Étape 1 — API locale

Cette étape charge les 4 artefacts `.joblib` et expose une API FastAPI.

## Arborescence à intégrer dans BLOC 5

Copier `app/` et `tests/` dans `application/` de ton projet, puis conserver `models_a53/` à la racine.

## Lancement

Depuis `BLOC 5/application/` :

```bash
pip install -r ../requirements.txt
uvicorn app.main:app --reload
```

Documentation : http://127.0.0.1:8000/docs

Endpoints :
- GET /health
- GET /models
- POST /predict/m1
- POST /predict/m2
- POST /predict/m3
- POST /predict/m4

M3 et M4 retournent les 24 heures et un total de 100 %.
M1 et M2 attendent exactement les variables enregistrées dans leurs artefacts.
