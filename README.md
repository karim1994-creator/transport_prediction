# Transport Prediction

Projet de prédiction dans le domaine du transport avec une architecture MLOps.

L'objectif de ce projet est de mettre en place une chaîne complète, depuis la récupération et la préparation des données jusqu'au déploiement du modèle, son monitoring et son réentraînement lorsqu'une dérive des données est détectée.

## Présentation du projet

Dans ce projet, j'utilise plusieurs sources de données pour construire mes modèles de prédiction :

* PostgreSQL
* Données météo
* Calendrier

Après la préparation des données, j'entraîne plusieurs modèles (**M1, M2, M3 et M4**) afin de comparer leurs performances et de retenir le modèle le plus adapté.

Les modèles sont sauvegardés avec **Joblib** et suivis avec **MLflow**.

L'application est ensuite exposée avec **FastAPI** et exécutée dans un environnement **Docker**. Une interface **Streamlit** permet ensuite à l'utilisateur de faire des prédictions.

---

## Architecture du projet

Voici l'architecture générale que j'ai mise en place :

```text
Sources de données
│
├── PostgreSQL
├── Données météo
└── Calendrier
        │
        ↓
Préparation des données
        │
        ↓
Modèles M1 – M2 – M3 – M4
        │
        ↓
Joblib / MLflow
        │
        ↓
FastAPI
        │
        ↓
Docker
        │
        ↓
Streamlit
        │
        ↓
Prédictions
        │
        ↓
PostgreSQL
        │
        ↓
Prometheus
        │
        ↓
Grafana
```

En parallèle, j'ai mis en place une partie permettant de surveiller les nouvelles données et de gérer le réentraînement :

```text
Nouvelles données
        ↓
Data Drift
        ↓
Évaluation
        ↓
Réentraînement candidat
        ↓
Validation
        ↓
Nouvelle version du modèle
```

---

## Fonctionnement de la prédiction

Le parcours d'une prédiction est le suivant :

```text
Utilisateur
     ↓
Streamlit
     ↓
FastAPI
     ↓
Modèle M1 / M2 / M3 / M4
     ↓
Résultat de prédiction
     ↓
Archivage dans PostgreSQL
```

L'utilisateur renseigne les informations nécessaires depuis l'interface Streamlit.

Streamlit envoie ensuite les données à l'API FastAPI. L'API charge le modèle sélectionné et retourne le résultat de la prédiction.

Les résultats peuvent ensuite être enregistrés dans PostgreSQL afin de garder un historique des prédictions.

---

## Les modèles

J'ai développé plusieurs modèles de Machine Learning :

```text
M1
M2
M3
M4
```

L'idée est de pouvoir entraîner plusieurs modèles et comparer leurs performances.

Après l'entraînement, les résultats sont évalués afin de déterminer quel modèle doit être utilisé comme modèle de référence.

Les modèles peuvent être sauvegardés sous forme de fichiers **Joblib** et enregistrés dans **MLflow**.

---

## MLflow

J'utilise MLflow pour garder une trace des différents entraînements.

Pour chaque entraînement, je peux enregistrer :

* les paramètres du modèle ;
* les métriques ;
* les artefacts ;
* les différentes versions des modèles.

Cela me permet de comparer les modèles et de garder une trace des versions utilisées.

---

## Docker

J'utilise Docker pour avoir un environnement reproductible pour l'application.

L'image Docker contient notamment :

```text
Image Docker
    ↓
Application FastAPI
    ↓
Dépendances Python
    ↓
Modèles M1 – M4
    ↓
Configuration
```

Cela permet de lancer l'application dans un environnement indépendant de la machine utilisée.

Le projet contient également un `docker-compose.yml` pour faciliter le lancement des différents services.

---

## Monitoring

Pour le monitoring, j'utilise :

* **Prometheus**
* **Grafana**

Le fonctionnement est le suivant :

```text
FastAPI
   ↓
Métriques
   ↓
Prometheus
   ↓
Grafana
```

Prometheus récupère les métriques de l'application et Grafana permet de les visualiser sous forme de dashboards.

Cela me permet notamment de suivre le fonctionnement de l'API et l'activité de l'application.

---

## Data Drift

Une autre partie du projet concerne la détection de dérive des données.

L'idée est de comparer les nouvelles données avec les données de référence.

```text
Données de référence
        ↓
Nouvelles données
        ↓
Calcul du PSI
        ↓
Comparaison avec un seuil
        ↓
Détection éventuelle d'une dérive
```

J'utilise le **PSI (Population Stability Index)** pour comparer les distributions.

Si une dérive importante est détectée, cela peut conduire à un nouveau cycle d'entraînement.

---

## Réentraînement du modèle

Lorsqu'une dérive est détectée, je peux lancer un processus de réentraînement candidat.

```text
Nouvelles données
        ↓
Collecte
        ↓
Contrôle des données
        ↓
Réentraînement candidat
        ↓
Évaluation
        ↓
Comparaison avec le modèle de référence
        ↓
Décision
```

Le nouveau modèle n'est pas automatiquement utilisé.

Je compare ses performances avec celles du modèle de référence.

```text
             Évaluation
                 ↓
        ┌────────┴────────┐
        ↓                 ↓
Performance          Performance
 suffisante          insuffisante
        ↓                 ↓
Nouveau modèle       Conserver
                     le modèle
                     de référence
```

Cela permet d'éviter de remplacer un modèle existant par un modèle moins performant.

---

## GitHub Actions

J'utilise également **GitHub Actions** pour automatiser certaines étapes du projet.

Le workflow permet notamment d'effectuer des vérifications et des tests lors des modifications du projet.

Le fichier se trouve dans :

```text
.github/
└── workflows/
    └── ci.yml
```

L'idée est d'avoir une chaîne automatisée :

```text
Modification du projet
        ↓
GitHub Actions
        ↓
Tests
        ↓
Évaluation
        ↓
Validation
        ↓
Build Docker
        ↓
Déploiement
```

---

## Cycle MLOps complet

Au final, le cycle que j'ai mis en place peut être résumé comme ceci :

```text
Collecte des données
        ↓
Préparation
        ↓
Entraînement
        ↓
Évaluation
        ↓
Sauvegarde
        ↓
Validation
        ↓
Déploiement
        ↓
Prédiction
        ↓
Historisation
        ↓
Monitoring
        ↓
Détection de Drift
        ↓
Nouvelles données
        ↓
Réentraînement
        ↓
Nouvelle version
```

Le but est donc de ne pas avoir uniquement un modèle qui fait des prédictions, mais une chaîne complète permettant de suivre le modèle dans le temps.

---

## Technologies utilisées

| Technologie    | Utilisation                       |
| -------------- | --------------------------------- |
| Python         | Développement et Machine Learning |
| PostgreSQL     | Stockage des données              |
| Joblib         | Sauvegarde des modèles            |
| MLflow         | Suivi et gestion des modèles      |
| FastAPI        | API de prédiction                 |
| Docker         | Conteneurisation                  |
| Streamlit      | Interface utilisateur             |
| Prometheus     | Monitoring                        |
| Grafana        | Visualisation des métriques       |
| GitHub Actions | CI/CD                             |
| PSI            | Détection du Data Drift           |

---

## Structure du projet

```text
transport_prediction/
│
├── .github/
│   └── workflows/
│
├── application/
│
├── models_a53/
│
├── mlflow_registry/
│
├── monitoring/
│
├── sql/
│
├── tests/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-ci.txt
│
├── train_models_ML.py
├── ml_lifecycle_m1_m2.py
├── evaluate_models_ci.py
├── evaluate_baseline_performance.py
├── evaluate_candidate_performance.py
├── compare_model_performance.py
├── model_promotion_check.py
├── register_models_mlflow.py
│
├── create_drift_reference.py
├── create_reference_test_set.py
└── check_new_training_data.py
```

---

## Installation

Pour récupérer le projet :

```bash
git clone https://github.com/karim1994-creator/transport_prediction.git

cd transport_prediction
```

Créer un environnement virtuel :

```bash
python -m venv .venv
```

Activer l'environnement :

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Installer les dépendances :

```bash
pip install -r requirements.txt
```

---

## Lancement avec Docker

Pour construire l'image :

```bash
docker build -t transport-prediction .
```

Puis lancer l'application :

```bash
docker run -p 8000:8000 transport-prediction
```

Ou avec Docker Compose :

```bash
docker compose up --build
```

---

## Tests

Les tests sont disponibles dans le dossier :

```text
tests/
```

Ils peuvent également être exécutés dans le workflow GitHub Actions.

---

## Résumé

Dans ce projet, j'ai essayé de mettre en place une architecture MLOps complète autour d'un problème de prédiction dans le transport.

Le projet couvre principalement :

```text
Données
   ↓
Machine Learning
   ↓
MLflow
   ↓
FastAPI
   ↓
Docker
   ↓
Streamlit
   ↓
Prédictions
   ↓
Monitoring
   ↓
Data Drift
   ↓
Réentraînement
```

L'objectif est d'avoir un système capable d'évoluer avec les nouvelles données tout en gardant un contrôle sur les performances des modèles.
