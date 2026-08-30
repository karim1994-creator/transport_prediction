from pathlib import Path
import json

import joblib
import numpy as np


# ============================================================
# CHEMINS
# ============================================================

BASE_DIR = Path(
    __file__
).resolve().parent

MODELS_DIR = (
    BASE_DIR
    / "models_a53"
)

OUTPUT_DIR = (
    BASE_DIR
    / "application"
    / "app"
    / "services"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "drift_reference.json"
)


# ============================================================
# MODELES
# ============================================================

MODEL_FILES = {

    "M1":
        "M1_hgb_historique_causal.joblib",

    "M2":
        "M2_hgb_historique_causal.joblib",

    "M3":
        "M3_baseline_historique.joblib",

    "M4":
        "M4_baseline_historique.joblib",
}


# ============================================================
# REFERENCE
# ============================================================

references = {}


# ============================================================
# M3 / M4
# ============================================================

for model_name in [
    "M3",
    "M4",
]:

    path = (
        MODELS_DIR
        / MODEL_FILES[model_name]
    )

    print()
    print(
        f"Chargement {model_name} :",
        path
    )

    artifact = joblib.load(
        path
    )

    baseline = artifact[
        "baseline"
    ].copy()

    hourly = (
        baseline
        .groupby(
            "heure_debut",
            observed=False
        )[
            "pourc_validations"
        ]
        .mean()
    )

    hourly = (
        hourly
        .reindex(
            range(24),
            fill_value=0
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    hourly = (
        hourly
        / hourly.sum()
    )

    references[
        model_name
    ] = hourly.tolist()

    print(
        "Reference construite."
    )


# ============================================================
# M1 / M2
#
# Les modèles M1/M2 produisent des valeurs journalières
# et ne possèdent pas directement un profil 24 heures.
#
# Pour le monitoring du profil horaire, on initialise donc
# une distribution neutre de référence.
# ============================================================

references["M1"] = (
    np.ones(24)
    / 24
).tolist()

references["M2"] = (
    np.ones(24)
    / 24
).tolist()


# ============================================================
# SAUVEGARDE
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        references,
        file,
        indent=2,
    )


print()
print(
    "=" * 80
)

print(
    "REFERENCE DE DERIVE CREEE"
)

print(
    OUTPUT_FILE
)

print(
    "=" * 80
)

for model_name, values in references.items():

    print(
        model_name,
        "somme =",
        round(
            sum(values),
            6
        )
    )