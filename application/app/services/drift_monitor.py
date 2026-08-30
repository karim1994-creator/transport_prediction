from collections import deque
from pathlib import Path
import json

import numpy as np
from prometheus_client import Gauge


# ============================================================
# PARAMETRES
# ============================================================

PSI_ALERT_THRESHOLD = 0.20

WINDOW_SIZE = 20

EPSILON = 1e-6


# ============================================================
# METRIQUES PROMETHEUS
# ============================================================

DRIFT_PSI = Gauge(
    "ml_drift_psi",
    "Score PSI de derive du profil horaire",
    ["model"],
)


DRIFT_ALERT = Gauge(
    "ml_drift_alert",
    "Indicateur de derive ML : 1 = derive detectee",
    ["model"],
)


DRIFT_SAMPLES = Gauge(
    "ml_drift_samples",
    "Nombre de profils utilises pour calculer la derive",
    ["model"],
)


# ============================================================
# CALCUL PSI
# ============================================================

def calculate_psi(reference, current):
    """
    Calcule le Population Stability Index (PSI).

    reference : distribution historique
    current   : distribution courante

    Les deux vecteurs doivent être de taille 24.
    """

    reference = np.asarray(
        reference,
        dtype=np.float64,
    )

    current = np.asarray(
        current,
        dtype=np.float64,
    )

    if reference.size != current.size:
        raise ValueError(
            "Les distributions doivent avoir la même taille."
        )

    reference = np.clip(
        reference,
        EPSILON,
        None,
    )

    current = np.clip(
        current,
        EPSILON,
        None,
    )

    reference = (
        reference
        / reference.sum()
    )

    current = (
        current
        / current.sum()
    )

    psi = np.sum(
        (
            current - reference
        )
        * np.log(
            current / reference
        )
    )

    return float(psi)


# ============================================================
# MONITORING
# ============================================================

class DriftMonitor:

    def __init__(
        self,
        reference_file,
        window_size=WINDOW_SIZE,
        threshold=PSI_ALERT_THRESHOLD,
    ):

        self.reference_file = Path(
            reference_file
        )

        self.window_size = window_size

        self.threshold = threshold

        self.references = {}

        self.windows = {
            "M1": deque(
                maxlen=window_size
            ),
            "M2": deque(
                maxlen=window_size
            ),
            "M3": deque(
                maxlen=window_size
            ),
            "M4": deque(
                maxlen=window_size
            ),
        }

        self.load_reference()

    # ========================================================
    # REFERENCE
    # ========================================================

    def load_reference(self):

        if not self.reference_file.exists():

            raise FileNotFoundError(
                f"Référence de dérive absente : "
                f"{self.reference_file}"
            )

        with open(
            self.reference_file,
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        for model in [
            "M1",
            "M2",
            "M3",
            "M4",
        ]:

            if model not in data:

                continue

            reference = np.asarray(
                data[model],
                dtype=np.float64,
            )

            if reference.size != 24:

                raise ValueError(
                    f"Référence {model} invalide : "
                    f"{reference.size} valeurs."
                )

            reference = (
                reference
                / reference.sum()
            )

            self.references[
                model
            ] = reference

    # ========================================================
    # AJOUT D'UNE PREDICTION
    # ========================================================

    def update(
        self,
        model,
        profile,
    ):

        if model not in self.references:

            return

        values = np.zeros(
            24,
            dtype=np.float64,
        )

        for item in profile:

            hour = int(
                item["heure_debut"]
            )

            prediction = float(
                item["prediction_percent"]
            )

            if 0 <= hour <= 23:

                values[hour] = max(
                    prediction,
                    0.0,
                )

        total = values.sum()

        if total <= 0:

            return

        values = (
            values
            / total
        )

        window = self.windows[
            model
        ]

        window.append(
            values
        )

        current = np.mean(
            np.vstack(
                window
            ),
            axis=0,
        )

        reference = self.references[
            model
        ]

        psi = calculate_psi(
            reference,
            current,
        )

        alert = int(
            psi >= self.threshold
        )

        DRIFT_PSI.labels(
            model=model
        ).set(
            psi
        )

        DRIFT_ALERT.labels(
            model=model
        ).set(
            alert
        )

        DRIFT_SAMPLES.labels(
            model=model
        ).set(
            len(window)
        )

    # ========================================================
    # ETAT
    # ========================================================

    def status(self):

        result = {}

        for model in [
            "M1",
            "M2",
            "M3",
            "M4",
        ]:

            result[model] = {

                "samples":
                    len(
                        self.windows[
                            model
                        ]
                    ),

                "threshold":
                    self.threshold,
            }

        return result