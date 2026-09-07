from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path
from typing import Any

from prometheus_client import Gauge


MODELS = ("M1", "M2", "M3", "M4")
PROFILE_MODELS = ("M3", "M4")
PREDICTION_MODELS = ("M1", "M2")

DRIFT_PSI = Gauge(
    "ml_drift_psi",
    "Score PSI de derive ML",
    ["model"],
)

DRIFT_ALERT = Gauge(
    "ml_drift_alert",
    "Indicateur de derive ML : 1 = derive detectee",
    ["model"],
)

DRIFT_SAMPLES = Gauge(
    "ml_drift_samples",
    "Nombre d'echantillons utilises pour calculer la derive",
    ["model"],
)

EPSILON = 1e-6


# ============================================================
# PSI
# ============================================================

def calculate_psi(
    reference: list[float],
    current: list[float],
) -> float:
    """
    Calcule le PSI entre une distribution de référence
    et une distribution courante.

    Les deux distributions doivent avoir le même nombre
    de classes.
    """

    if not reference or not current:
        return 0.0

    if len(reference) != len(current):
        raise ValueError(
            "Les distributions de reference et courante "
            "doivent avoir la meme longueur."
        )

    reference_sum = sum(reference)
    current_sum = sum(current)

    if reference_sum <= 0 or current_sum <= 0:
        return 0.0

    # Normalisation + epsilon pour éviter log(0)
    ref = [
        max(
            float(value) / reference_sum,
            EPSILON,
        )
        for value in reference
    ]

    cur = [
        max(
            float(value) / current_sum,
            EPSILON,
        )
        for value in current
    ]

    # Renormalisation après application de epsilon
    ref_sum = sum(ref)
    cur_sum = sum(cur)

    ref = [
        value / ref_sum
        for value in ref
    ]

    cur = [
        value / cur_sum
        for value in cur
    ]

    psi = 0.0

    for r, c in zip(ref, cur):
        psi += (c - r) * math.log(c / r)

    return float(psi)


# ============================================================
# OUTILS
# ============================================================

def _normalise(
    values: list[float],
) -> list[float]:
    """
    Transforme une liste de comptes en proportions.
    """

    total = sum(values)

    if total <= 0:
        return [0.0 for _ in values]

    return [
        value / total
        for value in values
    ]


def _prediction_value(
    value: Any,
) -> float | None:
    """
    Convertit une valeur en float valide.

    Les valeurs négatives sont ramenées à 0.
    Les NaN / inf sont ignorés.
    """

    try:
        value = float(value)

        if not math.isfinite(value):
            return None

        return max(value, 0.0)

    except (TypeError, ValueError):
        return None


def calculate_value_distribution(
    values: list[float],
    bin_edges: list[float],
) -> list[float]:
    """
    Transforme les prédictions en distribution par bins.

    Exemple :

        bins = [0, 10, 20, 30]

    donne :

        [0,10[
        [10,20[
        [20,30]

    Les valeurs inférieures au premier bin sont placées
    dans le premier bin.

    Les valeurs supérieures ou égales au dernier bin
    sont placées dans le dernier bin.
    """

    if len(bin_edges) < 2:
        raise ValueError(
            "Il faut au moins deux bornes de bins."
        )

    # Vérification de l'ordre des bins
    for i in range(len(bin_edges) - 1):
        if bin_edges[i] >= bin_edges[i + 1]:
            raise ValueError(
                "Les bornes des bins doivent être "
                "strictement croissantes."
            )

    counts = [
        0
    ] * (len(bin_edges) - 1)

    valid_values = []

    for value in values:

        parsed = _prediction_value(value)

        if parsed is not None:
            valid_values.append(parsed)

    if not valid_values:
        return [
            0.0
        ] * len(counts)

    for value in valid_values:

        # Valeur sous le premier bin
        if value < bin_edges[0]:
            counts[0] += 1
            continue

        # Valeur au-dessus du dernier bin
        if value >= bin_edges[-1]:
            counts[-1] += 1
            continue

        # Recherche du bin
        for i in range(len(bin_edges) - 1):

            lower = bin_edges[i]
            upper = bin_edges[i + 1]

            if lower <= value < upper:
                counts[i] += 1
                break

    return _normalise(counts)


# ============================================================
# DRIFT MONITOR
# ============================================================

class DriftMonitor:

    def __init__(
        self,
        reference_file: str | Path,
        window_size: int = 7,
        threshold: float = 0.20,
    ):
        self.reference_file = Path(
            reference_file
        )

        self.window_size = window_size
        self.threshold = threshold

        self.references: dict[str, Any] = {}

        self.windows: dict[str, deque] = {
            model: deque(
                maxlen=window_size
            )
            for model in MODELS
        }

        # Charge les références avant d'utiliser
        # les métriques.
        self.load_reference()

        # Initialise les séries Prometheus
        # pour les quatre modèles.
        for model in MODELS:

            DRIFT_PSI.labels(
                model=model
            ).set(0)

            DRIFT_ALERT.labels(
                model=model
            ).set(0)

            DRIFT_SAMPLES.labels(
                model=model
            ).set(0)

    # ========================================================
    # REFERENCE
    # ========================================================

    def load_reference(self) -> None:
        """
        Charge et valide drift_reference.json.
        """

        if not self.reference_file.exists():
            raise FileNotFoundError(
                "Fichier de reference PSI introuvable : "
                f"{self.reference_file}"
            )

        with self.reference_file.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError(
                "Le fichier de reference PSI doit contenir "
                "un objet JSON."
            )

        for model in MODELS:

            if model not in data:
                raise ValueError(
                    f"Reference PSI absente pour {model}"
                )

            reference = data[model]

            # ------------------------------------------------
            # M1 / M2
            # ------------------------------------------------

            if model in PREDICTION_MODELS:

                if not isinstance(
                    reference,
                    dict,
                ):
                    raise ValueError(
                        f"La reference de {model} doit etre "
                        "un objet contenant 'bins' "
                        "et 'distribution'."
                    )

                bins = reference.get(
                    "bins"
                )

                distribution = reference.get(
                    "distribution"
                )

                if not isinstance(
                    bins,
                    list,
                ):
                    raise ValueError(
                        f"'bins' invalide pour {model}."
                    )

                if not isinstance(
                    distribution,
                    list,
                ):
                    raise ValueError(
                        f"'distribution' invalide pour {model}."
                    )

                if len(bins) < 2:
                    raise ValueError(
                        f"Pas assez de bins pour {model}."
                    )

                if len(distribution) != len(bins) - 1:
                    raise ValueError(
                        f"Pour {model}, la distribution doit "
                        "contenir exactement len(bins) - 1 "
                        "valeurs."
                    )

                bins = [
                    float(value)
                    for value in bins
                ]

                distribution = [
                    float(value)
                    for value in distribution
                ]

                # Vérification des bornes
                for i in range(len(bins) - 1):

                    if bins[i] >= bins[i + 1]:
                        raise ValueError(
                            f"Les bins de {model} ne sont pas "
                            "strictement croissants."
                        )

                if sum(distribution) <= 0:
                    raise ValueError(
                        f"La distribution de reference "
                        f"{model} est vide."
                    )

                self.references[model] = {
                    **reference,
                    "bins": bins,
                    "distribution": distribution,
                }

            # ------------------------------------------------
            # M3 / M4
            # ------------------------------------------------

            else:

                if not isinstance(
                    reference,
                    list,
                ):
                    raise ValueError(
                        f"La reference de {model} doit "
                        "etre une liste."
                    )

                if len(reference) != 24:
                    raise ValueError(
                        f"La reference de {model} doit "
                        "contenir exactement 24 valeurs "
                        "horaires."
                    )

                self.references[model] = [
                    float(value)
                    for value in reference
                ]

    # ========================================================
    # M3 / M4
    # PSI SUR PROFIL HORAIRE
    # ========================================================

    def update(
        self,
        model: str,
        profile: list[dict[str, Any]],
    ) -> None:
        """
        Met à jour le drift pour M3/M4.

        Le profil doit contenir une heure et une prédiction
        pour chaque ligne.
        """

        if model not in MODELS:
            raise ValueError(
                f"Modele inconnu : {model}"
            )

        if model in PREDICTION_MODELS:
            raise ValueError(
                "update() est reserve a M3/M4. "
                "Utiliser update_from_predictions() "
                "pour M1/M2."
            )

        hourly_values = [
            0.0
        ] * 24

        for row in profile:

            hour = row.get(
                "heure_debut",
                row.get(
                    "hour",
                    row.get(
                        "heure"
                    ),
                ),
            )

            value = row.get(
                "prediction_percent",
                row.get(
                    "prediction",
                    row.get(
                        "value"
                    ),
                ),
            )

            try:
                hour = int(hour)
                value = float(value)

            except (TypeError, ValueError):
                continue

            if 0 <= hour < 24:
                hourly_values[hour] += max(
                    value,
                    0.0,
                )

        hourly_values = _normalise(
            hourly_values
        )

        self._update_from_array(
            model=model,
            current_distribution=hourly_values,
        )

    # ========================================================
    # M1 / M2
    # PSI SUR DISTRIBUTION DES PREDICTIONS QUOTIDIENNES
    # ========================================================

    def update_from_predictions(
        self,
        model: str,
        predictions: list[float],
    ) -> None:
        """
        Met à jour le drift M1/M2 à partir des prédictions
        quotidiennes.

        Contrairement à M3/M4, aucune notion d'heure n'est
        utilisée ici.

        Les prédictions sont réparties dans les bins définis
        dans drift_reference.json.
        """

        if model not in PREDICTION_MODELS:
            raise ValueError(
                "update_from_predictions() est reserve "
                "a M1/M2."
            )

        reference = self.references.get(
            model
        )

        if not isinstance(
            reference,
            dict,
        ):
            raise ValueError(
                f"Reference invalide pour {model}."
            )

        bins = reference.get(
            "bins"
        )

        reference_distribution = reference.get(
            "distribution"
        )

        if not bins:
            raise ValueError(
                f"Bins absents pour {model}."
            )

        if not reference_distribution:
            raise ValueError(
                f"Distribution absente pour {model}."
            )

        current_distribution = calculate_value_distribution(
            values=predictions,
            bin_edges=bins,
        )

        if len(current_distribution) != len(
            reference_distribution
        ):
            raise ValueError(
                f"Nombre de bins incompatible pour {model}."
            )

        # Si aucune prédiction valide n'est disponible,
        # on ne modifie pas le monitoring.
        if sum(current_distribution) <= 0:
            return

        self._update_from_array(
            model=model,
            current_distribution=current_distribution,
        )

    # ========================================================
    # CALCUL COMMUN
    # ========================================================

    def _update_from_array(
        self,
        model: str,
        current_distribution: list[float],
    ) -> None:
        """
        Ajoute une distribution à la fenêtre glissante
        puis recalcule le PSI.
        """

        if model not in MODELS:
            raise ValueError(
                f"Modele inconnu : {model}"
            )

        self.windows[model].append(
            current_distribution
        )

        if not self.windows[model]:
            return

        n_bins = len(
            current_distribution
        )

        mean_distribution = [
            0.0
        ] * n_bins

        for distribution in self.windows[model]:

            if len(distribution) != n_bins:
                raise ValueError(
                    f"Nombre de bins incoherent pour {model}."
                )

            for i, value in enumerate(
                distribution
            ):
                mean_distribution[i] += value

        count = len(
            self.windows[model]
        )

        mean_distribution = [
            value / count
            for value in mean_distribution
        ]

        reference = self.references[
            model
        ]

        # M1 / M2 : référence = dictionnaire
        if isinstance(
            reference,
            dict,
        ):

            reference_distribution = reference[
                "distribution"
            ]

        # M3 / M4 : référence = liste de 24 valeurs
        else:

            reference_distribution = reference

        psi = calculate_psi(
            reference=[
                float(value)
                for value in reference_distribution
            ],
            current=mean_distribution,
        )

        alert = (
            1
            if psi >= self.threshold
            else 0
        )

        DRIFT_PSI.labels(
            model=model
        ).set(psi)

        DRIFT_ALERT.labels(
            model=model
        ).set(alert)

        DRIFT_SAMPLES.labels(
            model=model
        ).set(count)

    # ========================================================
    # STATUS
    # ========================================================

    def status(self) -> dict[str, Any]:
        """
        Retourne l'état du monitoring pour les quatre modèles.
        """

        result = {}

        for model in MODELS:

            result[model] = {
                "samples": len(
                    self.windows[model]
                ),
                "threshold": self.threshold,
                "reference_loaded": (
                    model in self.references
                ),
                "psi": float(
                    DRIFT_PSI.labels(
                        model=model
                    )._value.get()
                ),
                "alert": bool(
                    DRIFT_ALERT.labels(
                        model=model
                    )._value.get()
                ),
            }

        return result

    # ========================================================
    # METRICS
    # ========================================================

    def get_psi(
        self,
        model: str,
    ) -> float:

        return float(
            DRIFT_PSI.labels(
                model=model
            )._value.get()
        )

    def get_alert(
        self,
        model: str,
    ) -> bool:

        return bool(
            DRIFT_ALERT.labels(
                model=model
            )._value.get()
        )

    def get_samples(
        self,
        model: str,
    ) -> int:

        return int(
            DRIFT_SAMPLES.labels(
                model=model
            )._value.get()
        )
