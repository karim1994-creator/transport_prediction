import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text


# ============================================================
# CONFIGURATION POSTGRESQL
# ============================================================

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres123@localhost:5434/idf_transport_prediction",
)

OUTPUT_FILE = Path("drift_reference.json")

# Nombre de classes utilisées pour le PSI M1/M2.
N_BINS = 10


# ============================================================
# REFERENCES EXISTANTES M3 / M4
# ============================================================

M3_REFERENCE = [
    0.007726907179392423,
    0.0020398984969140594,
    0.0001998512644870793,
    0.00011611633070887336,
    0.0019040320925371743,
    0.015441957802776558,
    0.0362157942989464,
    0.06859389822198224,
    0.0766939787596204,
    0.05709443602682552,
    0.05091153174707913,
    0.054316683134245786,
    0.058402204856214945,
    0.059240560879200266,
    0.06082974469589758,
    0.0648187784183382,
    0.07590315863321526,
    0.08562321713140818,
    0.07675874278282754,
    0.05705420050718026,
    0.036055999013737586,
    0.022542877329532175,
    0.017438758713998068,
    0.014076671682934307,
]


M4_REFERENCE = [
    0.014881921891292938,
    0.01011546842366392,
    0.010104341428663469,
    0.010560222782324019,
    0.01230758061409083,
    0.022778239073683555,
    0.05632593111262494,
    0.11503593355687432,
    0.07756708085037943,
    0.04704886876843256,
    0.03955653875217736,
    0.046543182966076506,
    0.06084426832736999,
    0.05346072155535854,
    0.047678675282107126,
    0.05934975101259544,
    0.07898404060197321,
    0.08013398108378544,
    0.06012983752388484,
    0.03888537376790651,
    0.023598732765445992,
    0.01628266560335478,
    0.011636682658626512,
    0.006189959597307756,
]


# ============================================================
# OUTILS
# ============================================================

def normalise(values: np.ndarray) -> list[float]:
    """
    Transforme des comptes en proportions.
    """

    values = np.asarray(values, dtype=float)

    total = values.sum()

    if total <= 0:
        return [0.0 for _ in values]

    return (values / total).tolist()


def make_unique_bins(
    values: np.ndarray,
    n_bins: int = N_BINS,
) -> list[float]:
    """
    Construit les bornes des bins à partir des quantiles.

    Les doublons sont supprimés pour éviter des intervalles
    de largeur nulle.
    """

    values = np.asarray(values, dtype=float)

    values = values[np.isfinite(values)]
    values = values[values >= 0]

    if len(values) == 0:
        raise ValueError(
            "Aucune valeur valide disponible."
        )

    quantiles = np.linspace(
        0,
        1,
        n_bins + 1,
    )

    bins = np.quantile(
        values,
        quantiles,
    )

    bins = np.unique(bins)

    # Si toutes les valeurs sont identiques,
    # on crée artificiellement un intervalle.
    if len(bins) == 1:

        value = float(bins[0])

        if value == 0:
            bins = np.array(
                [0.0, 1.0]
            )
        else:
            bins = np.array(
                [
                    0.0,
                    value,
                    value + 1.0,
                ]
            )

    return [
        float(value)
        for value in bins
    ]


def distribution_from_bins(
    values: np.ndarray,
    bins: list[float],
) -> list[float]:
    """
    Calcule la distribution historique correspondant
    exactement aux bins utilisés par le PSI.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[np.isfinite(values)]
    values = values[values >= 0]

    if len(values) == 0:
        raise ValueError(
            "Aucune prédiction valide."
        )

    counts = np.zeros(
        len(bins) - 1,
        dtype=int,
    )

    for value in values:

        if value < bins[0]:
            counts[0] += 1
            continue

        if value >= bins[-1]:
            counts[-1] += 1
            continue

        index = np.searchsorted(
            bins,
            value,
            side="right",
        ) - 1

        if index < 0:
            index = 0

        if index >= len(counts):
            index = len(counts) - 1

        counts[index] += 1

    return normalise(counts)


# ============================================================
# M1
# ============================================================

def load_m1_predictions(
    engine,
) -> np.ndarray:
    """
    Récupère les prédictions historiques M1.
    """

    query = text("""
        SELECT
            nb_vald_predit
        FROM prediction.frequentation_journaliere_ferre
        WHERE nb_vald_predit IS NOT NULL
          AND nb_vald_predit >= 0
    """)

    df = pd.read_sql(
        query,
        engine,
    )

    if df.empty:
        raise RuntimeError(
            "Aucune prédiction historique M1 trouvée "
            "dans prediction.frequentation_journaliere_ferre."
        )

    values = pd.to_numeric(
        df["nb_vald_predit"],
        errors="coerce",
    ).dropna()

    values = values[values >= 0]

    if values.empty:
        raise RuntimeError(
            "Aucune valeur M1 valide."
        )

    return values.to_numpy(
        dtype=float
    )


def generate_m1_reference(
    engine,
) -> dict:
    """
    Génère la référence PSI M1.
    """

    values = load_m1_predictions(
        engine
    )

    bins = make_unique_bins(
        values,
        N_BINS,
    )

    distribution = distribution_from_bins(
        values,
        bins,
    )

    return {
        "bins": bins,
        "distribution": distribution,
        "n_observations": int(len(values)),
    }


# ============================================================
# M2
# ============================================================

def load_m2_predictions(
    engine,
) -> np.ndarray:
    """
    Récupère les prédictions historiques M2.
    """

    query = text("""
        SELECT
            nb_vald_predit
        FROM prediction.frequentation_journaliere_surface
        WHERE nb_vald_predit IS NOT NULL
          AND nb_vald_predit >= 0
    """)

    df = pd.read_sql(
        query,
        engine,
    )

    if df.empty:
        raise RuntimeError(
            "Aucune prédiction historique M2 trouvée "
            "dans prediction.frequentation_journaliere_surface."
        )

    values = pd.to_numeric(
        df["nb_vald_predit"],
        errors="coerce",
    ).dropna()

    values = values[values >= 0]

    if values.empty:
        raise RuntimeError(
            "Aucune valeur M2 valide."
        )

    return values.to_numpy(
        dtype=float
    )


def generate_m2_reference(
    engine,
) -> dict:
    """
    Génère la référence PSI M2.
    """

    values = load_m2_predictions(
        engine
    )

    bins = make_unique_bins(
        values,
        N_BINS,
    )

    distribution = distribution_from_bins(
        values,
        bins,
    )

    return {
        "bins": bins,
        "distribution": distribution,
        "n_observations": int(len(values)),
    }


# ============================================================
# GENERATION COMPLETE
# ============================================================

def generate_reference() -> dict:

    print("=" * 60)
    print("GENERATION DU FICHIER DE REFERENCE PSI")
    print("=" * 60)

    print()
    print("Connexion à PostgreSQL...")

    engine = create_engine(
        DATABASE_URL
    )

    # Test connexion
    with engine.begin() as conn:
        conn.execute(
            text("SELECT 1")
        )

    print("Connexion OK.")

    # --------------------------------------------------------
    # M1
    # --------------------------------------------------------

    print()
    print("Génération référence M1...")

    m1_reference = generate_m1_reference(
        engine
    )

    print(
        f"M1 : "
        f"{m1_reference['n_observations']} observations"
    )

    print(
        f"M1 : "
        f"{len(m1_reference['distribution'])} bins"
    )

    # --------------------------------------------------------
    # M2
    # --------------------------------------------------------

    print()
    print("Génération référence M2...")

    m2_reference = generate_m2_reference(
        engine
    )

    print(
        f"M2 : "
        f"{m2_reference['n_observations']} observations"
    )

    print(
        f"M2 : "
        f"{len(m2_reference['distribution'])} bins"
    )

    # --------------------------------------------------------
    # RESULTAT
    # --------------------------------------------------------

    reference = {
        "M1": m1_reference,
        "M2": m2_reference,
        "M3": M3_REFERENCE,
        "M4": M4_REFERENCE,
    }

    return reference


# ============================================================
# MAIN
# ============================================================

def main():

    try:

        reference = generate_reference()

        print()
        print("Écriture du fichier :")
        print(OUTPUT_FILE.resolve())

        with OUTPUT_FILE.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                reference,
                file,
                indent=2,
                ensure_ascii=False,
            )

        print()
        print("=" * 60)
        print("REFERENCE PSI GENEREE AVEC SUCCES")
        print("=" * 60)

        print()
        print("Résumé :")

        for model in (
            "M1",
            "M2",
            "M3",
            "M4",
        ):

            if model in ("M1", "M2"):

                data = reference[model]

                print(
                    f"{model}: "
                    f"{len(data['distribution'])} bins, "
                    f"{data['n_observations']} observations"
                )

            else:

                print(
                    f"{model}: "
                    f"{len(reference[model])} valeurs horaires"
                )

        print()

    except Exception as exc:

        print()
        print("=" * 60)
        print("ERREUR")
        print("=" * 60)

        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise


if __name__ == "__main__":
    main()
