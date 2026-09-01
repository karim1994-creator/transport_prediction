from pathlib import Path
import json


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

SNAPSHOT_FILE = (
    BASE_DIR
    / "ci_reports"
    / "training_data_snapshot.json"
)


# ============================================================
# VERIFICATION
# ============================================================

if not SNAPSHOT_FILE.exists():

    raise FileNotFoundError(
        f"Snapshot absent : {SNAPSHOT_FILE}"
    )


with open(
    SNAPSHOT_FILE,
    "r",
    encoding="utf-8",
) as file:

    current_snapshot = json.load(file)


# ============================================================
# PRECEDENT SNAPSHOT
# ============================================================

PREVIOUS_FILE = (
    BASE_DIR
    / "ci_reports"
    / "training_data_previous_snapshot.json"
)


# ============================================================
# PREMIERE EXECUTION
# ============================================================

if not PREVIOUS_FILE.exists():

    print("=" * 90)
    print("PREMIERE COLLECTE")
    print("=" * 90)

    print()
    print(
        "Aucun snapshot précédent."
    )

    print(
        "Le snapshot actuel devient la référence."
    )

    with open(
        PREVIOUS_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            current_snapshot,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        "Snapshot précédent créé :"
    )

    print(
        PREVIOUS_FILE
    )

    print()
    print(
        "Nouvelles données détectées :"
    )

    print(
        "NON — première collecte"
    )

    raise SystemExit(0)


# ============================================================
# CHARGEMENT PRECEDENT
# ============================================================

with open(
    PREVIOUS_FILE,
    "r",
    encoding="utf-8",
) as file:

    previous_snapshot = json.load(file)


# ============================================================
# COMPARAISON
# ============================================================

print("=" * 90)
print("DETECTION DES NOUVELLES DONNEES D'APPRENTISSAGE")
print("=" * 90)


new_data_detected = False


for model_name in [
    "M1",
    "M2",
    "M3",
    "M4",
]:

    previous = previous_snapshot[
        "sources"
    ][model_name]

    current = current_snapshot[
        "sources"
    ][model_name]

    previous_date = previous[
        "date_max"
    ]

    current_date = current[
        "date_max"
    ]

    previous_rows = previous[
        "nb_lignes"
    ]

    current_rows = current[
        "nb_lignes"
    ]


    print()
    print("-" * 70)
    print(model_name)
    print("-" * 70)

    print(
        "Ancienne date max :",
        previous_date
    )

    print(
        "Nouvelle date max :",
        current_date
    )

    print(
        "Ancien nombre de lignes :",
        previous_rows
    )

    print(
        "Nouveau nombre de lignes :",
        current_rows
    )


    date_changed = (
        current_date
        > previous_date
    )

    rows_changed = (
        current_rows
        > previous_rows
    )


    if date_changed or rows_changed:

        print(
            "NOUVELLES DONNEES : OUI"
        )

        new_data_detected = True

    else:

        print(
            "NOUVELLES DONNEES : NON"
        )


# ============================================================
# MISE A JOUR DU SNAPSHOT PRECEDENT
# ============================================================

with open(
    PREVIOUS_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        current_snapshot,
        file,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# RESULTAT GLOBAL
# ============================================================

print()
print("=" * 90)

if new_data_detected:

    print(
        "STATUT GLOBAL : NOUVELLES DONNEES DETECTEES"
    )

else:

    print(
        "STATUT GLOBAL : AUCUNE NOUVELLE DONNEE"
    )

print("=" * 90)