from pathlib import Path
import json


BASE_DIR = Path(__file__).resolve().parent

REFERENCE_FILE = (
    BASE_DIR
    / "ci_reports"
    / "training_data_previous_snapshot.json"
)

SIMULATED_FILE = (
    BASE_DIR
    / "ci_reports"
    / "training_data_simulated.json"
)


def load_snapshot(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Fichier absent : {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


reference = load_snapshot(
    REFERENCE_FILE
)

simulated = load_snapshot(
    SIMULATED_FILE
)


print("=" * 90)
print("COMPARAISON SNAPSHOT REEL / SNAPSHOT SIMULE")
print("=" * 90)

new_data_detected = False


for model_name in ["M1", "M2", "M3", "M4"]:

    old = reference["sources"][model_name]
    new = simulated["sources"][model_name]

    old_date = old["date_max"]
    new_date = new["date_max"]

    old_rows = old["nb_lignes"]
    new_rows = new["nb_lignes"]

    date_changed = new_date > old_date
    rows_changed = new_rows > old_rows

    detected = (
        date_changed
        or rows_changed
    )

    print()
    print("-" * 70)
    print(model_name)
    print("-" * 70)

    print(
        "Ancienne date max :",
        old_date
    )

    print(
        "Nouvelle date max :",
        new_date
    )

    print(
        "Ancien nombre de lignes :",
        old_rows
    )

    print(
        "Nouveau nombre de lignes :",
        new_rows
    )

    print(
        "NOUVELLES DONNEES :",
        "OUI" if detected else "NON"
    )

    if detected:
        new_data_detected = True


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