from pathlib import Path
import json


BASE_DIR = Path(__file__).resolve().parent

SNAPSHOT_FILE = (
    BASE_DIR
    / "ci_reports"
    / "training_data_snapshot.json"
)

SIMULATED_FILE = (
    BASE_DIR
    / "ci_reports"
    / "training_data_simulated.json"
)


if not SNAPSHOT_FILE.exists():
    raise FileNotFoundError(
        f"Snapshot absent : {SNAPSHOT_FILE}"
    )


with open(
    SNAPSHOT_FILE,
    "r",
    encoding="utf-8",
) as file:

    snapshot = json.load(file)


# ------------------------------------------------------------
# Simulation d'une nouvelle date pour M1
# ------------------------------------------------------------

snapshot["snapshot_timestamp"] = (
    snapshot.get(
        "snapshot_timestamp",
        ""
    )
    + " | simulation"
)

snapshot["sources"]["M1"]["date_max"] = "2026-01-01"

# Le volume peut également augmenter dans un vrai scénario.
# Ici, on ne modifie volontairement que la date afin de tester
# le mécanisme de détection.


with open(
    SIMULATED_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        snapshot,
        file,
        indent=2,
        ensure_ascii=False,
    )


print("=" * 80)
print("SNAPSHOT SIMULE CREE")
print("=" * 80)

print(
    "Fichier :",
    SIMULATED_FILE
)

print(
    "M1 date_max simulée :",
    snapshot["sources"]["M1"]["date_max"]
)

print("=" * 80)