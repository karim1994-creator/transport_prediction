from pathlib import Path
import json
import shutil


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

REFERENCE_FILE = (
    BASE_DIR
    / "ci_reports"
    / "reference_model_performance.json"
)

CANDIDATE_FILE = (
    BASE_DIR
    / "ci_reports"
    / "candidate_model_performance.json"
)

MODELS_DIR = (
    BASE_DIR
    / "models_a53"
)

CANDIDATE_DIR = (
    MODELS_DIR
    / "candidate"
)

BACKUP_DIR = (
    MODELS_DIR
    / "backup"
)


MODEL_FILES = {
    "M1":
        "M1_hgb_historique_causal.joblib",

    "M2":
        "M2_hgb_historique_causal.joblib",
}

# ============================================================
# TOLERANCE DE COMPARAISON
# ============================================================

PERFORMANCE_TOLERANCE = 0.001

# ============================================================
# CHARGEMENT JSON
# ============================================================

def load_json(path):

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


# ============================================================
# COMPARAISON
# ============================================================

def compare_model(
    model_name,
    reference,
    candidate,
):

    old = next(
        item
        for item in reference
        if item["model"] == model_name
    )

    new = next(
        item
        for item in candidate
        if item["model"] == model_name
    )

    tolerance = PERFORMANCE_TOLERANCE

    mae_ok = (
        new["MAE"]
        <= old["MAE"] * (1 + tolerance)
    )

    rmse_ok = (
        new["RMSE"]
        <= old["RMSE"] * (1 + tolerance)
    )

    r2_ok = (
        new["R2"]
        >= old["R2"] * (1 - tolerance)
    )
    accepted = (
        mae_ok
        and rmse_ok
        and r2_ok
    )

    return {
        "model":
            model_name,

        "accepted":
            accepted,

        "reference":
            {
                "MAE":
                    old["MAE"],

                "RMSE":
                    old["RMSE"],

                "R2":
                    old["R2"],
            },

        "candidate":
            {
                "MAE":
                    new["MAE"],

                "RMSE":
                    new["RMSE"],

                "R2":
                    new["R2"],
            },

        "checks":
            {
                "MAE":
                    mae_ok,

                "RMSE":
                    rmse_ok,

                "R2":
                    r2_ok,
            },
    }


# ============================================================
# PROMOTION
# ============================================================

def promote_model(model_name):

    filename = MODEL_FILES[
        model_name
    ]

    candidate = (
        CANDIDATE_DIR
        / filename
    )

    production = (
        MODELS_DIR
        / filename
    )

    backup = (
        BACKUP_DIR
        / filename
    )

    if not candidate.exists():

        raise FileNotFoundError(
            f"Candidat absent : {candidate}"
        )

    # Sauvegarde de la production
    if production.exists():

        BACKUP_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        shutil.copy2(
            production,
            backup
        )

    # Promotion
    shutil.copy2(
        candidate,
        production
    )

    print(
        f"{model_name} : PROMOTION EFFECTUEE"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    reference = load_json(
        REFERENCE_FILE
    )

    candidate = load_json(
        CANDIDATE_FILE
    )

    print("=" * 90)
    print("DECISION AUTOMATIQUE DE PROMOTION")
    print("=" * 90)

    results = []

    for model_name in MODEL_FILES:

        result = compare_model(
            model_name,
            reference,
            candidate,
        )

        results.append(
            result
        )

        print()
        print("-" * 70)

        print(
            model_name
        )

        print(
            "Ancien MAE :",
            result["reference"]["MAE"]
        )

        print(
            "Nouveau MAE :",
            result["candidate"]["MAE"]
        )

        print(
            "Ancien RMSE :",
            result["reference"]["RMSE"]
        )

        print(
            "Nouveau RMSE :",
            result["candidate"]["RMSE"]
        )

        print(
            "Ancien R² :",
            result["reference"]["R2"]
        )

        print(
            "Nouveau R² :",
            result["candidate"]["R2"]
        )

        print()

        if result["accepted"]:

            print(
                "DECISION : PROMOTE"
            )

            # Promotion volontairement activée
            # uniquement si les 3 critères sont satisfaits.
            promote_model(
                model_name
            )

        else:

            print(
                "DECISION : REJECT"
            )

    # --------------------------------------------------------
    # RAPPORT
    # --------------------------------------------------------

    report_file = (
        BASE_DIR
        / "ci_reports"
        / "model_promotion_report.json"
    )

    with open(
        report_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 90)

    print(
        "RAPPORT DE PROMOTION :"
    )

    print(
        report_file
    )

    print("=" * 90)


if __name__ == "__main__":

    main()