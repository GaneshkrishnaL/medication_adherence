import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUT_DIR = os.path.join(PROC_DIR, "outputs")

os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(PROC_DIR, "training_dataset.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "rf_best_model.joblib")


DROP_COLS = [
    "label_non_adherent",
    "pdc_180d",
]

def assign_risk_category(prob_nonadherent):
    if prob_nonadherent >= 0.8:
        return "HIGH"
    elif prob_nonadherent >= 0.6:
        return "MEDIUM"
    else:
        return "LOW"


def main():
    print("Loading data and model...")
    df = pd.read_csv(TRAIN_FILE, parse_dates=["start_date"])
    model = joblib.load(MODEL_PATH)

    # Features for model
    X = df.drop(columns=DROP_COLS, errors="ignore")

    # Predict probabilities
    proba = model.predict_proba(X)[:, 1]

    df_out = df.copy()
    df_out["prob_non_adherent"] = proba
    df_out["risk_category"] = df_out["prob_non_adherent"].apply(assign_risk_category)

    # This is what you feed into Power BI / Tableau
    out_path = os.path.join(OUT_DIR, "scored_members.csv")
    df_out.to_csv(out_path, index=False)
    print(f"Saved scored dataset to {out_path}")


if __name__ == "__main__":
    main()
