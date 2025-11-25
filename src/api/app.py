import os
import json
import socket

import pandas as pd
import joblib
from flask import Flask, request, jsonify

# ---------------- CONFIG ---------------- #

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

TRAIN_FILE = os.path.join(PROC_DIR, "training_dataset.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "rf_best_model.joblib")

# ---------------- LOAD MODEL + FEATURE SCHEMA ---------------- #

print("Loading model and training schema...")

model = joblib.load(MODEL_PATH)
df_train = pd.read_csv(TRAIN_FILE, parse_dates=["start_date"])

# Same feature logic as training
DROP_COLS = [
    "label_non_adherent",
    "pdc_180d",
    "member_id",
    "start_date",
    "drug_code",
]

X_train = df_train.drop(columns=DROP_COLS, errors="ignore")
FEATURE_COLS = list(X_train.columns)

print("Expected feature columns for scoring:")
print(FEATURE_COLS)

# ---------------- FLASK APP ---------------- #

app = Flask(__name__)


def preprocess_input(payload):
    """
    Accept either:
      - a single JSON object (dict) with feature values
      - a list of such objects

    Return a pandas DataFrame with columns in FEATURE_COLS order.
    Missing columns are filled with NaN.
    Extra keys are ignored.
    """
    if isinstance(payload, dict):
        records = [payload]
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError("Payload must be a dict or list of dicts.")

    df = pd.DataFrame(records)

    # Keep only known feature columns and add missing ones as NaN
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[FEATURE_COLS]
    return df


def assign_risk_category(prob_nonadherent):
    """
    Map probability of non-adherence to risk category.
    """
    if prob_nonadherent >= 0.8:
        return "HIGH"
    elif prob_nonadherent >= 0.6:
        return "MEDIUM"
    else:
        return "LOW"


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/score", methods=["POST"])
def score():
    """
    Example input (single member):

    {
      "age_at_start": 65,
      "gender": "F",
      "race": "WHITE",
      "county": "Greenville",
      "state": "SC",
      "urban_rural": "URBAN",
      "pct_uninsured": 0.12,
      "pct_food_stamp": 0.20,
      "pct_public_transport": 0.03,
      "pct_less_hs_edu": 0.10,
      "pct_disabled": 0.18,
      "total_mh_providers": 25,
      "num_fills_lookback": 5,
      "total_days_supply_lookback": 150,
      "num_distinct_drugs_lookback": 2,
      "num_distinct_pharmacies_lookback": 1,
      "num_distinct_providers_lookback": 2,
      "total_paid_lookback": 350.0,
      "num_stays_lookback": 1,
      "total_stay_days_lookback": 4,
      "drug_class": "DIABETES"
    }

    You can also send a list of such objects.
    """
    try:
        payload = request.get_json()
        if payload is None:
            return jsonify({"error": "Invalid or missing JSON payload"}), 400

        df_input = preprocess_input(payload)

        # Predict probabilities
        proba = model.predict_proba(df_input)[:, 1]  # class 1 = non-adherent

        results = []
        for i, p in enumerate(proba):
            risk = assign_risk_category(p)
            row = df_input.iloc[i].to_dict()
            row.update({
                "prob_non_adherent": float(p),
                "risk_category": risk
            })
            results.append(row)

        return jsonify({"results": results}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _pick_port(default_port=5000):
    """
    Choose a port:
    - Respect PORT/FLASK_PORT if set.
    - If default is busy, fall back to an ephemeral free port.
    """
    env_port = os.environ.get("PORT") or os.environ.get("FLASK_PORT")
    if env_port:
        return int(env_port)

    # Check if default_port is free
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        if s.connect_ex(("127.0.0.1", default_port)) != 0:
            return default_port

    # Find a free ephemeral port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


if __name__ == "__main__":
    port = _pick_port(default_port=5000)
    print(f"Starting Flask on port {port} (set PORT/FLASK_PORT to override).")
    app.run(host="0.0.0.0", port=port, debug=True)
