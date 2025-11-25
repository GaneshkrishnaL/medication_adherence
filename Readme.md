Repository Structure
med-adherence-risk/
├─ data/
│  ├─ raw/
│  │  ├─ member_info.csv
│  │  ├─ sdoh_county.csv
│  │  ├─ drug_lookup.csv
│  │  ├─ pharmacy_claims.csv
│  │  └─ facility_stays.csv
│  ├─ processed/
│  │  ├─ training_dataset.csv
│  │  ├─ outputs/
│  │  │  └─ scored_members.csv
│  │  └─ bi/
│  │     ├─ kpi_by_drug_class.csv
│  │     ├─ geo_high_risk_by_county_state.csv
│  │     ├─ trend_high_risk_by_month.csv
│  │     ├─ high_risk_by_top_pharmacies.csv
│  │     └─ high_risk_by_top_providers.csv
│  └─ README.md  (optional notes about data generation)
│
├─ models/
│  ├─ rf_best_model.joblib
│  └─ explain/
│     ├─ shap_summary_beeswarm.png
│     ├─ shap_dependence_<top_feature>.png
│     ├─ shap_force_example.html
│     └─ lime_example.html
│
├─ notebooks/
│  ├─ 01_generate_synthetic_data.ipynb      (optional, if you convert)
│  ├─ 02_train_model_sklearn.ipynb          (optional, if you convert)
│  ├─ 03_explain_model_shap_lime.ipynb      (optional, if you convert)
│  └─ med_adherence_bi_dashboard_prep.ipynb
│
├─ src/
│  ├─ data/
│  │  └─ generate_synthetic_data.py         (raw data generator)
│  ├─ features/
│  │  └─ build_training_dataset.py          (feature engineering + label generation)
│  ├─ models/
│  │  ├─ train_sklearn_model.py             (pipeline + GridSearchCV + MLflow)
│  │  └─ explain_model_shap_lime.py         (SHAP + LIME)
│  ├─ pipeline/
│  │  └─ batch_score.py                     (batch scoring → scored_members.csv)
│  └─ api/
│     └─ app.py                             (Flask REST API for /score)
│
├─ requirements.txt
├─ Dockerfile
├─ README.md
└─ .gitignore
🧾 Example README.md (you can paste & adjust)
# Medication Adherence Risk Prediction (End-to-End ML + MLOps Demo)

This project simulates a **McKesson-style medication adherence risk product**:

- Synthetic patient, claims, and SDOH data
- End-to-end ML pipeline (feature engineering → model training → explainability)
- REST API deployment with Flask + Docker
- BI-ready outputs for Power BI / Tableau dashboards

---

## 1. Project Overview

**Goal:**  
Identify patients at high risk of being **non-adherent** to newly initiated medications, so care teams can prioritize outreach and interventions.

**Key capabilities:**

- Builds a risk model tailored to a specific patient population
- Scores new users initiating therapy and assigns **LOW / MEDIUM / HIGH** risk
- Aggregates risk by **drug class, geography, pharmacy, provider**
- Provides **explainability** using SHAP & LIME
- Exposes a **REST API** for near real-time scoring
- Produces **BI-ready datasets** for Power BI / Tableau

---

## 2. Data & Features

Synthetic raw tables in `data/raw/`:

- `member_info.csv` – demographics, enrollment
- `sdoh_county.csv` – social determinants by county (pct_uninsured, etc.)
- `drug_lookup.csv` – drug→drug class mapping
- `pharmacy_claims.csv` – historical pharmacy claims
- `facility_stays.csv` – inpatient stays

Feature engineering (`src/features/build_training_dataset.py`) creates:

- Demographics: `age_at_start`, `gender`, `race`, `county`, `state`, `urban_rural`
- SDOH: `pct_uninsured`, `pct_food_stamp`, `pct_public_transport`, `pct_less_hs_edu`,
  `pct_disabled`, `total_mh_providers`
- Utilization (lookback): `num_fills_lookback`, `total_days_supply_lookback`,
  `num_distinct_drugs_lookback`, `num_distinct_pharmacies_lookback`,
  `num_distinct_providers_lookback`, `total_paid_lookback`,
  `num_stays_lookback`, `total_stay_days_lookback`
- Label:
  - Synthetic risk score using SDOH + utilization
  - Converted to:
    - `pdc_180d` – synthetic proportion of days covered
    - `label_non_adherent` – 1 if high risk, 0 otherwise

---

## 3. Modeling

Training script: `src/models/train_sklearn_model.py`

- Uses a **Scikit-learn Pipeline**:
  - `ColumnTransformer` for numeric scaling + categorical one-hot encoding
  - `RandomForestClassifier` with `class_weight="balanced"`
- Hyperparameter tuning:
  - `GridSearchCV` over `n_estimators`, `max_depth`, `min_samples_split`
  - Optimizes **ROC-AUC / balanced accuracy**
- Experiment tracking:
  - Logs best params & metrics to **MLflow**
  - Saves best model as `models/rf_best_model.joblib`

Typical performance (on synthetic data):

- Accuracy: ~0.65–0.70
- ROC-AUC: ~0.70
- Balanced confusion matrix for adherent/non-adherent

---

## 4. Explainability (SHAP + LIME)

Script: `src/models/explain_model_shap_lime.py`

- Loads `rf_best_model.joblib` + `training_dataset.csv`
- Uses **SHAP TreeExplainer**:
  - Global summary beeswarm plot
  - Dependence plot for top feature (e.g., `pct_uninsured`)
  - Force plot HTML for a single patient
- Uses **LIME Tabular**:
  - Local explanation for a single prediction
  - Ranks features driving non-adherence risk

Outputs in `models/explain/`:

- `shap_summary_beeswarm.png`
- `shap_dependence_<feature>.png`
- `shap_force_example.html`
- `lime_example.html`

---

## 5. REST API (Flask + Docker)

API script: `src/api/app.py`

- Loads the trained pipeline
- Exposes:
  - `GET /health` – health check
  - `POST /score` – score one or multiple members

Example request:

```json
POST /score
Content-Type: application/json

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
Example response:
{
  "results": [
    {
      "age_at_start": 65,
      "gender": "F",
      "race": "WHITE",
      "...": "...",
      "prob_non_adherent": 0.73,
      "risk_category": "MEDIUM"
    }
  ]
}
Docker
Dockerfile:
Uses python:3.11-slim
Installs requirements.txt
Runs src/api/app.py on port 5000
Usage:
docker build -t med-adherence-api .
docker run -p 5000:5000 med-adherence-api
6. Batch Scoring & BI Outputs
Script: src/pipeline/batch_score.py
Loads training_dataset.csv + rf_best_model.joblib
Creates data/processed/outputs/scored_members.csv with:
prob_non_adherent
risk_category
Notebook: notebooks/med_adherence_bi_dashboard_prep.ipynb
Reads scored_members.csv + pharmacy_claims.csv
Produces aggregated CSVs in data/processed/bi/:
kpi_by_drug_class.csv
geo_high_risk_by_county_state.csv
trend_high_risk_by_month.csv
high_risk_by_top_pharmacies.csv
high_risk_by_top_providers.csv
These feed Power BI / Tableau dashboards:
KPI tiles:
% high-risk members
Avg non-adherence risk by drug class
Maps:
High-risk count & rate by county/state
Overlays for SDOH metrics
Trend charts:
High-risk count & rate by month
Bar charts:
High-risk rate by top pharmacies/providers
7. Compliance & Governance (HIPAA/GDPR Considerations)
Although this project uses synthetic data, the design mirrors real-world controls:
Data minimization:
No names or direct identifiers in modeling/BI artifacts
Member IDs can be hashed or pseudonymized
Access control:
Only ML pipeline & scoring services access raw data
BI layer uses aggregated, de-identified outputs
Security:
Data at rest encrypted (e.g., S3 + KMS, RDS encryption)
All service-to-service calls over TLS/HTTPS
Auditability:
Model version + parameters tracked in MLflow
Each batch scoring run can log timestamp, model version, and metrics
8. Getting Started
# 1. Create & activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate  # (Windows)
# source .venv/bin/activate  # (Mac/Linux)

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate raw synthetic data (if script provided)
python src/data/generate_synthetic_data.py

# 4. Build training dataset
python src/features/build_training_dataset.py

# 5. Train model
python src/models/train_sklearn_model.py

# 6. Batch score members
python src/pipeline/batch_score.py

# 7. Run BI prep notebook for dashboard datasets
# (open notebooks/med_adherence_bi_dashboard_prep.ipynb in Jupyter / VS Code)