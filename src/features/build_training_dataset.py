import os
import numpy as np
import pandas as pd
from datetime import timedelta

# ---------------- CONFIG ---------------- #

RAW_DIR = os.path.join("data", "raw")
PROC_DIR = os.path.join("data", "processed")
os.makedirs(PROC_DIR, exist_ok=True)

LOOKBACK_DAYS = 180       # how far back we look at history
FOLLOWUP_DAYS = 180       # kept for semantics (PDC_180d name)

np.random.seed(42)        # for reproducible synthetic labels


# ---------------- LOAD RAW TABLES ---------------- #

def load_raw():
    """
    Load all raw CSVs from data/raw.
    """
    df_members = pd.read_csv(
        os.path.join(RAW_DIR, "member_info.csv"),
        parse_dates=["dob", "enrollment_start", "enrollment_end"]
    )
    df_sdoh = pd.read_csv(os.path.join(RAW_DIR, "sdoh_county.csv"))
    df_drugs = pd.read_csv(os.path.join(RAW_DIR, "drug_lookup.csv"))
    df_claims = pd.read_csv(
        os.path.join(RAW_DIR, "pharmacy_claims.csv"),
        parse_dates=["fill_date"]
    )
    df_stays = pd.read_csv(
        os.path.join(RAW_DIR, "facility_stays.csv"),
        parse_dates=["admit_date", "discharge_date"]
    )
    df_new = pd.read_csv(
        os.path.join(RAW_DIR, "new_users.csv"),
        parse_dates=["start_date"]
    )
    return df_members, df_sdoh, df_drugs, df_claims, df_stays, df_new


# ---------------- ADD DEMOGRAPHICS + SDOH ---------------- #

def add_demo_sdoh(df_new, df_members, df_sdoh):
    """
    For each new user, attach demographics and social determinant data.
    """
    # Join member_info to new_users
    df = df_new.merge(df_members, on="member_id", how="left", validate="m:1")

    # Age at therapy start
    df["age_at_start"] = (df["start_date"] - df["dob"]).dt.days // 365

    # Join SDOH by county/state
    df = df.merge(df_sdoh, on=["county", "state"], how="left", validate="m:1")
    return df


# ---------------- BUILD BEHAVIOR FEATURES (LOOKBACK ONLY) ---------------- #

def build_features(df_new, df_claims, df_stays):
    """
    For each (member, drug) start, compute:
    - historical claims features in lookback window
    - facility stays features in lookback window

    NOTE: We no longer compute labels here.
    Labels will be generated later based on demographics + SDOH + these features.
    """
    # We still ensure we have enough claims history for lookback
    claims_min = df_claims["fill_date"].min()
    claims_max = df_claims["fill_date"].max()

    # Just enforce that start is not *before* claims_min + LOOKBACK_DAYS
    min_start_for_full_lookback = claims_min + timedelta(days=LOOKBACK_DAYS)
    df_new = df_new[df_new["start_date"] >= min_start_for_full_lookback].copy()

    # Optional: sample to keep run time manageable for first try
    if len(df_new) > 20000:
        df_new = df_new.sample(n=20000, random_state=42).copy()

    feature_rows = []

    for idx, row in df_new.iterrows():
        member_id = row["member_id"]
        start_date = row["start_date"]
        drug_code = row["drug_code"]
        drug_class = row["drug_class"]

        lookback_start = start_date - timedelta(days=LOOKBACK_DAYS)

        # ---- 1) Lookback pharmacy claims (any drug) ---- #
        mask_lb = (
            (df_claims["member_id"] == member_id) &
            (df_claims["fill_date"] >= lookback_start) &
            (df_claims["fill_date"] < start_date)
        )
        claims_lb = df_claims.loc[mask_lb]

        num_fills_lb = len(claims_lb)
        total_days_supply_lb = claims_lb["days_supply"].sum() if num_fills_lb > 0 else 0
        num_distinct_drugs_lb = claims_lb["drug_code"].nunique() if num_fills_lb > 0 else 0
        num_distinct_pharm_lb = claims_lb["pharmacy_id"].nunique() if num_fills_lb > 0 else 0
        num_distinct_prov_lb = claims_lb["ordering_provider_id"].nunique() if num_fills_lb > 0 else 0
        total_paid_lb = claims_lb["paid_amount"].sum() if num_fills_lb > 0 else 0.0

        # ---- 2) Lookback facility stays ---- #
        mask_stays = (
            (df_stays["member_id"] == member_id) &
            (df_stays["admit_date"] >= lookback_start) &
            (df_stays["admit_date"] < start_date)
        )
        stays_lb = df_stays.loc[mask_stays]
        num_stays_lb = len(stays_lb)

        total_stay_days_lb = 0
        if num_stays_lb > 0:
            total_stay_days_lb = (
                (stays_lb["discharge_date"] - stays_lb["admit_date"])
                .dt.days.clip(lower=0)
                .sum()
            )

        feature_rows.append({
            "member_id": member_id,
            "drug_code": drug_code,
            "drug_class": drug_class,
            "start_date": start_date,

            # Lookback features
            "num_fills_lookback": num_fills_lb,
            "total_days_supply_lookback": total_days_supply_lb,
            "num_distinct_drugs_lookback": num_distinct_drugs_lb,
            "num_distinct_pharmacies_lookback": num_distinct_pharm_lb,
            "num_distinct_providers_lookback": num_distinct_prov_lb,
            "total_paid_lookback": total_paid_lb,
            "num_stays_lookback": num_stays_lb,
            "total_stay_days_lookback": total_stay_days_lb,
        })

    df_feat = pd.DataFrame(feature_rows)
    return df_feat


# ---------------- SYNTHETIC RISK & LABEL GENERATION ---------------- #
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate_labels_with_risk(df_full: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a synthetic non-adherence probability and label based on:
    - demographics (age)
    - SDOH (uninsured %, food stamps %, disabled %)
    - utilization (fills, stays, total paid)

    Steps:
    1) Build a continuous risk_score from these features.
    2) Convert to p_nonadherent via sigmoid(risk_score).
    3) Choose a percentile-based threshold on p_nonadherent
       so that we control the fraction labeled as 1 (non-adherent).
    4) Create a synthetic PDC inversely related to p_nonadherent.

    This guarantees a more balanced label distribution (e.g., ~40% non-adherent).
    """

    df = df_full.copy()

    # Fill NaNs with 0 for features we use
    for col in [
        "age_at_start",
        "pct_uninsured",
        "pct_food_stamp",
        "pct_public_transport",
        "pct_less_hs_edu",
        "pct_disabled",
        "num_fills_lookback",
        "num_stays_lookback",
        "total_paid_lookback",
    ]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Scale some features into reasonable ranges
    age = df.get("age_at_start", 0) / 100.0            # 0–0.9
    pct_uninsured = df.get("pct_uninsured", 0)         # ~0–0.3
    pct_food = df.get("pct_food_stamp", 0)
    pct_disabled = df.get("pct_disabled", 0)
    num_fills = df.get("num_fills_lookback", 0)
    num_stays = df.get("num_stays_lookback", 0)
    total_paid = df.get("total_paid_lookback", 0) / 1000.0  # scale payments

    # Random noise to avoid a perfectly sharp boundary
    noise = np.random.normal(0, 0.5, size=len(df))

    # Risk score: higher with worse SDOH + more complex utilization
    risk_score = (
        0.8 * pct_uninsured +
        0.6 * pct_food +
        0.5 * pct_disabled +
        0.05 * num_fills +
        0.15 * num_stays +
        0.3 * age +
        0.4 * total_paid +
        noise
    )

    # Convert to probability of non-adherence
    p_nonadherent = sigmoid(risk_score)

    # ---- KEY CHANGE: percentile-based threshold ----
    # We want, say, ~40% of members to be non-adherent.
    target_fraction_nonadherent = 0.4
    thresh = np.quantile(p_nonadherent, 1 - target_fraction_nonadherent)
    # Example: if target_fraction_nonadherent=0.4, threshold is 60th percentile.
    # Those above threshold are labeled non-adherent (1).

    labels = (p_nonadherent > thresh).astype(int)

    # Synthetic PDC: high risk -> low PDC (with some noise), clipped to [0,1]
    pdc = np.clip(
        1.0 - p_nonadherent + np.random.normal(0, 0.1, size=len(df)),
        0.0,
        1.0
    )

    df["pdc_180d"] = pdc
    df["label_non_adherent"] = labels

    return df


# ---------------- MAIN PIPELINE ---------------- #

def main():
    # 1) Load raw tables
    df_members, df_sdoh, df_drugs, df_claims, df_stays, df_new = load_raw()

    # 2) Attach demographics + SDOH to each new user row
    df_new_demo = add_demo_sdoh(df_new, df_members, df_sdoh)

    # 3) Build behavior features (lookback) + merge
    df_behavior = build_features(
        df_new_demo[["member_id", "drug_code", "drug_class", "start_date"]].copy(),
        df_claims,
        df_stays
    )

    df_full = df_behavior.merge(
        df_new_demo,
        on=["member_id", "drug_code", "drug_class", "start_date"],
        how="left",
        validate="m:1"
    )

    # 4) Generate synthetic PDC + label from risk model
    df_full = generate_labels_with_risk(df_full)

    # 5) Arrange columns nicely
    cols_label = ["label_non_adherent", "pdc_180d"]
    cols_id = ["member_id", "drug_code", "drug_class", "start_date"]
    cols_demo = ["age_at_start", "gender", "race", "county", "state"]
    cols_sdoh = [
        "pct_uninsured", "pct_food_stamp", "pct_public_transport",
        "pct_less_hs_edu", "pct_disabled", "urban_rural", "total_mh_providers"
    ]
    cols_behavior = [
        "num_fills_lookback", "total_days_supply_lookback",
        "num_distinct_drugs_lookback", "num_distinct_pharmacies_lookback",
        "num_distinct_providers_lookback", "total_paid_lookback",
        "num_stays_lookback", "total_stay_days_lookback"
    ]

    ordered_cols = cols_id + cols_demo + cols_sdoh + cols_behavior + cols_label
    ordered_cols = [c for c in ordered_cols if c in df_full.columns]

    df_full = df_full[ordered_cols].copy()

    # 6) Save training dataset
    out_path = os.path.join(PROC_DIR, "training_dataset.csv")
    df_full.to_csv(out_path, index=False)
    print(f"training_dataset.csv -> {df_full.shape}")
    print("Label distribution after synthetic generation:")
    print(df_full["label_non_adherent"].value_counts(normalize=True))


if __name__ == "__main__":
    main()
