import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

# --------- CONFIG ------------

N_MEMBERS = 50_0        # you can increase later (100k, 500k)
N_PHARMACY_CLAIMS = 400_0
N_FACILITY_STAYS = 30_0

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

RAW_DIR = os.path.join("data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

# --------- HELPER FUNCTIONS ------------

def random_date(start, end):
    """
    Return a random datetime between `start` and `end`.
    """
    delta = end - start
    rand_days = np.random.randint(0, delta.days + 1)
    return start + timedelta(days=int(rand_days))

def choose_with_probs(choices, probs):
    """
    Helper that picks one item from choices given probabilities.
    """
    return np.random.choice(choices, p=probs)

# --------- 1. MEMBER INFO ------------

def generate_members():
    """
    Generate synthetic member information: demographics + enrollment.
    """
    member_ids = [f"M{str(i).zfill(7)}" for i in range(1, N_MEMBERS + 1)]

    # define possible values
    genders = ["M", "F"]
    gender_probs = [0.48, 0.52]

    races = ["WHITE", "BLACK", "ASIAN", "HISPANIC", "OTHER"]
    race_probs = [0.6, 0.2, 0.1, 0.05, 0.05]

    counties = ["Charleston", "Greenville", "Horry", "Lexington",
                "Spartanburg", "Anderson", "Beaufort", "Pickens"]
    states = ["SC"]  # you can add more states if you like

    today = datetime(2024, 12, 31)
    start_enroll_base = datetime(2018, 1, 1)

    rows = []
    for m in member_ids:
        # age 18-90
        age = np.random.randint(18, 90)
        # dob = today - age years (approx)
        dob_year = today.year - age
        dob = datetime(dob_year, np.random.randint(1, 13), np.random.randint(1, 28))

        gender = choose_with_probs(genders, gender_probs)
        race = choose_with_probs(races, race_probs)
        county = np.random.choice(counties)
        state = np.random.choice(states)

        # enrollment: random start/end within range
        enroll_start = random_date(start_enroll_base, today - timedelta(days=365))
        # ensure at least 1 year enrollment
        enroll_end = random_date(enroll_start + timedelta(days=365), today)

        rows.append({
            "member_id": m,
            "dob": dob.date(),
            "gender": gender,
            "race": race,
            "county": county,
            "state": state,
            "enrollment_start": enroll_start.date(),
            "enrollment_end": enroll_end.date()
        })

    df_members = pd.DataFrame(rows)
    df_members.to_csv(os.path.join(RAW_DIR, "member_info.csv"), index=False)
    print(f"member_info.csv -> {df_members.shape}")

# --------- 2. SDOH (SOCIAL DETERMINANTS) ------------

def generate_sdoh():
    """
    Generate one row per (county, state) with SDOH metrics.
    """
    counties = ["Charleston", "Greenville", "Horry", "Lexington",
                "Spartanburg", "Anderson", "Beaufort", "Pickens"]
    states = ["SC"]

    rows = []
    for state in states:
        for county in counties:
            pct_uninsured = np.round(np.random.uniform(0.05, 0.25), 3)
            pct_food_stamp = np.round(np.random.uniform(0.05, 0.35), 3)
            pct_public_transport = np.round(np.random.uniform(0.0, 0.10), 3)
            pct_less_hs_edu = np.round(np.random.uniform(0.05, 0.30), 3)
            pct_disabled = np.round(np.random.uniform(0.10, 0.30), 3)
            urban_rural = np.random.choice(["URBAN", "RURAL"])
            total_mh_providers = np.random.randint(5, 200)

            rows.append({
                "county": county,
                "state": state,
                "pct_uninsured": pct_uninsured,
                "pct_food_stamp": pct_food_stamp,
                "pct_public_transport": pct_public_transport,
                "pct_less_hs_edu": pct_less_hs_edu,
                "pct_disabled": pct_disabled,
                "urban_rural": urban_rural,
                "total_mh_providers": total_mh_providers
            })

    df_sdoh = pd.DataFrame(rows)
    df_sdoh.to_csv(os.path.join(RAW_DIR, "sdoh_county.csv"), index=False)
    print(f"sdoh_county.csv -> {df_sdoh.shape}")

# --------- 3. DRUG LOOKUP ------------

def generate_drug_lookup():
    """
    Generate a small drug lookup table mapping drug_code -> drug_class.
    """
    drug_classes = [
        "CARDIOVASCULAR",
        "CNS_AGENTS",
        "DIABETES",
        "RESPIRATORY",
        "GI_AGENTS"
    ]

    rows = []
    for i in range(1, 101):  # 100 drug codes
        drug_code = f"{i:08d}"  # 8-digit code
        drug_class = np.random.choice(drug_classes)
        rows.append({
            "drug_code": drug_code,
            "drug_class": drug_class
        })

    df_drugs = pd.DataFrame(rows)
    df_drugs.to_csv(os.path.join(RAW_DIR, "drug_lookup.csv"), index=False)
    print(f"drug_lookup.csv -> {df_drugs.shape}")

# --------- 4. PHARMACY CLAIMS ------------

def generate_pharmacy_claims():
    """
    Generate synthetic pharmacy claims.
    """
    df_members = pd.read_csv(os.path.join(RAW_DIR, "member_info.csv"))
    df_drugs = pd.read_csv(os.path.join(RAW_DIR, "drug_lookup.csv"))

    member_ids = df_members["member_id"].tolist()
    drug_codes = df_drugs["drug_code"].tolist()
    drug_class_map = dict(zip(df_drugs["drug_code"], df_drugs["drug_class"]))

    # define date range for claims
    claims_start = datetime(2020, 1, 1)
    claims_end = datetime(2024, 12, 31)

    rows = []
    for claim_id in range(1, N_PHARMACY_CLAIMS + 1):
        member_id = random.choice(member_ids)
        drug_code = random.choice(drug_codes)
        drug_class = drug_class_map[drug_code]

        pharmacy_id = f"PH{np.random.randint(1, 500):04d}"
        provider_id = f"PR{np.random.randint(1, 2000):05d}"

        fill_date = random_date(claims_start, claims_end).date()
        days_supply = np.random.choice([30, 60, 90], p=[0.6, 0.25, 0.15])
        paid_amount = np.round(np.random.uniform(10, 400), 2)

        rows.append({
            "claim_id": claim_id,
            "member_id": member_id,
            "pharmacy_id": pharmacy_id,
            "ordering_provider_id": provider_id,
            "drug_code": drug_code,
            "drug_class": drug_class,
            "fill_date": fill_date,
            "days_supply": days_supply,
            "paid_amount": paid_amount
        })

    df_claims = pd.DataFrame(rows)
    df_claims.to_csv(os.path.join(RAW_DIR, "pharmacy_claims.csv"), index=False)
    print(f"pharmacy_claims.csv -> {df_claims.shape}")

# --------- 5. FACILITY STAYS ------------

def generate_facility_stays():
    """
    Generate synthetic inpatient facility stays.
    Only some fraction of members will have stays.
    """
    df_members = pd.read_csv(os.path.join(RAW_DIR, "member_info.csv"))
    member_ids = df_members["member_id"].tolist()

    stays_start = datetime(2020, 1, 1)
    stays_end = datetime(2024, 12, 31)

    rows = []
    for stay_id in range(1, N_FACILITY_STAYS + 1):
        member_id = random.choice(member_ids)

        admit_date = random_date(stays_start, stays_end)
        # stay length 1–10 days
        length = np.random.randint(1, 11)
        discharge_date = admit_date + timedelta(days=int(length))

        facility_type = np.random.choice(
            ["ACUTE", "PSYCH", "REHAB", "SNF"],
            p=[0.6, 0.1, 0.2, 0.1]
        )

        rows.append({
            "stay_id": stay_id,
            "member_id": member_id,
            "admit_date": admit_date.date(),
            "discharge_date": discharge_date.date(),
            "facility_type": facility_type
        })

    df_stays = pd.DataFrame(rows)
    df_stays.to_csv(os.path.join(RAW_DIR, "facility_stays.csv"), index=False)
    print(f"facility_stays.csv -> {df_stays.shape}")

# --------- 6. NEW USERS ------------

def generate_new_users():
    """
    Identify synthetic 'new users' who start a drug (for scoring).
    For simplicity we just sample claim records that look like 'first fill'
    for a member-drug pair.
    """
    df_claims = pd.read_csv(os.path.join(RAW_DIR, "pharmacy_claims.csv"))
    df_claims["fill_date"] = pd.to_datetime(df_claims["fill_date"])

    # Sort by member, drug, date
    df_claims_sorted = df_claims.sort_values(["member_id", "drug_code", "fill_date"])

    # For each (member, drug_code), take the earliest claim as the 'start'
    df_first = df_claims_sorted.groupby(["member_id", "drug_code"], as_index=False).first()

    # We'll call those 'new users'
    df_new_users = df_first[["member_id", "drug_code", "drug_class", "fill_date"]].rename(
        columns={"fill_date": "start_date"}
    )

    # For demo, we can optionally sample to reduce size
    df_new_users_sample = df_new_users.sample(frac=0.3, random_state=RANDOM_SEED)

    df_new_users_sample.to_csv(os.path.join(RAW_DIR, "new_users.csv"), index=False)
    print(f"new_users.csv -> {df_new_users_sample.shape}")

# --------- MAIN ------------

if __name__ == "__main__":
    generate_members()
    generate_sdoh()
    generate_drug_lookup()
    generate_pharmacy_claims()
    generate_facility_stays()
    generate_new_users()
    print("Synthetic raw data generation complete.")
