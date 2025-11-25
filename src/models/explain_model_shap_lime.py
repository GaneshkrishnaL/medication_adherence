import os

# Ensure matplotlib has a writable cache directory before importing pyplot
_DEFAULT_MPL_CACHE = os.path.join("models", "explain", ".matplotlib_cache")
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = _DEFAULT_MPL_CACHE

import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Backward-compatibility for older libraries expecting np.bool
if not hasattr(np, "bool"):
    np.bool = np.bool_

import shap
try:
    from lime.lime_tabular import LimeTabularExplainer
except ImportError:
    LimeTabularExplainer = None  # optional dependency; LIME explanations skipped if missing


# ---------------- PATHS / CONFIG ---------------- #

PROC_DIR = os.path.join("data", "processed")
MODEL_DIR = os.path.join("models")
EXPLAIN_DIR = os.path.join(MODEL_DIR, "explain")

os.makedirs(EXPLAIN_DIR, exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

TRAIN_FILE = os.path.join(PROC_DIR, "training_dataset.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "rf_best_model.joblib")


# ---------------- HELPERS: SAME FEATURE LOGIC AS TRAINING ---------------- #

def load_training_data():
    df = pd.read_csv(TRAIN_FILE, parse_dates=["start_date"])
    print(f"Loaded training data: {df.shape}")
    return df


def prepare_features_labels(df: pd.DataFrame):
    """
    Same logic as in train_sklearn_model.py
    Separate features (X) and label (y).
    Drop ID-like columns and leakage columns.
    """
    y = df["label_non_adherent"].values

    drop_cols = [
        "label_non_adherent",
        "pdc_180d",
        "member_id",
        "start_date",
        "drug_code",
    ]

    X = df.drop(columns=drop_cols, errors="ignore")

    categorical_cols = ["gender", "race", "county", "state", "urban_rural", "drug_class"]
    categorical_cols = [c for c in categorical_cols if c in X.columns]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    print("Feature columns (for explanation):")
    print("  Numeric:", numeric_cols)
    print("  Categorical:", categorical_cols)

    return X, y, numeric_cols, categorical_cols


def get_feature_names_from_preprocessor(preprocessor, numeric_cols, categorical_cols):
    """
    Extract feature names after ColumnTransformer (numeric + one-hot categorical).
    """
    feature_names = []

    # Numeric features (StandardScaler)
    if len(numeric_cols) > 0:
        feature_names.extend(numeric_cols)

    # Categorical features (OneHotEncoder)
    if len(categorical_cols) > 0:
        ohe = preprocessor.named_transformers_["cat"]
        cat_feature_names = list(ohe.get_feature_names_out(categorical_cols))
        feature_names.extend(cat_feature_names)

    return feature_names


# ---------------- SHAP EXPLANATIONS ---------------- #

def run_shap_explanations(pipeline, X, numeric_cols, categorical_cols, sample_size=200):
    """
    Use SHAP TreeExplainer on the RandomForest model inside the pipeline.
    Generate:
      - Global summary plot (beeswarm)
      - Dependence plot for top feature
      - Force plot for a single example (saved as HTML)
    """

    print("\n=== Running SHAP explanations ===")

    # Extract fitted preprocessor + model from pipeline
    preprocessor = pipeline.named_steps["preprocessor"]
    rf_model = pipeline.named_steps["model"]

    # Sample a subset of X for speed
    if len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=42)
    else:
        X_sample = X.copy()

    # Transform X_sample into model input space
    X_enc = preprocessor.transform(X_sample)

    # Get feature names after preprocessing
    feature_names = get_feature_names_from_preprocessor(
        preprocessor, numeric_cols, categorical_cols
    )

    # TreeExplainer works directly on tree-based models
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_enc)

    # We care about shap values for class 1 (non-adherent)
    # shap_values is a list [for_class_0, for_class_1] for classifiers
    if isinstance(shap_values, list):
        sv_class1 = shap_values[1]
        expected_value = explainer.expected_value[1]
    else:
        # In some SHAP versions, shap_values is just a 2D array
        sv_class1 = shap_values
        expected_value = explainer.expected_value

    # ---- 1) Global summary (beeswarm) ---- #
    plt.figure()
    shap.summary_plot(
        sv_class1,
        X_enc,
        feature_names=feature_names,
        show=False
    )
    summary_path = os.path.join(EXPLAIN_DIR, "shap_summary_beeswarm.png")
    plt.tight_layout()
    plt.savefig(summary_path, dpi=200)
    plt.close()
    print(f"Saved SHAP summary beeswarm plot to {summary_path}")

    # ---- 2) Dependence plot for top feature ---- #
    mean_abs_shap = np.mean(np.abs(sv_class1), axis=0)
    top_idx = int(np.argmax(mean_abs_shap))
    top_feature_name = feature_names[top_idx]

    plt.figure()
    shap.dependence_plot(
        top_idx,
        sv_class1,
        X_enc,
        feature_names=feature_names,
        show=False
    )
    dep_path = os.path.join(EXPLAIN_DIR, f"shap_dependence_{top_feature_name}.png")
    plt.tight_layout()
    plt.savefig(dep_path, dpi=200)
    plt.close()
    print(f"Saved SHAP dependence plot for '{top_feature_name}' to {dep_path}")

    # ---- 3) Force plot for a single example ---- #
    # Pick the first row from the sample
    i = 0
    force_plot = shap.force_plot(
        expected_value,
        sv_class1[i, :],
        X_enc[i, :],
        feature_names=feature_names,
        matplotlib=False
    )

    force_path = os.path.join(EXPLAIN_DIR, "shap_force_example.html")
    shap.save_html(force_path, force_plot)
    print(f"Saved SHAP force plot example to {force_path}")


def run_feature_importance_plot(pipeline, numeric_cols, categorical_cols):
    """
    Simple fallback explanation using the model's feature_importances_.
    """
    print("\n=== Running feature importance fallback ===")
    preprocessor = pipeline.named_steps["preprocessor"]
    rf_model = pipeline.named_steps["model"]
    feature_names = get_feature_names_from_preprocessor(preprocessor, numeric_cols, categorical_cols)

    importances = rf_model.feature_importances_
    order = np.argsort(importances)[::-1]

    plt.figure(figsize=(8, 6))
    plt.barh(
        [feature_names[i] for i in order],
        importances[order]
    )
    plt.gca().invert_yaxis()
    plt.tight_layout()
    fi_path = os.path.join(EXPLAIN_DIR, "feature_importance.png")
    plt.savefig(fi_path, dpi=200)
    plt.close()
    print(f"Saved feature importance plot to {fi_path}")


# ---------------- LIME EXPLANATIONS ---------------- #

def run_lime_explanations(pipeline, X, y, categorical_cols, num_features=10):
    """
    Use LIME to explain a single prediction.
    We treat the pipeline as a black-box and wrap predict_proba so that
    LIME can pass numpy arrays in and we convert them back to DataFrames.
    """

    if LimeTabularExplainer is None:
        print("\nLIME not installed; skipping LIME explanations.")
        return

    print("\n=== Running LIME explanations ===")

    feature_names_orig = list(X.columns)

    # Encode categorical columns as integer codes so LIME's scaler/discretizer don't choke on strings
    X_encoded = X.copy()
    cat_value_maps = {}
    categorical_indices = []
    categorical_names = {}
    for c in categorical_cols:
        if c not in X_encoded.columns:
            continue
        categorical_indices.append(feature_names_orig.index(c))
        categories = sorted(X_encoded[c].astype(str).unique().tolist())
        value_to_code = {v: i for i, v in enumerate(categories)}
        code_to_value = {i: v for v, i in value_to_code.items()}
        cat_value_maps[c] = code_to_value
        categorical_names[feature_names_orig.index(c)] = categories
        X_encoded[c] = X_encoded[c].astype(str).map(value_to_code).astype(int)

    def pipeline_predict_proba(np_array):
        """
        LIME will call this with a numpy array of shape (n_samples, n_features).
        We convert it back to a DataFrame with the correct column names,
        then call pipeline.predict_proba().
        """
        df_in = pd.DataFrame(np_array, columns=feature_names_orig)
        # Convert encoded categorical integers back to original strings
        for c in categorical_cols:
            if c not in df_in.columns or c not in cat_value_maps:
                continue
            mapping = cat_value_maps[c]
            df_in[c] = df_in[c].round().astype(int).map(mapping)
        return pipeline.predict_proba(df_in)

    # LIME uses the raw X values as its training background
    explainer = LimeTabularExplainer(
        training_data=X_encoded.values,
        feature_names=feature_names_orig,
        class_names=["adherent", "non_adherent"],
        categorical_features=categorical_indices,
        categorical_names=categorical_names,
        discretize_continuous=True,
        mode="classification"
    )

    # Pick an example to explain: here, choose one non-adherent sample if possible
    idx = None
    if 1 in np.unique(y):
        idx_candidates = np.where(y == 1)[0]
        if len(idx_candidates) > 0:
            idx = int(idx_candidates[0])

    if idx is None:
        idx = 0  # fallback

    x_instance = X_encoded.iloc[idx].values

    print(f"Explaining instance index: {idx}, label = {y[idx]}")

    exp = explainer.explain_instance(
        data_row=x_instance,
        predict_fn=pipeline_predict_proba,
        num_features=num_features
    )

    lime_path = os.path.join(EXPLAIN_DIR, "lime_example.html")
    exp.save_to_file(lime_path)
    print(f"Saved LIME explanation HTML to {lime_path}")


# ---------------- MAIN ---------------- #

def main():
    # 1) Load data + model
    df = load_training_data()
    print("\nLabel distribution in training data:")
    print(df["label_non_adherent"].value_counts(normalize=True))

    model = joblib.load(MODEL_PATH)
    print(f"\nLoaded trained pipeline model from {MODEL_PATH}")

    # 2) Prepare features and labels
    X, y, numeric_cols, categorical_cols = prepare_features_labels(df)

    # 3) Run SHAP explanations (optional; set ENABLE_SHAP=1 to force, otherwise fallback)
    enable_shap = os.environ.get("ENABLE_SHAP", "0") == "1"
    if enable_shap:
        try:
            run_shap_explanations(model, X, numeric_cols, categorical_cols, sample_size=200)
        except Exception as exc:
            print(f"SHAP explanations failed ({exc}); using feature importance fallback.")
            run_feature_importance_plot(model, numeric_cols, categorical_cols)
    else:
        print("\nSHAP explanations disabled by default in this environment. Set ENABLE_SHAP=1 to run them.")
        run_feature_importance_plot(model, numeric_cols, categorical_cols)

    # 4) Run LIME explanations
    run_lime_explanations(model, X, y, categorical_cols, num_features=10)

    print("\nAll SHAP and LIME artifacts saved in:", EXPLAIN_DIR)


if __name__ == "__main__":
    main()
