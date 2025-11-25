import os
from contextlib import nullcontext
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

try:
    import mlflow
    import mlflow.sklearn
except ImportError:
    mlflow = None  # optional dependency; skip logging if unavailable

# ---------------- CONFIG ---------------- #

PROC_DIR = os.path.join("data", "processed")
MODEL_DIR = os.path.join("models")
os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(PROC_DIR, "training_dataset.csv")

EXPERIMENT_NAME = "med_adherence_rf"


# ---------------- LOAD DATA ---------------- #

def load_training_data():
    df = pd.read_csv(TRAIN_FILE, parse_dates=["start_date"])
    print(f"Loaded training data: {df.shape}")
    return df


# ---------------- PREPARE FEATURES / LABEL ---------------- #

def prepare_features_labels(df: pd.DataFrame):
    """
    Separate features (X) and label (y).
    Drop ID-like columns and leakage columns.
    """

    # 1) Define label
    y = df["label_non_adherent"].values

    # 2) Drop columns that should NOT be used as input features
    drop_cols = [
        "label_non_adherent",
        "pdc_180d",          # contains outcome info -> leaks the label
        "member_id",
        "start_date",
        "drug_code"
    ]

    # Keep everything else as candidate features
    X = df.drop(columns=drop_cols, errors="ignore")

    # 3) Identify categorical vs numeric columns
    categorical_cols = ["gender", "race", "county", "state", "urban_rural", "drug_class"]
    categorical_cols = [c for c in categorical_cols if c in X.columns]

    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    print("Feature columns:")
    print("  Numeric:", numeric_cols)
    print("  Categorical:", categorical_cols)

    return X, y, numeric_cols, categorical_cols


# ---------------- BUILD PIPELINE ---------------- #

def build_pipeline(numeric_cols, categorical_cols):
    """
    Build a Scikit-learn Pipeline:
    - ColumnTransformer for preprocessing
    - RandomForestClassifier as the model
    """

    # Preprocess numeric features: scale to zero mean, unit variance
    numeric_transformer = StandardScaler()

    # Preprocess categorical features: one-hot encode
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # Base model
    rf_clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    # Full pipeline
    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", rf_clf),
        ]
    )

    return clf


# ---------------- BASELINE TRAINING ---------------- #

def train_baseline_model(X_train, X_test, y_train, y_test, numeric_cols, categorical_cols):
    """
    Train a baseline RandomForest model without hyperparameter tuning.
    """

    clf = build_pipeline(numeric_cols, categorical_cols)

    print("\nFitting baseline model...")
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        auc = np.nan

    print("\nBaseline performance:")
    print(f"  Accuracy: {acc:.3f}")
    print(f"  ROC-AUC:  {auc:.3f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    return clf, acc, auc


# ---------------- GRID SEARCH + MLFLOW ---------------- #

def tune_with_grid_search(X_train, X_test, y_train, y_test, numeric_cols, categorical_cols):
    """
    Run GridSearchCV to find better hyperparameters.
    Log results to MLflow.
    """

    clf = build_pipeline(numeric_cols, categorical_cols)

    # Define hyperparameter grid for Random Forest
    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 5, 10],
        "model__min_samples_split": [2, 5],
    }

    grid_search = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        cv=3,
        scoring="balanced_accuracy",
        n_jobs=1,  # avoid multiprocessing issues in restricted environments
        verbose=2,
    )

    # MLflow experiment setup
    use_mlflow = mlflow is not None
    if use_mlflow:
        mlflow.set_experiment(EXPERIMENT_NAME)
    else:
        print("\nMLflow not installed; skipping experiment logging.")

    run_ctx = mlflow.start_run(run_name="rf_grid_search") if use_mlflow else nullcontext()
    with run_ctx:
        print("\nStarting GridSearchCV...")
        grid_search.fit(X_train, y_train)

        best_clf = grid_search.best_estimator_
        print("\nBest params:", grid_search.best_params_)
        print("Best CV ROC-AUC:", grid_search.best_score_)

        # Evaluate on the test set
        y_pred = best_clf.predict(X_test)
        y_proba = best_clf.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_proba)
        except ValueError:
            auc = np.nan

        print("\nTuned model performance on TEST set:")
        print(f"  Accuracy: {acc:.3f}")
        print(f"  ROC-AUC:  {auc:.3f}")
        print("\nConfusion matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification report:")
        print(classification_report(y_test, y_pred))

        # Log parameters and metrics to MLflow
        if use_mlflow:
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric("test_accuracy", acc)
            mlflow.log_metric("test_roc_auc", auc)

            # Log model
            mlflow.sklearn.log_model(best_clf, artifact_path="rf_model")

    return best_clf, grid_search.best_params_, acc, auc


# ---------------- MAIN ---------------- #

def main():
    # 1) Load data
    df = load_training_data()

    # Quick class balance check
    print("\nLabel distribution (0 = adherent, 1 = non-adherent):")
    print(df["label_non_adherent"].value_counts(normalize=True))

    # 2) Prepare X, y
    X, y, numeric_cols, categorical_cols = prepare_features_labels(df)

    # 3) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y  # keep label proportion similar in train and test
    )

    print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # 4) Baseline model
    baseline_clf, base_acc, base_auc = train_baseline_model(
        X_train, X_test, y_train, y_test, numeric_cols, categorical_cols
    )

    # 5) Hyperparameter tuning with GridSearchCV + MLflow
    best_clf, best_params, tuned_acc, tuned_auc = tune_with_grid_search(
        X_train, X_test, y_train, y_test, numeric_cols, categorical_cols
    )

    # 6) Save the best model locally as a pickle/joblib via MLflow artifact or manually if desired
    # NOTE: simplest is to rely on MLflow logged model, but we can also save manually:
    try:
        import joblib
        model_path = os.path.join(MODEL_DIR, "rf_best_model.joblib")
        joblib.dump(best_clf, model_path)
        print(f"\nSaved best model to {model_path}")
    except ImportError:
        print("\njoblib not installed; skipping local model save.")

    print("\nDone.")


if __name__ == "__main__":
    main()
