"""
scikit-learn benchmark using two Kaggle datasets:

- Adult Census Income (classification + clustering)
- California Housing   (regression)

All six algorithms are timed with the median of 5 runs using
time.perf_counter(), and results are written to results_sklearn.json with a
fixed schema consumed by generate_charts.py.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
ADULT_CSV = DATA_DIR / "adult.csv"
HOUSING_CSV = DATA_DIR / "housing.csv"
OUTPUT_PATH = ROOT / "results_sklearn.json"

RANDOM_STATE = 42
N_RUNS = 5
TEST_SIZE = 0.2
SILHOUETTE_SAMPLE_SIZE = 5000

ADULT_NUMERIC = ["age", "fnlwgt", "educational-num", "capital-gain", "capital-loss", "hours-per-week"]
ADULT_CATEGORICAL = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "gender",
    "native-country",
]
ADULT_TARGET = "income"

HOUSING_NUMERIC = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]
HOUSING_CATEGORICAL = ["ocean_proximity"]
HOUSING_TARGET = "median_house_value"


def _load_adult() -> tuple[pd.DataFrame, np.ndarray]:
    if not ADULT_CSV.exists():
        raise FileNotFoundError(
            f"Missing {ADULT_CSV}. Download from "
            "https://www.kaggle.com/datasets/uciml/adult-census-income"
        )

    df = pd.read_csv(ADULT_CSV)

    # The Kaggle uciml/adult-census-income export uses dot-separated names and
    # 'sex' instead of 'gender'. Normalize to the canonical UCI names so the
    # rest of the pipeline is dataset-variant agnostic.
    rename_map = {
        "education.num": "educational-num",
        "marital.status": "marital-status",
        "capital.gain": "capital-gain",
        "capital.loss": "capital-loss",
        "hours.per.week": "hours-per-week",
        "native.country": "native-country",
        "sex": "gender",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    string_columns = ADULT_CATEGORICAL + [ADULT_TARGET]
    for column in string_columns:
        if column in df.columns:
            df[column] = df[column].astype(str).str.strip()

    # Drop rows where any field is '?' (typical in adult-census-income).
    mask_missing = (df == "?").any(axis=1)
    df = df.loc[~mask_missing].reset_index(drop=True)

    target = (df[ADULT_TARGET].str.replace(".", "", regex=False) == ">50K").astype(int).to_numpy()
    features = df[ADULT_NUMERIC + ADULT_CATEGORICAL].copy()
    return features, target


def _load_housing() -> tuple[pd.DataFrame, np.ndarray]:
    if not HOUSING_CSV.exists():
        raise FileNotFoundError(
            f"Missing {HOUSING_CSV}. Download with: "
            "kaggle datasets download -d camnugent/california-housing-prices -p data --unzip"
        )

    df = pd.read_csv(HOUSING_CSV)

    for column in HOUSING_CATEGORICAL:
        if column in df.columns:
            df[column] = df[column].astype(str).str.strip()

    # Impute missing total_bedrooms with the column median.
    median_bedrooms = df["total_bedrooms"].median()
    df["total_bedrooms"] = df["total_bedrooms"].fillna(median_bedrooms)

    target = df[HOUSING_TARGET].astype(np.float64).to_numpy()
    features = df[HOUSING_NUMERIC + HOUSING_CATEGORICAL].copy()
    return features, target


def _encode_adult(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    encoder.fit(train_df[ADULT_CATEGORICAL])

    train_numeric = train_df[ADULT_NUMERIC].to_numpy(dtype=np.float64)
    test_numeric = test_df[ADULT_NUMERIC].to_numpy(dtype=np.float64)
    train_categorical = encoder.transform(train_df[ADULT_CATEGORICAL])
    test_categorical = encoder.transform(test_df[ADULT_CATEGORICAL])

    x_train = np.concatenate([train_numeric, train_categorical], axis=1)
    x_test = np.concatenate([test_numeric, test_categorical], axis=1)
    return x_train, x_test


def _encode_housing(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    encoder.fit(train_df[HOUSING_CATEGORICAL])

    train_numeric = train_df[HOUSING_NUMERIC].to_numpy(dtype=np.float64)
    test_numeric = test_df[HOUSING_NUMERIC].to_numpy(dtype=np.float64)
    train_categorical = encoder.transform(train_df[HOUSING_CATEGORICAL])
    test_categorical = encoder.transform(test_df[HOUSING_CATEGORICAL])

    x_train = np.concatenate([train_numeric, train_categorical], axis=1)
    x_test = np.concatenate([test_numeric, test_categorical], axis=1)
    return x_train, x_test


def _time_ms(fn):
    start = time.perf_counter()
    result = fn()
    elapsed = (time.perf_counter() - start) * 1000.0
    return elapsed, result


def _median_of_5(train_fn, infer_fn) -> tuple[float, float, object, object]:
    train_times: list[float] = []
    infer_times: list[float] = []
    last_model = None
    last_preds = None
    for _ in range(N_RUNS):
        train_ms, model = _time_ms(train_fn)
        infer_ms, preds = _time_ms(lambda: infer_fn(model))
        train_times.append(train_ms)
        infer_times.append(infer_ms)
        last_model = model
        last_preds = preds
    return median(train_times), median(infer_times), last_model, last_preds


def _classification_metrics(y_true, y_pred, y_score) -> dict:
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred, average="weighted")),
        "Precision": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "Recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y_true, y_score)),
    }


def _run_classifiers(x_train, x_test, y_train, y_test) -> dict:
    results: dict = {}

    # Logistic Regression
    def _train_logreg():
        model = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE)
        model.fit(x_train, y_train)
        return model

    train_ms, infer_ms, model, preds = _median_of_5(
        _train_logreg, lambda m: m.predict(x_test)
    )
    scores = model.predict_proba(x_test)[:, 1]
    results["LogisticRegression"] = {
        "train_ms": train_ms,
        "infer_ms": infer_ms,
        **_classification_metrics(y_test, preds, scores),
    }

    # Decision Tree
    def _train_tree():
        model = DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE)
        model.fit(x_train, y_train)
        return model

    train_ms, infer_ms, model, preds = _median_of_5(
        _train_tree, lambda m: m.predict(x_test)
    )
    scores = model.predict_proba(x_test)[:, 1]
    results["DecisionTree"] = {
        "train_ms": train_ms,
        "infer_ms": infer_ms,
        **_classification_metrics(y_test, preds, scores),
    }

    # Random Forest
    def _train_rf():
        model = RandomForestClassifier(
            n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE
        )
        model.fit(x_train, y_train)
        return model

    train_ms, infer_ms, model, preds = _median_of_5(
        _train_rf, lambda m: m.predict(x_test)
    )
    scores = model.predict_proba(x_test)[:, 1]
    results["RandomForest"] = {
        "train_ms": train_ms,
        "infer_ms": infer_ms,
        **_classification_metrics(y_test, preds, scores),
    }

    # Linear SVM
    def _train_svm():
        model = LinearSVC(max_iter=2000, random_state=RANDOM_STATE)
        model.fit(x_train, y_train)
        return model

    train_ms, infer_ms, model, preds = _median_of_5(
        _train_svm, lambda m: m.predict(x_test)
    )
    scores = model.decision_function(x_test)
    results["SVM_Linear"] = {
        "train_ms": train_ms,
        "infer_ms": infer_ms,
        **_classification_metrics(y_test, preds, scores),
    }

    return results


def _run_clustering(x_train, x_test) -> dict:
    def _train_kmeans():
        model = MiniBatchKMeans(
            n_clusters=3,
            n_init=5,
            batch_size=4096,
            random_state=RANDOM_STATE,
        )
        model.fit(x_train)
        return model

    train_ms, infer_ms, model, preds = _median_of_5(
        _train_kmeans, lambda m: m.predict(x_test)
    )

    rng = np.random.default_rng(RANDOM_STATE)
    if x_test.shape[0] > SILHOUETTE_SAMPLE_SIZE:
        idx = rng.choice(x_test.shape[0], size=SILHOUETTE_SAMPLE_SIZE, replace=False)
        sample_x = x_test[idx]
        sample_labels = preds[idx]
    else:
        sample_x = x_test
        sample_labels = preds

    if len(np.unique(sample_labels)) > 1:
        silhouette = float(silhouette_score(sample_x, sample_labels))
    else:
        silhouette = float("nan")

    return {
        "KMeans": {
            "train_ms": train_ms,
            "infer_ms": infer_ms,
            "Silhouette": silhouette,
            "Inertia": float(model.inertia_),
        }
    }


def _run_regression(x_train, x_test, y_train, y_test) -> dict:
    def _train_linreg():
        model = LinearRegression()
        model.fit(x_train, y_train)
        return model

    train_ms, infer_ms, _, preds = _median_of_5(
        _train_linreg, lambda m: m.predict(x_test)
    )

    rmse = float(np.sqrt(np.mean((y_test - preds) ** 2)))
    return {
        "LinearRegression": {
            "train_ms": train_ms,
            "infer_ms": infer_ms,
            "R2": float(r2_score(y_test, preds)),
            "RMSE": rmse,
            "MAE": float(mean_absolute_error(y_test, preds)),
        }
    }


def _print_summary(results: dict) -> None:
    print()
    print("=" * 92)
    print(
        f"{'Algorithm':<22}{'Train (ms)':>14}{'Infer (ms)':>14}"
        f"{'Metric 1':>16}{'Metric 2':>14}{'Metric 3':>14}"
    )
    print("-" * 92)

    metric_labels = {
        "LogisticRegression": ("Accuracy", "F1", "ROC_AUC"),
        "DecisionTree": ("Accuracy", "F1", "ROC_AUC"),
        "RandomForest": ("Accuracy", "F1", "ROC_AUC"),
        "SVM_Linear": ("Accuracy", "F1", "ROC_AUC"),
        "KMeans": ("Silhouette", "Inertia", None),
        "LinearRegression": ("R2", "RMSE", "MAE"),
    }

    for name, metrics in results.items():
        m1_key, m2_key, m3_key = metric_labels[name]
        m1 = f"{m1_key}={metrics[m1_key]:.4f}"
        m2 = f"{m2_key}={metrics[m2_key]:.4f}" if m2_key else ""
        m3 = f"{m3_key}={metrics[m3_key]:.4f}" if m3_key else ""
        print(
            f"{name:<22}{metrics['train_ms']:>14.2f}{metrics['infer_ms']:>14.2f}"
            f"{m1:>16}{m2:>14}{m3:>14}"
        )
    print("=" * 92)


def main() -> None:
    print("Loading Adult Census Income dataset...")
    adult_features, adult_target = _load_adult()
    print(
        f"  rows={len(adult_features)}, positive_rate={adult_target.mean() * 100:.1f}%"
    )

    adult_train_df, adult_test_df, y_train_cls, y_test_cls = train_test_split(
        adult_features,
        adult_target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=adult_target,
    )

    x_train_cls, x_test_cls = _encode_adult(adult_train_df, adult_test_df)

    scaler_cls = StandardScaler().fit(x_train_cls)
    x_train_cls = scaler_cls.transform(x_train_cls)
    x_test_cls = scaler_cls.transform(x_test_cls)

    print("\nRunning classifiers...")
    cls_results = _run_classifiers(x_train_cls, x_test_cls, y_train_cls, y_test_cls)

    print("Running clustering...")
    clu_results = _run_clustering(x_train_cls, x_test_cls)

    print("\nLoading California Housing dataset...")
    housing_features, housing_target = _load_housing()
    print(f"  rows={len(housing_features)}")

    housing_train_df, housing_test_df, y_train_reg, y_test_reg = train_test_split(
        housing_features,
        housing_target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    x_train_reg, x_test_reg = _encode_housing(housing_train_df, housing_test_df)

    scaler_reg = StandardScaler().fit(x_train_reg)
    x_train_reg = scaler_reg.transform(x_train_reg)
    x_test_reg = scaler_reg.transform(x_test_reg)

    print("\nRunning regression...")
    reg_results = _run_regression(x_train_reg, x_test_reg, y_train_reg, y_test_reg)

    results = {**cls_results, **clu_results, **reg_results}

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    _print_summary(results)
    print(f"\nResults written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
