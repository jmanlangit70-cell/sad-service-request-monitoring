"""Re-create the midterm model and save it ONCE with joblib (never retrained in the app).

Midterm settings reproduced from the report:
  * Decision Tree, max_depth=4, min_samples_leaf=10, random_state=42
  * 80/20 stratified split, positive class = SLA breached
  * nine submission-time features; imputation + one-hot encoding live INSIDE the
    Pipeline and are fitted on the training split only
  * leakage fields (Processing_Days, Status, Reopened, Satisfaction_Score) excluded

Usage:
    python tools/train_model.py                      # synthetic demo data
    python tools/train_model.py --data my_midterm.csv  # YOUR real midterm CSV
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules.data_processing import (CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES,  # noqa: E402
                                     TARGET, WEEKDAYS, labelled_subset, prepare_dataset)

MODEL_VERSION = "1.0.0"


def build_pipeline() -> Pipeline:
    prep = ColumnTransformer([
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL_FEATURES),
        ("num", Pipeline([("impute", SimpleImputer(strategy="median"))]), NUMERIC_FEATURES),
    ])
    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=42)
    return Pipeline([("prep", prep), ("model", tree)])


def _importance_by_feature(pipe: Pipeline) -> dict:
    names = pipe.named_steps["prep"].get_feature_names_out()
    imp = pipe.named_steps["model"].feature_importances_
    out = {f: 0.0 for f in FEATURES}
    for name, value in zip(names, imp):
        original = name.split("__", 1)[1]
        for f in FEATURES:
            if original == f or original.startswith(f + "_"):
                out[f] += float(value)
                break
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "sample_data" / "synthetic_service_requests.csv"))
    ap.add_argument("--source-label", default=None,
                    help="Text shown in the app, e.g. 'Midterm dataset' or 'Synthetic demo data'")
    args = ap.parse_args()

    raw = pd.read_csv(args.data)
    df = labelled_subset(prepare_dataset(raw))
    X, y = df[FEATURES], df[TARGET].astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pipe = build_pipeline().fit(X_tr, y_tr)
    pred = pipe.predict(X_te)

    majority = int(y_tr.mode()[0])
    base = [majority] * len(y_te)
    cm = confusion_matrix(y_te, pred, labels=[0, 1])

    categories = {c: sorted(df[c].dropna().unique().tolist()) for c in CATEGORICAL_FEATURES
                  if c != "Submission_Weekday"}
    categories["Submission_Weekday"] = WEEKDAYS
    meta = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "algorithm": "DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=42)",
        "data_source": args.source_label or ("Synthetic demo data" if "synthetic" in args.data else Path(args.data).name),
        "features": FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "excluded_leakage_fields": ["Processing_Days", "Status", "Reopened", "Satisfaction_Score"],
        "categories": categories,
        "numeric_ranges": {"Queue_Length": [0, 200], "SLA_Days": [1, 60]},
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "metrics": {
            "accuracy": round(accuracy_score(y_te, pred), 3),
            "precision": round(precision_score(y_te, pred, zero_division=0), 3),
            "recall": round(recall_score(y_te, pred, zero_division=0), 3),
            "f1": round(f1_score(y_te, pred, zero_division=0), 3),
        },
        "baseline_metrics": {
            "accuracy": round(accuracy_score(y_te, base), 3),
            "precision": round(precision_score(y_te, base, zero_division=0), 3),
            "recall": round(recall_score(y_te, base, zero_division=0), 3),
            "f1": round(f1_score(y_te, base, zero_division=0), 3),
        },
        "confusion_matrix": cm.tolist(),
        "feature_importance": _importance_by_feature(pipe),
        # Figures printed in the submitted midterm report, shown for comparison.
        "midterm_report_metrics": {"accuracy": 0.667, "precision": 0.72, "recall": 0.692,
                                   "f1": 0.706, "baseline_f1": 0.732,
                                   "confusion_matrix": [[12, 7], [8, 18]],
                                   "n_train": 178, "n_test": 45},
    }

    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(pipe, ROOT / "models" / "trained_model.joblib")
    (ROOT / "models" / "model_metadata.json").write_text(json.dumps(meta, indent=2))
    print("saved models/trained_model.joblib and models/model_metadata.json")
    print(json.dumps({k: meta[k] for k in ("n_train", "n_test", "metrics", "baseline_metrics",
                                            "confusion_matrix")}, indent=2))


if __name__ == "__main__":
    main()
