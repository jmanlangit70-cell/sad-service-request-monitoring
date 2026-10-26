"""Model loading, one-row inference, batch scoring and the Prediction page.

The model is loaded with @st.cache_resource and never retrained in the app.
The core functions below do not need Streamlit, so they can be unit-tested.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from modules.data_processing import FEATURES, WEEKDAYS

try:  # Streamlit is optional for unit tests
    import streamlit as st
    _cache_resource = st.cache_resource
except ImportError:  # pragma: no cover
    st = None

    def _cache_resource(func):
        return func

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "trained_model.joblib"
META_PATH = ROOT / "models" / "model_metadata.json"

# Risk bands applied to P(breach). The midterm model is weak, so these are
# review-priority labels, not decisions.
HIGH_RISK = 0.70
MEDIUM_RISK = 0.50

LIMITATIONS = [
    "Trained on historical requests only; the split was random, not time-based, so it may not hold on newer months.",
    "The midterm test set had only 45 rows, so the reported metrics are noisy.",
    "On the midterm run the tree did not beat the majority-class baseline on breach-class F1, "
    "so scores are advisory triage hints only.",
    "Importance shows what the tree split on, not what causes delays.",
    "The model cannot see special cases (for example a student's urgent need).",
]


@_cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@_cache_resource
def load_metadata() -> dict:
    return json.loads(META_PATH.read_text())


# ------------------------------------------------------------------ logic ----
def risk_label(prob: float) -> str:
    if prob >= HIGH_RISK:
        return "High"
    if prob >= MEDIUM_RISK:
        return "Medium"
    return "Low"


def build_input_row(values: dict) -> pd.DataFrame:
    """One-row DataFrame with the exact training feature names and order."""
    missing = [f for f in FEATURES if f not in values]
    if missing:
        raise ValueError("Missing input(s): " + ", ".join(missing))
    row = pd.DataFrame([{f: values[f] for f in FEATURES}], columns=FEATURES)
    row["Queue_Length"] = pd.to_numeric(row["Queue_Length"], errors="raise")
    row["SLA_Days"] = pd.to_numeric(row["SLA_Days"], errors="raise")
    return row


def validate_inputs(values: dict, meta: dict) -> list[str]:
    """Return a list of human-readable problems (empty list = valid)."""
    problems = []
    for col, allowed in meta["categories"].items():
        if values.get(col) in (None, ""):
            problems.append(f"{col} is required.")
        elif values[col] not in allowed:
            problems.append(f"{col}: '{values[col]}' was not seen in training data.")
    for col, (lo, hi) in meta["numeric_ranges"].items():
        v = values.get(col)
        try:
            v = float(v)
        except (TypeError, ValueError):
            problems.append(f"{col} must be a number.")
            continue
        if not lo <= v <= hi:
            problems.append(f"{col} must be between {lo} and {hi}.")
    return problems


def predict_one(model, values: dict) -> dict:
    row = build_input_row(values)
    label = int(model.predict(row)[0])
    result = {"label": label, "prob": None, "input": row}
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        result["prob"] = float(model.predict_proba(row)[0][classes.index(1)])
        result["risk"] = risk_label(result["prob"])
    return result


def score_dataframe(model, df: pd.DataFrame) -> pd.DataFrame:
    """Score requests using submission-time features only (no leakage columns)."""
    scored = df.copy()
    X = scored[FEATURES]
    scored["Breach_Probability"] = model.predict_proba(X)[:, list(model.classes_).index(1)]
    scored["Risk_Level"] = scored["Breach_Probability"].map(risk_label)
    scored["Predicted_Breach"] = model.predict(X).astype(int)
    scored["Missing_Predictors"] = X.isna().sum(axis=1)
    return scored


# ------------------------------------------------------------------- page ----
def render():
    meta = load_metadata()
    model = load_model()

    st.header("Prediction - SLA breach risk for a new request")
    st.caption(f"Model v{meta['model_version']} | {meta['algorithm']} | "
               f"trained {meta['trained_at']} | data: {meta['data_source']}")
    st.warning("Predictions are decision-support only. A staff member reviews every flagged "
               "request; the system never rejects or denies a request automatically.")

    cats = meta["categories"]
    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        values = {
            "Service_Type": c1.selectbox("Service type", cats["Service_Type"]),
            "Requester_Type": c1.selectbox("Requester type", cats["Requester_Type"]),
            "Requested_Channel": c1.selectbox("Requested channel", cats["Requested_Channel"]),
            "Priority": c2.selectbox("Priority", cats["Priority"]),
            "Assigned_Office": c2.selectbox("Assigned office", cats["Assigned_Office"]),
            "Required_Documents_Complete": c2.selectbox("Required documents complete?",
                                                        cats["Required_Documents_Complete"]),
            "Submission_Weekday": c3.selectbox("Submission weekday", WEEKDAYS),
            "Queue_Length": c3.number_input("Queue length at submission", min_value=-5, max_value=500,
                                            value=10, step=1),
            "SLA_Days": c3.number_input("SLA target (days)", min_value=-5, max_value=100, value=5, step=1),
        }
        submitted = st.form_submit_button("Predict")

    if submitted:
        problems = validate_inputs(values, meta)
        if problems:
            for p in problems:
                st.error(p)
        else:
            res = predict_one(model, values)
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            a, b, c = st.columns(3)
            a.metric("Predicted class", "SLA breach likely" if res["label"] == 1 else "On time likely")
            if res["prob"] is not None:
                b.metric("Breach probability", f"{res['prob']:.0%}")
                c.metric("Review priority", res["risk"])
            st.caption(f"Prediction timestamp: {stamp} | Model version: {meta['model_version']}")
            st.info("Next step: route to Office Staff for review. Staff confirm or override "
                    "the flag on the Dashboard page and record a reason.")
            with st.expander("Exact one-row input sent to the saved pipeline"):
                st.dataframe(res["input"])
            st.session_state["last_prediction"] = {**values, "Breach_Probability": res["prob"],
                                                   "Timestamp": stamp}

    st.subheader("Model metrics (held-out test set)")
    m, base, rep = meta["metrics"], meta["baseline_metrics"], meta["midterm_report_metrics"]
    table = pd.DataFrame({
        "This saved model": m,
        "Majority-class baseline": base,
        "Midterm report": {k: rep[k] for k in ("accuracy", "precision", "recall", "f1")},
    }).T
    st.dataframe(table)
    if meta["data_source"].lower().startswith("synthetic"):
        st.caption("The saved model in this deployment was trained on SYNTHETIC demo data, so its "
                   "numbers differ from the midterm report. Re-run tools/train_model.py with the "
                   "real midterm CSV to make them match.")
    cm = pd.DataFrame(meta["confusion_matrix"], index=["Actual on time", "Actual breached"],
                      columns=["Predicted on time", "Predicted breached"])
    st.write("Confusion matrix (test set)")
    st.dataframe(cm)
    st.write("Feature importance (share of impurity reduction)")
    st.bar_chart(pd.Series(meta["feature_importance"]).sort_values())

    with st.expander("Limitations"):
        for item in LIMITATIONS:
            st.write("- " + item)
