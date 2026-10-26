"""Unit tests for the non-UI logic. Run:  python tests/test_core.py  (pytest also works)

If streamlit is not installed, a tiny stub is injected so the page modules can be imported
and their pure functions tested. With streamlit installed the real package is used.
"""
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import streamlit  # noqa: F401
except ImportError:
    stub = types.ModuleType("streamlit")
    stub.cache_resource = lambda f: f
    stub.cache_data = lambda f: f
    sys.modules["streamlit"] = stub

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from modules import dashboard, exploratory_analysis, prediction
from modules.data_processing import (FEATURES, LEAKAGE_COLUMNS, REQUIRED_COLUMNS, TARGET,
                                     clean_data, labelled_subset, pending_subset,
                                     prepare_dataset, quality_summary, validate_schema)

RAW = pd.read_csv(ROOT / "sample_data" / "synthetic_service_requests.csv")
DF = prepare_dataset(RAW)
MODEL = joblib.load(ROOT / "models" / "trained_model.joblib")
META = prediction.load_metadata()

VALID = {"Service_Type": "Records", "Requester_Type": "Student", "Requested_Channel": "Portal",
         "Priority": "Normal", "Assigned_Office": "Registrar", "Required_Documents_Complete": "Yes",
         "Submission_Weekday": "Monday", "Queue_Length": 10, "SLA_Days": 5}


# ---- upload validation
def test_valid_file_passes():
    assert validate_schema(RAW)["ok"]


def test_missing_column_reports_name():
    res = validate_schema(RAW.drop(columns=["Priority", "SLA_Days"]))
    assert not res["ok"] and "Priority" in res["errors"][0] and "SLA_Days" in res["errors"][0]


def test_empty_file_rejected():
    assert not validate_schema(pd.DataFrame(columns=REQUIRED_COLUMNS))["ok"]


def test_quality_summary_counts_dupes_and_nulls():
    dup = pd.concat([RAW, RAW.iloc[[0]]], ignore_index=True)
    q = quality_summary(dup)
    assert q["exact_duplicate_rows"] == 1 and q["repeated_request_ids"] == 1
    assert "Processing_Days" in q["null_counts"]


# ---- cleaning / target
def test_cleaning_dedupes_and_normalises():
    dirty = RAW.copy()
    dirty.loc[0, "Required_Documents_Complete"] = " y "
    dirty.loc[1, "Service_Type"] = "  records "
    dirty = pd.concat([dirty, dirty.iloc[[2]]], ignore_index=True)
    out = clean_data(dirty)
    assert len(out) == 300 and out["Request_ID"].is_unique
    assert out.loc[0, "Required_Documents_Complete"] == "Yes"
    assert out.loc[1, "Service_Type"] == "Records"


def test_target_definition_and_scope():
    lab = labelled_subset(DF)
    assert set(lab["Status"]) == {"Completed"}
    expected = (lab["Processing_Days"] > lab["SLA_Days"]).astype(int)
    assert (lab[TARGET].astype(int) == expected).all()
    assert DF.loc[DF["Status"] != "Completed", TARGET].isna().all()


# ---- leakage
def test_no_leakage_features():
    assert not set(FEATURES) & set(LEAKAGE_COLUMNS) and TARGET not in FEATURES
    assert list(MODEL.feature_names_in_) == FEATURES


# ---- inference
def test_one_row_has_exact_training_columns():
    row = prediction.build_input_row(VALID)
    assert list(row.columns) == FEATURES and len(row) == 1


def test_prediction_matches_pipeline_and_probability():
    res = prediction.predict_one(MODEL, VALID)
    assert res["label"] == int(MODEL.predict(res["input"])[0])
    assert 0 <= res["prob"] <= 1 and res["risk"] in ("Low", "Medium", "High")


def test_invalid_inputs_rejected():
    bad = {**VALID, "Service_Type": "Cafeteria", "Queue_Length": -3, "SLA_Days": 0}
    problems = prediction.validate_inputs(bad, META)
    assert len(problems) == 3
    assert prediction.validate_inputs(VALID, META) == []
    assert prediction.validate_inputs({**VALID, "Queue_Length": "abc"}, META)


def test_missing_input_key_raises():
    try:
        prediction.build_input_row({k: v for k, v in VALID.items() if k != "Priority"})
    except ValueError:
        return
    raise AssertionError("expected ValueError for a missing input")


def test_consistent_with_saved_metrics():
    """Re-evaluating the saved model on the same split reproduces the stored test F1."""
    lab = labelled_subset(DF)
    _, X_te, _, y_te = train_test_split(lab[FEATURES], lab[TARGET].astype(int), test_size=0.2,
                                        stratify=lab[TARGET].astype(int), random_state=42)
    assert round(f1_score(y_te, MODEL.predict(X_te)), 3) == META["metrics"]["f1"]


def test_missing_values_are_imputed_not_crash():
    row = {**VALID, "Queue_Length": np.nan}
    assert prediction.predict_one(MODEL, row)["prob"] is not None


# ---- dashboard / EDA logic
def test_scoring_and_review_queue():
    scored = prediction.score_dataframe(MODEL, pending_subset(DF))
    assert {"Breach_Probability", "Risk_Level"} <= set(scored.columns)
    k = dashboard.compute_kpis(DF, scored)
    assert k["total"] == len(DF) and k["open"] == len(scored)
    q = dashboard.build_review_queue(scored, 5)
    assert len(q) <= 5 and (q["Breach_Probability"] >= prediction.HIGH_RISK).all()
    assert q["Breach_Probability"].is_monotonic_decreasing


def test_breach_rate_table_matches_manual():
    t = exploratory_analysis.breach_rate_by(DF, "Assigned_Office")
    lab = labelled_subset(DF)
    manual = lab.groupby("Assigned_Office")[TARGET].mean().round(3)
    for _, r in t.iterrows():
        assert r["breach_rate"] == manual[r["Assigned_Office"]]


if __name__ == "__main__":  # works with or without pytest installed
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print("FAIL", name, "->", repr(exc))
    print(f"{failed} failed")
    sys.exit(1 if failed else 0)
