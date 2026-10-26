"""Data loading, validation, cleaning and feature engineering.

Reused from the midterm notebook (Student Service Request Monitoring):
  * 14 expected raw fields
  * engineered fields: Submission_Weekday, Queue_Category, SLA_Breached
  * target: SLA_Breached = 1 if Processing_Days > SLA_Days (Completed requests only)

This module deliberately has no Streamlit import so it can be unit-tested and
re-used by tools/train_model.py.
"""
from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------- schema ----
REQUIRED_COLUMNS = [
    "Request_ID", "Service_Type", "Requester_Type", "Submission_Date",
    "Requested_Channel", "Priority", "Assigned_Office", "Queue_Length",
    "Required_Documents_Complete", "Processing_Days", "SLA_Days", "Status",
    "Reopened", "Satisfaction_Score",
]

# Submission-time predictors used by the midterm Decision Tree (nine features).
CATEGORICAL_FEATURES = [
    "Service_Type", "Requester_Type", "Requested_Channel", "Priority",
    "Assigned_Office", "Required_Documents_Complete", "Submission_Weekday",
]
NUMERIC_FEATURES = ["Queue_Length", "SLA_Days"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Fields that only exist after work starts/ends. They must never reach inference.
LEAKAGE_COLUMNS = ["Processing_Days", "Status", "Reopened", "Satisfaction_Score"]
TARGET = "SLA_Breached"

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
QUEUE_BINS = [-float("inf"), 5, 15, float("inf")]
QUEUE_LABELS = ["Low", "Medium", "High"]


# ------------------------------------------------------------ validation ----
def validate_schema(df: pd.DataFrame) -> dict:
    """Return {'ok': bool, 'errors': [...], 'warnings': [...]} for a raw upload."""
    errors, warnings = [], []
    if df is None or df.empty:
        return {"ok": False, "errors": ["The file is empty (no data rows)."], "warnings": []}

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append("Missing required column(s): " + ", ".join(missing))
    extra = [c for c in df.columns
             if c not in REQUIRED_COLUMNS
             and c not in ("Submission_Weekday", "Queue_Category", TARGET)]
    if extra:
        warnings.append("Unrecognised column(s) will be ignored: " + ", ".join(extra))

    if not missing:
        if df["Request_ID"].isna().all():
            errors.append("Request_ID is entirely empty.")
        if len(df) < 10:
            warnings.append(f"Only {len(df)} rows - charts and metrics will be unreliable.")
    return {"ok": not errors, "errors": errors, "warnings": warnings}


def quality_summary(df: pd.DataFrame) -> dict:
    """Null / duplicate summary for the Dataset page."""
    nulls = df.isna().sum()
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "null_counts": nulls[nulls > 0].astype(int).to_dict(),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "repeated_request_ids": int(df["Request_ID"].duplicated().sum())
        if "Request_ID" in df.columns else 0,
    }


# -------------------------------------------------------------- cleaning ----
def _yes_no(value):
    if pd.isna(value):
        return pd.NA
    v = str(value).strip().lower()
    if v in ("y", "yes", "true", "1"):
        return "Yes"
    if v in ("n", "no", "false", "0"):
        return "No"
    return str(value).strip()


def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the midterm cleaning rules (see data-quality log in the report)."""
    df = raw.copy()

    # 1. exact duplicates, then repeated IDs (keep first occurrence)
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset="Request_ID", keep="first")

    # 2. trim whitespace on text columns
    text_cols = ["Service_Type", "Requester_Type", "Requested_Channel", "Priority",
                 "Assigned_Office", "Status", "Required_Documents_Complete", "Reopened"]
    for col in text_cols:
        df[col] = df[col].astype("string").str.strip()

    # 3. unify case/spelling to the most frequent form
    for col in ["Service_Type", "Requester_Type", "Requested_Channel", "Priority",
                "Assigned_Office", "Status"]:
        key = df[col].str.lower()
        canonical = df.groupby(key)[col].agg(lambda s: s.value_counts().idxmax())
        df[col] = key.map(canonical)

    # 4. Y/N variants -> Yes/No
    for col in ["Required_Documents_Complete", "Reopened"]:
        df[col] = df[col].map(_yes_no).astype("string")

    # 5. dates: unparseable or future -> NaT
    df["Submission_Date"] = pd.to_datetime(df["Submission_Date"], errors="coerce")
    df.loc[df["Submission_Date"] > pd.Timestamp.today().normalize(), "Submission_Date"] = pd.NaT

    # 6. numbers: non-numeric / out-of-range -> missing
    for col in ["Queue_Length", "Processing_Days", "SLA_Days", "Satisfaction_Score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.loc[df["Queue_Length"] < 0, "Queue_Length"] = float("nan")
    df.loc[df["Processing_Days"] < 0, "Processing_Days"] = float("nan")
    df.loc[df["SLA_Days"] <= 0, "SLA_Days"] = float("nan")
    df.loc[df["Satisfaction_Score"] < 0, "Satisfaction_Score"] = float("nan")

    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Submission_Weekday, Queue_Category and the SLA_Breached target."""
    out = df.copy()
    out["Submission_Weekday"] = out["Submission_Date"].dt.day_name()
    out["Queue_Category"] = pd.cut(out["Queue_Length"], bins=QUEUE_BINS,
                                   labels=QUEUE_LABELS).astype("string")

    labelled = (out["Status"] == "Completed") & out["Processing_Days"].notna() & out["SLA_Days"].notna()
    out[TARGET] = pd.NA
    out.loc[labelled, TARGET] = (out.loc[labelled, "Processing_Days"]
                                 > out.loc[labelled, "SLA_Days"]).astype(int)
    out[TARGET] = out[TARGET].astype("Int64")
    return out


def prepare_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw upload -> cleaned + engineered dataframe."""
    return engineer_features(clean_data(raw))


def labelled_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Completed requests with a known target (used for EDA rates and training)."""
    return df[df[TARGET].notna()].copy()


def pending_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Open requests - the ones the dashboard scores for human review."""
    return df[df["Status"] == "Pending"].copy()
