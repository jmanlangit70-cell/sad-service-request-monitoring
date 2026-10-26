import streamlit as st

st.set_page_config(page_title="Student Service Request Monitoring", layout="wide")

import pandas as pd  # noqa: E402

from modules import dashboard, exploratory_analysis, prediction, system_design  # noqa: E402
from modules.data_processing import (REQUIRED_COLUMNS, LEAKAGE_COLUMNS, prepare_dataset,  # noqa: E402
                                     quality_summary, validate_schema)

SAMPLE_PATH = "sample_data/synthetic_service_requests.csv"
PAGES = ["Home", "Dataset", "Data Analysis", "Prediction", "Dashboard", "System Design"]

page = st.sidebar.selectbox("Module", PAGES)
st.sidebar.caption("Decision-support tool. Staff make every final decision.")


@st.cache_data
def load_sample() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_PATH)


def current_data() -> pd.DataFrame:
    """Prepared dataframe: the uploaded file if valid, otherwise the synthetic sample."""
    if "prepared" in st.session_state:
        return st.session_state["prepared"]
    st.info("No file uploaded yet - using the synthetic sample dataset. Upload a CSV on the Dataset page.")
    return prepare_dataset(load_sample())


if page == "Home":
    st.title("Student Service Request Monitoring System")
    st.write("Streamlit deployment of the midterm SLA-breach model (Decision Tree) for "
             "Systems Analysis and Design, Section B.")
    st.markdown(
        "- **Dataset** - upload and validate a CSV, preview, null/duplicate summary\n"
        "- **Data Analysis** - statistics, class distribution, interactive charts\n"
        "- **Prediction** - score a new request with the saved model\n"
        "- **Dashboard** - KPIs, ranked review queue, staff decisions\n"
        "- **System Design** - requirements, actors, use case diagram, workflow")
    meta = prediction.load_metadata()
    st.caption(f"Model v{meta['model_version']} trained {meta['trained_at']} on: {meta['data_source']}")
    st.warning("Predictions are decision-support outputs, not authorisation to reject a request or "
               "deny a student service. Public deployment uses synthetic or authorised data only.")

elif page == "Dataset":
    st.header("Dataset Upload and Validation")
    st.caption("Required columns: " + ", ".join(REQUIRED_COLUMNS))
    up = st.file_uploader("Upload CSV", type="csv")
    if up is not None:
        try:
            raw = pd.read_csv(up)
        except pd.errors.EmptyDataError:
            st.error("The file is empty or has no columns.")
            st.stop()
        except Exception as exc:  # malformed CSV
            st.error(f"Could not read the file as CSV: {exc}")
            st.stop()
        check = validate_schema(raw)
        for w in check["warnings"]:
            st.warning(w)
        if not check["ok"]:
            for e in check["errors"]:
                st.error(e)
            st.stop()
        st.success("Schema check passed.")
        st.session_state["prepared"] = prepare_dataset(raw)
        st.session_state["raw"] = raw
    elif "raw" not in st.session_state:
        st.info("Showing the synthetic sample dataset until you upload a file.")
        st.session_state["raw"] = load_sample()
        st.session_state["prepared"] = prepare_dataset(load_sample())

    raw = st.session_state["raw"]
    q = quality_summary(raw)
    a, b, c, d = st.columns(4)
    a.metric("Rows", q["rows"]); b.metric("Columns", q["columns"])
    c.metric("Exact duplicate rows", q["exact_duplicate_rows"])
    d.metric("Repeated Request_IDs", q["repeated_request_ids"])
    st.subheader("Preview (first 20 rows)")
    st.dataframe(raw.head(20))
    st.subheader("Null values")
    if q["null_counts"]:
        st.dataframe(pd.Series(q["null_counts"], name="nulls"))
        st.caption("Processing_Days / Satisfaction_Score are empty for Pending and Cancelled requests; "
                   "this is expected and they are never used as predictors.")
    else:
        st.write("No null values.")
    st.caption("Leakage fields excluded from the model: " + ", ".join(LEAKAGE_COLUMNS))

elif page == "Data Analysis":
    exploratory_analysis.render(current_data())

elif page == "Prediction":
    prediction.render()

elif page == "Dashboard":
    dashboard.render(current_data())

else:
    system_design.render()
