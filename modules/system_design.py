"""SAD Integration page: requirements, actors, use case diagram, decision workflow."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]

FUNCTIONAL = pd.DataFrame([
    ("FR-A", "Display SLA breach rate by assigned office for completed requests so administrators can identify bottleneck offices.",
     "Midterm Q6(b)/Chart 1: Registrar highest at 60.0% (n=55) vs overall 57.8%.", "Data Analysis; Dashboard"),
    ("FR-B", "Record queue length at submission, classify it Low/Medium/High and show breach rate by queue category.",
     "Midterm Q6(c)/Chart 2: Low 67%, Medium 58%, High 53%.", "Data Analysis; Dashboard"),
    ("FR-C", "Flag each new request with a delay-risk indicator calculated from submission-time fields only and route flagged requests to a staff member for review instead of automatically denying them.",
     "Midterm Q9/Q10 and 14a: precision 0.72, recall 0.69; model flags 41 of 58 pending requests.", "Prediction; Dashboard"),
    ("FR-D", "Accept a CSV upload, check it against the 14-field schema and report empty files, missing columns, nulls and duplicates.",
     "Midterm data-quality log: 77 null values in Processing_Days, Satisfaction_Score and SLA_Breached; duplicates must be removed.", "Dataset"),
    ("FR-E", "Let Office Staff or an Administrator confirm or override each flag and record a written reason in a decision log.",
     "Midterm 14c(6): staff need buttons to confirm or override the flag and write a reason.", "Dashboard"),
], columns=["ID", "Requirement", "Evidence", "Implemented in page"])

NON_FUNCTIONAL = pd.DataFrame([
    ("NFR-1 Transparency", "Every prediction shows model version, timestamp, limitations and a reminder that staff decide.",
     "Prediction; Dashboard"),
    ("NFR-2 Performance", "The trained pipeline is loaded once with st.cache_resource and never retrained on refresh, so pages respond quickly.",
     "Prediction"),
    ("NFR-3 Privacy & data protection", "Only synthetic or authorised data is used in public deployment; personal identifiers are removed or masked and no leakage fields reach inference.",
     "All pages"),
], columns=["ID", "Requirement", "Where visible"])

ACTORS = pd.DataFrame([
    ("Requester (Student / Faculty / Staff)", "Submit and track own requests", "No risk scores", "No"),
    ("Office Staff", "Process/update requests, review flagged requests", "Yes - flagged queue and score explanation", "Yes - confirm/override with reason"),
    ("Administrator", "View SLA performance, export reports, upload/validate data", "Yes - all KPIs and scores", "Yes - confirm/override with reason"),
], columns=["Actor", "Main use cases", "Can view prediction results?", "Can record final decision?"])

WORKFLOW_DOT = """
digraph G {
  rankdir=LR; node [shape=box, style=rounded, fontsize=11];
  A [label="Request submitted\\n(submission-time fields)"];
  B [label="Validate & clean data"];
  C [label="Saved pipeline computes\\nbreach probability"];
  D [label="Risk >= 70%?", shape=diamond];
  E [label="Ranked, capped\\nreview queue"];
  F [label="Office Staff / Administrator\\nreviews case"];
  G [label="Confirm flag:\\nexpedite or reassign"];
  H [label="Override flag:\\nno action"];
  I [label="Decision + reason\\nlogged"];
  J [label="Normal processing"];
  A -> B -> C -> D;
  D -> E [label="yes"]; D -> J [label="no"];
  E -> F; F -> G; F -> H; G -> I; H -> I;
}
"""


def render():
    st.header("System Design (SAD Integration)")

    st.subheader("Functional requirements")
    st.dataframe(FUNCTIONAL, hide_index=True)
    st.subheader("Non-functional requirements")
    st.dataframe(NON_FUNCTIONAL, hide_index=True)

    st.subheader("Actors and permissions")
    st.dataframe(ACTORS, hide_index=True)

    st.subheader("Updated use case diagram")
    img = ROOT / "assets" / "use_case_diagram.png"
    if img.exists():
        st.image(str(img))
    else:
        st.error("assets/use_case_diagram.png is missing.")
    st.caption("Changes from the midterm diagram: added Upload & Validate Dataset (FR-D) and "
               "Record Final Decision (FR-E); Review Delay Risk includes Compute Delay Risk Score; "
               "Export Performance Report extends View SLA Performance.")

    st.subheader("Decision workflow")
    st.graphviz_chart(WORKFLOW_DOT)
    st.markdown(
        "1. A request's submission-time fields are validated and scored.\n"
        "2. High-risk requests enter a ranked, capped review queue.\n"
        "3. A staff member investigates and either expedites/reassigns or overrides the flag.\n"
        "4. The decision and reason are logged. **The model never denies or rejects a request.**")
