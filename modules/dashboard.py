"""Decision-support dashboard: KPIs, ranked review queue, human decisions.

Roles (simulated - this demo has no login):
  Office Staff   : view risk scores for their requests, record final decisions
  Administrator  : view all KPIs/performance, record final decisions, export reports
  Requester      : tracks own request status only; never sees risk scores
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from modules.data_processing import TARGET, labelled_subset, pending_subset
from modules.prediction import (HIGH_RISK, LIMITATIONS, load_metadata, load_model,
                                score_dataframe)

ROLES = ["Office Staff", "Administrator", "Requester"]


def compute_kpis(df: pd.DataFrame, scored_pending: pd.DataFrame) -> dict:
    lab = labelled_subset(df)
    return {
        "total": int(len(df)),
        "breach_rate": float(lab[TARGET].mean()) if len(lab) else float("nan"),
        "open": int(len(scored_pending)),
        "flagged": int((scored_pending["Breach_Probability"] >= HIGH_RISK).sum()),
    }


def build_review_queue(scored_pending: pd.DataFrame, cap: int) -> pd.DataFrame:
    """Highest-risk first, capped so staff are not flooded."""
    cols = ["Request_ID", "Service_Type", "Assigned_Office", "Priority", "Queue_Length",
            "SLA_Days", "Required_Documents_Complete", "Breach_Probability", "Risk_Level",
            "Missing_Predictors"]
    q = scored_pending[scored_pending["Breach_Probability"] >= HIGH_RISK]
    return q.sort_values(["Breach_Probability", "Queue_Length"], ascending=False).head(cap)[cols]


def render(df: pd.DataFrame):
    import plotly.express as px

    st.header("Decision-Support Dashboard")
    role = st.sidebar.selectbox("Acting as (demo role)", ROLES)

    if role == "Requester":
        st.info("Requesters can track the status of their own request but cannot see delay-risk "
                "scores or the review queue. Switch role to Office Staff or Administrator.")
        rid = st.text_input("Enter your Request ID")
        if rid:
            hit = df[df["Request_ID"] == rid.strip()]
            if hit.empty:
                st.error("No request found with that ID.")
            else:
                st.dataframe(hit[["Request_ID", "Service_Type", "Assigned_Office", "Status"]],
                             hide_index=True)
        return

    model, meta = load_model(), load_metadata()
    pending = pending_subset(df)
    if pending.empty:
        st.warning("There are no Pending requests in this dataset, so nothing needs scoring.")
        scored = pending.assign(Breach_Probability=[], Risk_Level=[], Predicted_Breach=[], Missing_Predictors=[])
    else:
        scored = score_dataframe(model, pending)
    k = compute_kpis(df, scored)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total requests", k["total"])
    c2.metric("SLA breach rate (completed)", "n/a" if pd.isna(k["breach_rate"]) else f"{k['breach_rate']:.1%}")
    c3.metric("Open queue (Pending)", k["open"])
    c4.metric("Flagged high-risk (Pending)", k["flagged"])
    st.caption(f"Model v{meta['model_version']} | scored {datetime.now():%Y-%m-%d %H:%M} | "
               f"high risk = probability >= {HIGH_RISK:.0%}")

    left, right = st.columns(2)
    vol = df["Service_Type"].value_counts().reset_index()
    vol.columns = ["Service_Type", "Requests"]
    left.plotly_chart(px.bar(vol, x="Requests", y="Service_Type", orientation="h",
                             title="Service volume by type"))
    lab = labelled_subset(df)
    if not lab.empty:
        off = lab.groupby("Assigned_Office")[TARGET].mean().reset_index()
        off.columns = ["Assigned_Office", "Breach rate"]
        right.plotly_chart(px.bar(off, x="Breach rate", y="Assigned_Office", orientation="h",
                                  title="SLA breach rate by office").update_layout(xaxis_tickformat=".0%"))

    st.subheader("Workload: pending requests by office and queue category")
    if not pending.empty:
        st.dataframe(pd.crosstab(pending["Assigned_Office"], pending["Queue_Category"]))

    st.subheader("Risk / review queue (for human review)")
    if k["flagged"] > 0:
        cap = st.slider("Maximum flags shown", 5, 50, 10)
        queue = build_review_queue(scored, cap)
        st.write(f"The model flags **{k['flagged']} of {k['open']}** pending requests. "
                 f"Showing the top {len(queue)} by risk - too many flags to act on blindly.")
        st.dataframe(queue.style.format({"Breach_Probability": "{:.0%}"}), hide_index=True)
        if (queue["Missing_Predictors"] > 0).any():
            st.warning("Some flagged requests have missing fields that were imputed. "
                       "Treat their scores as less reliable.")

        st.subheader("Record final decision")
        with st.form("decision_form", clear_on_submit=True):
            rid = st.selectbox("Request", queue["Request_ID"])
            decision = st.radio("Staff decision", ["Confirm flag - expedite / follow up",
                                                   "Override flag - no action needed"], horizontal=True)
            reason = st.text_area("Reason (required)")
            ok = st.form_submit_button("Save decision")
        if ok:
            if not reason.strip():
                st.error("A reason is required to record a decision.")
            else:
                st.session_state.setdefault("decision_log", []).append({
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Request_ID": rid, "Decision": decision, "Reason": reason.strip(), "Decided_by": role})
                st.success(f"Decision saved for {rid}.")
    else:
        st.success("No pending requests are currently flagged high-risk.")

    log = st.session_state.get("decision_log", [])
    if log:
        st.subheader("Decision log (this session)")
        log_df = pd.DataFrame(log)
        st.dataframe(log_df, hide_index=True)
        if role == "Administrator":
            st.download_button("Export decision log (CSV)", log_df.to_csv(index=False), "decision_log.csv")
    if role == "Administrator" and not scored.empty:
        st.download_button("Export scored pending requests (CSV)",
                           scored[["Request_ID", "Breach_Probability", "Risk_Level"]].to_csv(index=False),
                           "scored_pending.csv")

    st.subheader("Interpretation and recommended human follow-up")
    st.markdown(
        "- **Risk percent** is a statistical estimate, not proof; the midterm model was often wrong.\n"
        "- **Follow up:** contact the requester, check documents, then expedite or reassign.\n"
        "- **Never** deny or reject a request because of a flag alone.\n"
        "- **Missing data** lowers reliability - verify key fields first.")
    with st.expander("Model limitations"):
        for item in LIMITATIONS:
            st.write("- " + item)
