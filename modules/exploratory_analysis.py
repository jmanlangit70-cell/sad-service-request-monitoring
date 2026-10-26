"""Exploratory Data Analysis page: statistics, class distribution, interactive charts."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from modules.data_processing import QUEUE_LABELS, TARGET, WEEKDAYS, labelled_subset


def breach_rate_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Breach rate per category among completed (labelled) requests."""
    lab = labelled_subset(df)
    g = lab.groupby(col)[TARGET].agg(requests="count", breaches="sum").reset_index()
    g["breach_rate"] = (g["breaches"] / g["requests"]).round(3)
    return g.sort_values("breach_rate", ascending=False)


@st.cache_data
def _describe(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe(include="all").T.astype(str)


def render(df: pd.DataFrame):
    import plotly.express as px

    st.header("Exploratory Data Analysis")
    lab = labelled_subset(df)
    if lab.empty:
        st.error("No completed requests with a known SLA outcome, so breach analysis is not possible.")
        return

    # ---- sidebar-style filters
    f1, f2 = st.columns(2)
    offices = f1.multiselect("Office", sorted(df["Assigned_Office"].dropna().unique()),
                             default=sorted(df["Assigned_Office"].dropna().unique()))
    services = f2.multiselect("Service type", sorted(df["Service_Type"].dropna().unique()),
                              default=sorted(df["Service_Type"].dropna().unique()))
    view = df[df["Assigned_Office"].isin(offices) & df["Service_Type"].isin(services)]
    lab = labelled_subset(view)
    if lab.empty:
        st.warning("No labelled requests match the filters.")
        return

    st.subheader("Descriptive statistics")
    st.dataframe(_describe(view))

    st.subheader("Class distribution (completed requests)")
    dist = lab[TARGET].map({0: "On time", 1: "SLA breached"}).value_counts().reset_index()
    dist.columns = ["Outcome", "Requests"]
    c1, c2 = st.columns([1, 2])
    c1.dataframe(dist, hide_index=True)
    c2.plotly_chart(px.pie(dist, names="Outcome", values="Requests", hole=0.4,
                           title="On time vs SLA breached"))
    rate = lab[TARGET].mean()
    st.caption(f"Overall breach rate: {rate:.1%} of {len(lab)} completed requests. "
               f"Always predicting the majority class already scores "
               f"{max(rate, 1 - rate):.1%}, so accuracy alone is misleading.")

    st.subheader("Interactive charts")
    tab1, tab2, tab3, tab4 = st.tabs(["Breach rate by office", "Breach rate by queue category",
                                      "Breach rate by service type", "Volume by weekday"])
    with tab1:
        t = breach_rate_by(view, "Assigned_Office")
        fig = px.bar(t, x="breach_rate", y="Assigned_Office", orientation="h", text="breach_rate",
                     hover_data=["requests", "breaches"], title="SLA breach rate by assigned office")
        fig.add_vline(x=rate, line_dash="dash", annotation_text=f"overall {rate:.1%}")
        fig.update_layout(xaxis_tickformat=".0%", xaxis_range=[0, 1])
        st.plotly_chart(fig)
        top = t.iloc[0]
        st.write(f"**Meaning:** {top['Assigned_Office']} has the highest breach rate "
                 f"({top['breach_rate']:.1%}, n={int(top['requests'])}). Check that its sample is "
                 "large enough before acting; this is an association, not proof of cause.")
    with tab2:
        t = breach_rate_by(view, "Queue_Category").set_index("Queue_Category").reindex(QUEUE_LABELS).reset_index()
        fig = px.bar(t, x="Queue_Category", y="breach_rate", text="breach_rate",
                     hover_data=["requests", "breaches"], title="SLA breach rate by queue length at submission")
        fig.add_hline(y=rate, line_dash="dash", annotation_text="overall")
        fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1])
        st.plotly_chart(fig)
        st.write("**Meaning:** shows whether queue length at submission works as an early-warning "
                 "signal. Queue categories: Low 0-5, Medium 6-15, High >15.")
    with tab3:
        t = breach_rate_by(view, "Service_Type")
        fig = px.bar(t, x="Service_Type", y="breach_rate", color="requests", text="breach_rate",
                     hover_data=["breaches"], title="SLA breach rate by service type")
        fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1])
        st.plotly_chart(fig)
    with tab4:
        wk = view["Submission_Weekday"].value_counts().reindex(WEEKDAYS).dropna().reset_index()
        wk.columns = ["Weekday", "Requests"]
        st.plotly_chart(px.bar(wk, x="Weekday", y="Requests", title="Requests submitted by weekday"))

    st.subheader("Queue length vs outcome")
    plot = lab.assign(Outcome=lab[TARGET].map({0: "On time", 1: "SLA breached"}))
    st.plotly_chart(px.histogram(plot, x="Queue_Length", color="Outcome", barmode="overlay",
                                 nbins=20, opacity=0.65, title="Queue length at submission by outcome"))
