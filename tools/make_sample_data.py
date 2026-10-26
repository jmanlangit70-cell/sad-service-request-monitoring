"""Generate a SYNTHETIC stand-in for the midterm dataset (300 rows, 14 fields).

It only exists so the app can be demonstrated and tested publicly without real
student records. Replace sample_data/ with your real midterm CSV (or an
anonymised copy) and re-run tools/train_model.py.

Usage:  python tools/make_sample_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 300

service_types = ["Scholarship Inquiry", "Records", "IT Support", "Enrollment", "Certification"]
service_p = [0.247, 0.23, 0.21, 0.16, 0.153]
offices = ["Registrar", "Accounting Office", "IT Office", "Scholarship Office"]
sla_by_service = {"Scholarship Inquiry": [5, 7], "Records": [3, 5, 7], "IT Support": [2, 3, 5],
                  "Enrollment": [3, 5], "Certification": [5, 7, 10]}

dates = pd.to_datetime("2026-01-05") + pd.to_timedelta(rng.integers(0, 170, N), unit="D")
service = rng.choice(service_types, N, p=service_p)
sla = np.array([rng.choice(sla_by_service[s]) for s in service])
queue = rng.integers(0, 30, N)
docs = rng.choice(["Yes", "No"], N, p=[0.75, 0.25])

status = rng.choice(["Completed", "Pending", "Cancelled"], N, p=[0.743, 0.193, 0.064])
# a little signal: shorter SLA and incomplete documents breach more often
p_breach = 0.50 + 0.04 * (sla <= 3) + 0.08 * (docs == "No") - 0.003 * (queue - 15)
breach = rng.random(N) < np.clip(p_breach, 0.15, 0.9)
processing = np.where(breach, sla + rng.integers(1, 6, N), np.maximum(1, sla - rng.integers(0, 3, N)))

df = pd.DataFrame({
    "Request_ID": [f"REQ{i:04d}" for i in range(1, N + 1)],
    "Service_Type": service,
    "Requester_Type": rng.choice(["Student", "Faculty", "Staff"], N, p=[0.7, 0.15, 0.15]),
    "Submission_Date": dates.strftime("%Y-%m-%d"),
    "Requested_Channel": rng.choice(["Portal", "Email", "Walk-in"], N, p=[0.5, 0.25, 0.25]),
    "Priority": rng.choice(["Low", "Normal", "High"], N, p=[0.25, 0.55, 0.2]),
    "Assigned_Office": rng.choice(offices, N),
    "Queue_Length": queue,
    "Required_Documents_Complete": docs,
    "Processing_Days": np.where(status == "Completed", processing, np.nan),
    "SLA_Days": sla,
    "Status": status,
    "Reopened": rng.choice(["Yes", "No"], N, p=[0.08, 0.92]),
    "Satisfaction_Score": np.where(status == "Completed", rng.integers(1, 6, N), np.nan),
})

out = Path(__file__).resolve().parents[1] / "sample_data" / "synthetic_service_requests.csv"
df.to_csv(out, index=False)
print(f"wrote {out} ({len(df)} rows)")
