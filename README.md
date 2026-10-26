# Student Service Request Monitoring System (Streamlit)

Systems Analysis and Design - Final Exercise 1, Section B.
Deploys the midterm SLA-breach Decision Tree as a decision-support web app.

> Predictions are advisory. A staff member reviews every flagged request; the system never
> rejects a request or denies a student service automatically.

## Pages
| Page | What it does |
|---|---|
| Home | Overview, model version, safety notice |
| Dataset | CSV upload, 14-field schema check, preview, null/duplicate summary |
| Data Analysis | Descriptive stats, class distribution, interactive Plotly charts (office, queue category, service type, weekday, queue histogram) |
| Prediction | Form -> one-row DataFrame -> saved pipeline -> class + probability, metrics, limitations |
| Dashboard | KPIs, ranked/capped review queue, staff confirm/override with reason, decision log, exports |
| System Design | 5 functional + 3 non-functional requirements, actors, use case diagram, workflow |

## Repository layout
```
app.py
requirements.txt
modules/        data_processing.py  exploratory_analysis.py  prediction.py  dashboard.py  system_design.py
models/         trained_model.joblib  model_metadata.json
assets/         use_case_diagram.png
sample_data/    synthetic_service_requests.csv   (SYNTHETIC - no real student records)
tools/          make_sample_data.py  train_model.py  make_use_case_diagram.py
tests/          test_core.py
docs/           deployment_report.md
```

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
python tests/test_core.py        # unit tests (no Streamlit needed)
```

## IMPORTANT - use your real midterm data before submitting
The model and sample CSV in this repo were built from **synthetic data** because the midterm
notebook/CSV were not available. To make the app match your submitted midterm:
```bash
python tools/train_model.py --data path/to/your_midterm.csv --source-label "Midterm dataset"
```
This reproduces the midterm settings (Decision Tree, max_depth=4, min_samples_leaf=10,
random_state=42, 80/20 stratified split, nine submission-time features, imputation and
encoding inside the Pipeline) and rewrites `models/trained_model.joblib` and
`models/model_metadata.json`. If your notebook already saved a pipeline with joblib, copy
it to `models/trained_model.joblib` instead and update the metadata JSON (metrics, categories).
Do not commit real student records; use synthetic or anonymised data for public deployment.

`scikit-learn` is pinned to the version used to save the model. If you retrain locally,
pin the version you trained with.

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repository (no passwords or real student data).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Create app -> choose repository and branch -> main file `app.py` -> Deploy.
4. If the build fails, open **Manage app -> Logs** (usually a missing/incompatible package version).

## Leakage controls
`Processing_Days`, `Status`, `Reopened` and `Satisfaction_Score` are never model inputs.
`Processing_Days` only builds the label; `Status` only filters completed rows.
