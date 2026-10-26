# Deployment Report - Student Service Request Monitoring (Section B)

**Name:** Manlangit, Janpol  **Course:** Systems Analysis and Design
**GitHub repository:** `<paste URL>`  **Deployed app:** `<paste URL>`

## 1. Reused code (from the midterm)
- Cleaning rules from the data-quality log: trim/standardise text, Y/N -> Yes/No, bad dates/numbers -> missing, drop exact duplicates and repeated Request_ID (keep first).
- Feature engineering: `Submission_Weekday`, `Queue_Category` (Low 0-5, Medium 6-15, High >15), target `SLA_Breached = Processing_Days > SLA_Days` for Completed requests only.
- Model: Decision Tree (max_depth=4, min_samples_leaf=10, random_state=42), 80/20 stratified split, nine submission-time features, imputation + one-hot encoding inside one scikit-learn Pipeline.
- Leakage controls: `Processing_Days`, `Status`, `Reopened`, `Satisfaction_Score` excluded from inputs.
- Requirements FR-A, FR-B, FR-C and the use case diagram from Part V of the midterm.

## 2. Changes made for deployment
- Saved the fitted pipeline once with joblib; the app loads it with `@st.cache_resource` and never retrains.
- Added FR-D (CSV upload/validation) and FR-E (staff confirm/override with reason) with matching use cases (Upload & Validate Dataset, Record Final Decision).
- Added three non-functional requirements (transparency, performance, privacy).
- Ranked and capped the review queue, because the midterm showed 41 of 58 pending requests flagged.
- Risk bands on P(breach): High >= 70%, Medium 50-69%, Low < 50% (review-priority labels only).

## 3. Differences from the midterm (explain these)
- `<Delete this line if you retrained on your real CSV.>` The model/sample data in the first build were synthetic, so metrics differ from the midterm report (accuracy 0.667, precision 0.72, recall 0.692, F1 0.706; baseline F1 0.732).

## 4. Tests performed
| Test | Result |
|---|---|
| Valid CSV passes schema check | Pass (automated) |
| Missing columns -> error names the columns | Pass (automated) |
| Empty file rejected | Pass (automated) |
| Duplicate/null counts correct | Pass (automated) |
| Cleaning dedupes and normalises categories | Pass (automated) |
| Target = Processing_Days > SLA_Days, Completed only | Pass (automated) |
| No leakage fields in model features | Pass (automated) |
| One-row input has exact training columns | Pass (automated) |
| Prediction equals pipeline output; probability in [0,1] | Pass (automated) |
| Invalid inputs (unknown category, negative queue, SLA <= 0, text in number field) rejected | Pass (automated) |
| Missing values imputed instead of crashing | Pass (automated) |
| Saved model reproduces stored test F1 on the same split | Pass (automated) |
| Review queue sorted, capped, only high-risk | Pass (automated) |
| Breach-rate tables match manual calculation | Pass (automated) |
| Deployed URL opens; all six pages render; charts respond to filters | `<test on Streamlit Cloud and record>` |
| Screenshots of five modules and five sample predictions | `<attach>` |

Automated tests: `python tests/test_core.py` (15 tests).

## 5. Limitations
- Small labelled set (about 220 completed requests) and a ~45-row test set, so metrics are noisy.
- Random, not time-based, split; the model needs re-validation on newer months.
- On the midterm run the tree did not beat the majority-class baseline on breach-class F1, so it is a triage hint only.
- Feature importance shows what the tree used, not what causes delays.
- Roles in the dashboard are simulated; there is no real authentication.
- The decision log lives in the browser session only; a real system needs a database.

## 6. Reflection
`<Write 3-5 sentences in your own words: what was hardest (e.g. keeping leakage fields out of inference), what you learned about turning a notebook model into an information-system workflow, and why human review matters when the model is only slightly better than guessing.>`
