# Crash Claim vs. Data Reality: Do Fords Cause More Fatal Crashes?

CIS 3920 coursework — Baruch College. **By Nicholas Osani.**

## The question

Media coverage reported that "the auto make involved in the most fatal crashes was Ford,"
implying Ford vehicles cause deadlier accidents. This project tests that claim against
the data instead of the headline.

## Data

NYC Open Data — Motor Vehicle Collisions (Crashes), ≈89,000 vehicle-level records,
≈48,000 crashes after aggregating to one row per collision. Only about 50 crashes were
fatal (≈0.06%) — an extremely imbalanced problem, which shapes every modeling choice.

## Approach

Two interpretable models, same verdict sought from both angles:

1. **Decision tree classifier** (70/30 train/test split, `class_weight='balanced'` for the
   rare fatal cases). Feature importances: male driver ~25%, vehicle occupant count ~24%,
   driver licensed ~16%, summer season ~15%, spring ~8%, winter ~7% — and **Ford involvement
   ~5%**, near the bottom.
2. **Logistic regression** as a second lens: the Ford coefficient came back with p > 0.05
   (not significant), while driver gender, license status, occupant count, and seasonal
   factors were significant — consistent with the tree.

Feature engineering along the way: Ford indicator, vehicle age, impairment / speeding /
distraction flags from contributing-factor text, time-of-day bins, weekend indicator,
borough and license-status dummies, crash-level aggregation (max for flags, mean for
vehicle age), and SMOTE oversampling for the tree models.

## Findings

No evidence that driving a Ford *causes* more fatalities. The raw association is
confounding, not causation: male drivers, unlicensed drivers, crowded vehicles, and
summer driving explain the pattern. Bottom line: it's not the car — it's the driver
and the context.

## Limitations (stated in the report, and they matter)

Observational data — associations are not causal. Only ~50 fatal cases, so estimates
carry wide uncertainty. NYC-only; patterns may not generalize.

## Tools

Python — pandas, NumPy, statsmodels, scikit-learn, imbalanced-learn (SMOTE), matplotlib.

## Files

- `Ford_Fatal_Crash_Report.docx` — the full written report (findings, diagnostics, caveats)
- `Final_Presentation_Poster.pdf` — the one-page final presentation poster (original course
  submission artifact)
- `code/clean_collisions.py` — data cleaning: target definition, feature engineering,
  crash-level aggregation
- `code/logit_fatal_crash.py` — logistic regression pipeline (BFGS fit, McFadden pseudo-R²)
- `code/tree_fatal_crash.py` — shallow + GridSearchCV-tuned decision trees with SMOTE

Note: the three scripts are related modeling code on the same NYC dataset, cleaned for
portfolio (hardcoded local paths removed, imports tidied; logic unchanged). The raw
`Motor_Vehicle_Collisions_-_Crashes.csv` is not included — it's NYC Open Data, download
it and point `DATA_PATH` at it to run.
