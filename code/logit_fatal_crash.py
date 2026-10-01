"""Logistic regression: which factors predict a fatal NYC crash?

By Nicholas Osani — CIS 3920 coursework, Baruch College.
Cleaned for portfolio: hardcoded local paths removed, imports tidied. Logic unchanged.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm

# Raw data: NYC Open Data, Motor Vehicle Collisions – Crashes (one row per vehicle).
# Download the CSV and point DATA_PATH at it.
DATA_PATH = "Motor_Vehicle_Collisions_-_Crashes.csv"

# 1. Load the dataset without nested parse_dates to avoid FutureWarnings.
df = pd.read_csv(DATA_PATH, low_memory=False)

# 2. Standardize column names: strip spaces, convert to uppercase, and replace spaces with underscores.
df.columns = df.columns.str.strip().str.upper().str.replace(' ', '_')

# At this point, the CSV columns include:
# UNIQUE_ID, COLLISION_ID, CRASH_DATE, CRASH_TIME, VEHICLE_TYPE, VEHICLE_MAKE, VEHICLE_MODEL,
# VEHICLE_YEAR, DRIVER_LICENSE_STATUS, NUMBER_OF_PERSONS_INJURED, NUMBER_OF_PERSONS_KILLED, etc.

# 3. Create CRASH_DATETIME by combining CRASH_DATE and CRASH_TIME.
df['CRASH_DATETIME'] = pd.to_datetime(df['CRASH_DATE'] + ' ' + df['CRASH_TIME'], errors='coerce')

# 4. Filter data to crashes on or after 2016-04-01.
df = df[df['CRASH_DATETIME'] >= '2016-04-01'].copy()

# 5. Ensure BOROUGH exists; if not, create it.
if 'BOROUGH' not in df.columns:
    df['BOROUGH'] = 'Unknown'
else:
    df['BOROUGH'] = df['BOROUGH'].fillna('Unknown')

# 6. Process VEHICLE_MAKE.
if 'VEHICLE_MAKE' not in df.columns:
    print("No VEHICLE_MAKE column found. Exiting.")
    exit()

# 7. Process VEHICLE_YEAR.
if 'VEHICLE_YEAR' in df.columns:
    df['VEHICLE_YEAR'] = pd.to_numeric(df['VEHICLE_YEAR'], errors='coerce')
    df.loc[(df['VEHICLE_YEAR'] < 1900) | (df['VEHICLE_YEAR'] > 2100), 'VEHICLE_YEAR'] = np.nan
    df = df.dropna(subset=['VEHICLE_YEAR']).copy()
else:
    print("No VEHICLE_YEAR column found. Exiting.")
    exit()

# 8. Create outcome variable 'fatal': 1 if NUMBER_OF_PERSONS_KILLED > 0, else 0.
if 'NUMBER_OF_PERSONS_KILLED' in df.columns:
    df['fatal'] = (df['NUMBER_OF_PERSONS_KILLED'] > 0).astype(int)
    df.drop(columns=['NUMBER_OF_PERSONS_KILLED'], inplace=True)
else:
    print("Column NUMBER_OF_PERSONS_KILLED not found. Exiting.")
    exit()

# 9. Create Ford involvement indicator: 1 if VEHICLE_MAKE contains 'FORD'.
df['Ford_involved'] = df['VEHICLE_MAKE'].str.upper().str.contains('FORD', na=False).astype(int)
df.drop(columns=['VEHICLE_MAKE'], inplace=True)

# 10. Create contributing factor indicators using CONTRIBUTING_FACTOR_1 and CONTRIBUTING_FACTOR_2.
factor_cols = ['CONTRIBUTING_FACTOR_1', 'CONTRIBUTING_FACTOR_2']
existing_factors = [col for col in factor_cols if col in df.columns]
if existing_factors:
    df['combined_factors'] = df[existing_factors].fillna('').agg(' '.join, axis=1).str.upper()
else:
    df['combined_factors'] = ""
df['impairment_involved'] = df['combined_factors'].str.contains('ALCOHOL|DRUG', na=False).astype(int)
df['speeding_involved'] = df['combined_factors'].str.contains('SPEED', na=False).astype(int)
df['distraction_involved'] = df['combined_factors'].str.contains('INATTENTION|DISTRACT|PHONE|ELECTRONIC', na=False).astype(int)

# 11. Create vehicle age: vehicle_age = crash year minus VEHICLE_YEAR.
df['vehicle_age'] = df['CRASH_DATETIME'].dt.year - df['VEHICLE_YEAR']
df.loc[df['vehicle_age'] < 0, 'vehicle_age'] = 0
df.drop(columns=['VEHICLE_YEAR'], inplace=True)

# 12. Create time-of-day features.
df['hour'] = df['CRASH_DATETIME'].dt.hour
bins = [0, 6, 12, 18, 24]
labels = ['Late_Night', 'Morning', 'Afternoon', 'Evening']
df['hour_bin'] = pd.cut(df['hour'], bins=bins, labels=labels, right=False, include_lowest=True)

# 13. Create weekend indicator.
df['weekend'] = df['CRASH_DATETIME'].dt.dayofweek.isin([5, 6]).astype(int)

# 14. Process DRIVER_LICENSE_STATUS.
if 'DRIVER_LICENSE_STATUS' in df.columns:
    df['DRIVER_LICENSE_STATUS'] = df['DRIVER_LICENSE_STATUS'].fillna('Unknown').str.strip()
    # Combine sparse categories: recode 'PERMIT' as 'UNLICENSED'
    df['DRIVER_LICENSE_STATUS'] = df['DRIVER_LICENSE_STATUS'].replace({'PERMIT': 'UNLICENSED'})
else:
    df['DRIVER_LICENSE_STATUS'] = 'Unknown'

# 15. Optionally drop NUMBER_OF_PERSONS_INJURED to reduce collinearity.
if 'NUMBER_OF_PERSONS_INJURED' in df.columns:
    df.drop(columns=['NUMBER_OF_PERSONS_INJURED'], inplace=True)

# Save a copy of the cleaned data (before dummy conversion) for Excel output.
output_df = df[['CRASH_DATETIME', 'BOROUGH', 'fatal', 'Ford_involved', 'vehicle_age',
                'impairment_involved', 'speeding_involved', 'distraction_involved',
                'weekend', 'hour_bin', 'DRIVER_LICENSE_STATUS']].copy()

# 16. Build the modeling dataset.
# Include predictors: Ford_involved, vehicle_age, impairment_involved, speeding_involved, distraction_involved, weekend,
# plus the categorical variables: BOROUGH, hour_bin, and DRIVER_LICENSE_STATUS.
model_df = df.copy()

# 17. Convert categorical variables to dummy variables.
categorical_cols = ['BOROUGH', 'hour_bin', 'DRIVER_LICENSE_STATUS']
model_df = pd.get_dummies(model_df, columns=categorical_cols, drop_first=True)

# 18. Define outcome and predictors.
y = model_df['fatal']
predictor_cols = [
    'Ford_involved', 'vehicle_age', 'impairment_involved', 'speeding_involved',
    'distraction_involved', 'weekend'
]
# Add dummy columns for BOROUGH, hour_bin, and DRIVER_LICENSE_STATUS.
dummy_cols = [col for col in model_df.columns if col.startswith('BOROUGH_') or
              col.startswith('hour_bin_') or col.startswith('DRIVER_LICENSE_STATUS_')]
predictor_cols.extend(dummy_cols)
X = model_df[predictor_cols]

# 19. Add constant and ensure numeric type.
X = sm.add_constant(X)
X = X.astype(float)
y = y.astype(float)

# 20. Fit the logistic regression model using BFGS solver with increased iterations.
model = sm.Logit(y, X)
result = model.fit(method='bfgs', maxiter=300, disp=False)

# 21. Output the model summary and McFadden's pseudo R².
print(result.summary())
print(f"\nMcFadden Pseudo R^2: {result.prsquared:.4f}")

# 22. Save the cleaned (pre-dummy) data to an Excel file.
output_df.to_excel("cleaned_data.xlsx", index=False)
print("\nCleaned dataset saved to 'cleaned_data.xlsx'")


