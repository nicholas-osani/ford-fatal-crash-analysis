"""Decision trees (shallow + tuned) for fatal-crash prediction, with SMOTE for class imbalance.

By Nicholas Osani — CIS 3920 coursework, Baruch College.
Cleaned for portfolio: hardcoded local paths removed, imports tidied. Logic unchanged.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV

# Raw data: NYC Open Data, Motor Vehicle Collisions – Crashes (one row per vehicle).
# Download the CSV and point DATA_PATH at it.
DATA_PATH = "Motor_Vehicle_Collisions_-_Crashes.csv"

# ------------------------------
# Data Loading & Preprocessing
# ------------------------------

# 1. Load the dataset (a DtypeWarning may appear; that's acceptable)
df = pd.read_csv(DATA_PATH, low_memory=False)

# 2. Standardize column names: strip spaces, uppercase, and replace spaces with underscores.
df.columns = df.columns.str.strip().str.upper().str.replace(' ', '_')

# Expected columns include:
# UNIQUE_ID, COLLISION_ID, CRASH_DATE, CRASH_TIME, VEHICLE_TYPE, VEHICLE_MAKE, VEHICLE_MODEL,
# VEHICLE_YEAR, DRIVER_LICENSE_STATUS, CONTRIBUTING_FACTOR_1, CONTRIBUTING_FACTOR_2,
# NUMBER_OF_PERSONS_INJURED, NUMBER_OF_PERSONS_KILLED, BOROUGH, etc.

# 3. Create CRASH_DATETIME by combining CRASH_DATE and CRASH_TIME.
df['CRASH_DATETIME'] = pd.to_datetime(df['CRASH_DATE'] + ' ' + df['CRASH_TIME'], errors='coerce')

# 4. Filter for crashes on or after 2016-04-01.
df = df[df['CRASH_DATETIME'] >= '2016-04-01'].copy()

# 5. Ensure BOROUGH exists; if missing, create with 'Unknown'.
if 'BOROUGH' not in df.columns:
    df['BOROUGH'] = 'Unknown'
else:
    df['BOROUGH'] = df['BOROUGH'].fillna('Unknown')

# 6. Outcome variable: fatal = 1 if NUMBER_OF_PERSONS_KILLED > 0, else 0.
if 'NUMBER_OF_PERSONS_KILLED' in df.columns:
    df['fatal'] = (df['NUMBER_OF_PERSONS_KILLED'] > 0).astype(int)
    df.drop(columns=['NUMBER_OF_PERSONS_KILLED'], inplace=True)
else:
    print("Column NUMBER_OF_PERSONS_KILLED not found. Exiting.")
    exit()

# 7. Ford involvement: 1 if VEHICLE_MAKE contains 'FORD'.
if 'VEHICLE_MAKE' in df.columns:
    df['Ford_involved'] = df['VEHICLE_MAKE'].str.upper().str.contains('FORD', na=False).astype(int)
    df.drop(columns=['VEHICLE_MAKE'], inplace=True)
else:
    print("Column VEHICLE_MAKE not found. Exiting.")
    exit()

# 8. Process DRIVER_LICENSE_STATUS.
if 'DRIVER_LICENSE_STATUS' in df.columns:
    df['DRIVER_LICENSE_STATUS'] = df['DRIVER_LICENSE_STATUS'].fillna('Unknown').str.strip().str.upper()
    # Recode 'PERMIT' as 'UNLICENSED' to combine sparse categories.
    df['DRIVER_LICENSE_STATUS'] = df['DRIVER_LICENSE_STATUS'].replace({'PERMIT': 'UNLICENSED'})
    # Create a flag for unlicensed drivers.
    df['Unlicensed_flag'] = (df['DRIVER_LICENSE_STATUS'] == 'UNLICENSED').astype(int)
else:
    df['DRIVER_LICENSE_STATUS'] = 'Unknown'
    df['Unlicensed_flag'] = 0

# 9. Contributing factor indicators from CONTRIBUTING_FACTOR_1 and CONTRIBUTING_FACTOR_2.
factor_cols = ['CONTRIBUTING_FACTOR_1', 'CONTRIBUTING_FACTOR_2']
existing_factors = [col for col in factor_cols if col in df.columns]
if existing_factors:
    df['combined_factors'] = df[existing_factors].fillna('').agg(' '.join, axis=1).str.upper()
else:
    df['combined_factors'] = ""
df['impair_flag'] = df['combined_factors'].str.contains('ALCOHOL|DRUG', na=False).astype(int)
df['speed_flag'] = df['combined_factors'].str.contains('SPEED', na=False).astype(int)
df['distract_flag'] = df['combined_factors'].str.contains('INATTENTION|DISTRACT|PHONE|ELECTRONIC', na=False).astype(int)

# 10. Vehicle age: use VEHICLE_YEAR to compute age.
if 'VEHICLE_YEAR' in df.columns:
    df['VEHICLE_YEAR'] = pd.to_numeric(df['VEHICLE_YEAR'], errors='coerce')
    df.loc[(df['VEHICLE_YEAR'] < 1900) | (df['VEHICLE_YEAR'] > 2100), 'VEHICLE_YEAR'] = np.nan
    df = df.dropna(subset=['VEHICLE_YEAR']).copy()
    df['vehicle_age'] = df['CRASH_DATETIME'].dt.year - df['VEHICLE_YEAR']
    df.loc[df['vehicle_age'] < 0, 'vehicle_age'] = 0
    df.drop(columns=['VEHICLE_YEAR'], inplace=True)
else:
    print("Column VEHICLE_YEAR not found. Exiting.")
    exit()

# 11. Time-of-day: Create hour_bin from crash hour.
df['hour'] = df['CRASH_DATETIME'].dt.hour
bins = [0, 6, 12, 18, 24]
labels = ['Late_Night', 'Morning', 'Afternoon', 'Evening']
df['hour_bin'] = pd.cut(df['hour'], bins=bins, labels=labels, right=False, include_lowest=True)

# 12. Weekend indicator.
df['weekend'] = df['CRASH_DATETIME'].dt.dayofweek.isin([5,6]).astype(int)

# ------------------------------
# Aggregation to Crash Level
# ------------------------------
# Aggregate data to one row per crash using COLLISION_ID.
agg_dict = {
    'fatal': 'max',                   # if any vehicle fatal, crash is fatal
    'Ford_involved': 'max',           # if any vehicle is Ford
    'Unlicensed_flag': 'max',         # if any driver is unlicensed
    'impair_flag': 'max',
    'speed_flag': 'max',
    'distract_flag': 'max',
    'vehicle_age': 'mean',            # average vehicle age
    'CRASH_DATETIME': 'first',
    'BOROUGH': 'first',
    'hour_bin': 'first',
    'weekend': 'max'
}
if 'COLLISION_ID' in df.columns:
    crash_df = df.groupby('COLLISION_ID').agg(agg_dict).reset_index()
else:
    crash_df = df.copy()

# Handle missing vehicle_age if any.
crash_df['vehicle_age'] = crash_df['vehicle_age'].fillna(crash_df['vehicle_age'].mean())

# ------------------------------
# Prepare Modeling Data
# ------------------------------
# Features to use: Ford_involved, vehicle_age, impairment_flag, speed_flag, distract_flag,
# weekend, Unlicensed_flag, hour_bin, BOROUGH.
model_data = crash_df[['Ford_involved', 'vehicle_age', 'impair_flag', 'speed_flag', 'distract_flag',
                       'weekend', 'Unlicensed_flag', 'hour_bin', 'BOROUGH', 'fatal']].copy()

# One-hot encode categorical variables: hour_bin and BOROUGH.
model_data = pd.get_dummies(model_data, columns=['hour_bin', 'BOROUGH'], drop_first=True)

# Define X (features) and y (target).
X = model_data.drop('fatal', axis=1)
y = model_data['fatal']

# Split into training and testing sets (80/20 split), stratify on y.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("Training set size:", X_train.shape[0], "Testing set size:", X_test.shape[0])
print("Fatal crashes in training:", y_train.sum(), "Fatal crashes in testing:", y_test.sum())

# ------------------------------
# Address Class Imbalance with SMOTE
# ------------------------------
# Apply SMOTE to the training data.
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("After SMOTE, training set size:", X_train_res.shape[0])
print("Fatal crashes in oversampled training:", y_train_res.sum())

# ------------------------------
# Model 1: Shallow Decision Tree (Interpretable)
# ------------------------------

clf_shallow = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)
clf_shallow.fit(X_train_res, y_train_res)
y_pred_shallow = clf_shallow.predict(X_test)
acc_shallow = accuracy_score(y_test, y_pred_shallow)
cm_shallow = confusion_matrix(y_test, y_pred_shallow)
report_shallow = classification_report(y_test, y_pred_shallow, target_names=['Non-fatal', 'Fatal'], zero_division=0)

print("\nShallow Tree Performance:")
print("Accuracy: {:.4f}".format(acc_shallow))
print("Confusion Matrix:\n", cm_shallow)
print("Classification Report:\n", report_shallow)

# Export shallow tree plot.
plt.figure(figsize=(12,8))
plot_tree(clf_shallow, feature_names=X_train_res.columns, class_names=['Non-fatal', 'Fatal'], filled=True)
plt.title("Shallow Decision Tree (max_depth=3)")
plt.savefig("decision_tree_shallow.png", dpi=300)
plt.close()
print("Shallow decision tree plot saved as 'decision_tree_shallow.png'")

# ------------------------------
# Model 2: Tuned Decision Tree (Optimized for Performance)
# ------------------------------

param_grid = {
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
}
clf = DecisionTreeClassifier(class_weight='balanced', random_state=42)
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_res, y_train_res)
best_tree = grid_search.best_estimator_
best_tree.fit(X_train_res, y_train_res)
y_pred_tuned = best_tree.predict(X_test)
acc_tuned = accuracy_score(y_test, y_pred_tuned)
cm_tuned = confusion_matrix(y_test, y_pred_tuned)
report_tuned = classification_report(y_test, y_pred_tuned, target_names=['Non-fatal', 'Fatal'], zero_division=0)

print("\nTuned Tree Performance:")
print("Best Parameters:", grid_search.best_params_)
print("Accuracy: {:.4f}".format(acc_tuned))
print("Confusion Matrix:\n", cm_tuned)
print("Classification Report:\n", report_tuned)

# Export tuned tree plot.
plt.figure(figsize=(16,10))
plot_tree(best_tree, feature_names=X_train_res.columns, class_names=['Non-fatal', 'Fatal'], filled=True)
plt.title("Tuned Decision Tree")
plt.savefig("decision_tree_tuned.png", dpi=300)
plt.close()
print("Tuned decision tree plot saved as 'decision_tree_tuned.png'")
