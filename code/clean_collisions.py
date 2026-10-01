"""Clean the NYC motor-vehicle collision data for fatal-crash modeling.

By Nicholas Osani — CIS 3920 coursework, Baruch College.
Cleaned for portfolio: hardcoded local paths removed, imports tidied. Logic unchanged.
"""

import pandas as pd
import numpy as np

# Raw data: NYC Open Data, Motor Vehicle Collisions – Crashes (one row per vehicle).
# Download the CSV and point DATA_PATH at it.
DATA_PATH = 'Motor_Vehicle_Collisions_-_Crashes.csv'

# Load the dataset (update the file path as needed)
df = pd.read_csv(DATA_PATH, low_memory=False)

# Display the original columns for reference
print("Original columns:")
print(df.columns.tolist())

# Drop the specified columns
columns_to_drop = [
        "UNIQUE_ID", "COLLISION_ID", "CRASH_DATE", "VEHICLE_ID",
        "TRAVEL_DIRECTION", "POINT_OF_IMPACT", "VEHICLE_DAMAGE",
        "VEHICLE_DAMAGE_1", "VEHICLE_DAMAGE_2", "VEHICLE_DAMAGE_3",
        "PUBLIC_PROPERTY_DAMAGE"
]
df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

# Drop other columns that are considered less relevant or overly granular.
# For example, if you decide to drop driver-specific details or vehicle model/year:
additional_columns_to_drop = [
        "VEHICLE_MODEL", "VEHICLE_YEAR", "STATE_REGISTRATION",
        "DRIVER_SEX", "DRIVER_LICENSE_STATUS", "DRIVER_LICENSE_JURISDICTION",
        "PRE_CRASH", "CONTRIBUTING_FACTOR_1", "CONTRIBUTING_FACTOR_2",
        "NUMBER OF PERSONS INJURED"  # Outcome-related and highly correlated with severity
]
df.drop(columns=[col for col in additional_columns_to_drop if col in df.columns], inplace=True)

# Create a binary variable 'Fatal_Crash': 1 if NUMBER OF PERSONS KILLED > 0, else 0.
if "NUMBER OF PERSONS KILLED" in df.columns:
        df["Fatal_Crash"] = (df["NUMBER OF PERSONS KILLED"] > 0).astype(int)
else:
        raise KeyError("Column 'NUMBER OF PERSONS KILLED' not found in the dataset.")

# Create a binary variable 'Ford_Involved' based on the vehicle make.
# Assuming the column is named 'VEHICLE_MAKE'
if "VEHICLE_MAKE" in df.columns:
        # Standardize text to lower-case and check if 'ford' is in the string
        df["Ford_Involved"] = df["VEHICLE_MAKE"].astype(str).str.lower().str.contains("ford").astype(int)
else:
        raise KeyError("Column 'VEHICLE_MAKE' not found in the dataset.")

# Process the 'CRASH_TIME' to create a categorical 'Time_of_Day' variable.
# We assume CRASH_TIME is in a format like 'HH:MM'
if "CRASH_TIME" in df.columns:
        try:
                # Convert to datetime and extract hour
                df["CRASH_TIME_parsed"] = pd.to_datetime(df["CRASH_TIME"], format="%H:%M", errors='coerce')
                # Create a categorical variable: Daytime (6:00-17:59) vs. Nighttime
                df["Time_of_Day"] = df["CRASH_TIME_parsed"].dt.hour.apply(
                        lambda x: "Daytime" if 6 <= x < 18 else "Nighttime")
        except Exception as e:
                print("Error parsing CRASH_TIME:", e)
                df["Time_of_Day"] = np.nan
else:
        print("CRASH_TIME column not found; 'Time_of_Day' will not be created.")

# Standardize the 'VEHICLE_TYPE' variable if it exists.
if "VEHICLE_TYPE" in df.columns:
        # Strip whitespace and convert to title-case. Further grouping can be applied as needed.
        df["Vehicle_Type"] = df["VEHICLE_TYPE"].astype(str).str.strip().str.title()
else:
        print("VEHICLE_TYPE column not found.")

# Optionally: Aggregate data at the crash level if the dataset includes multiple rows per crash.
# For example, if a crash appears multiple times (once per vehicle), you may want to:
# - Keep one record per crash.
# - Set 'Ford_Involved' = 1 if any vehicle in the crash is Ford.
# - Set 'Fatal_Crash' = 1 if any record in the crash has a fatality.
# This step assumes there is a common crash identifier (which might have been dropped).
# If needed, uncomment and adapt the following:

# if 'CRASH_ID' in df.columns:
#     df = df.groupby('CRASH_ID').agg({
#         'Fatal_Crash': 'max',
#         'Ford_Involved': 'max',
#         'Time_of_Day': 'first',  # or use mode if varying
#         'Vehicle_Type': 'first'  # or apply a grouping strategy
#     }).reset_index()

# Check for missing values in the key columns and report
print("\nMissing values in key columns:")
print(df[["Fatal_Crash", "Ford_Involved", "Time_of_Day", "Vehicle_Type"]].isnull().sum())

# Save the cleaned dataset to a new CSV file
cleaned_data_path = 'Motor_Vehicle_Collisions_Cleaned.csv'
df.to_csv(cleaned_data_path, index=False)
print(f"\nCleaned data saved to {cleaned_data_path}")
