import pandas as pd

# Load the dataset
data = pd.read_csv("water_potability.csv")

# Calculate the number of positives, negatives, and missing values
positives = data[data["Potability"] == 1].shape[0]
negatives = data[data["Potability"] == 0].shape[0]
missing_values = data.isna().sum()

# Get the statistical summary for all columns
summary = data.describe(include='all')

# Add custom statistics to the summary
summary.loc['positives'] = [positives if col == "Potability" else None for col in summary.columns]
summary.loc['negatives'] = [negatives if col == "Potability" else None for col in summary.columns]
summary.loc['missing_values'] = missing_values

# Export the summary to an Excel File
summary.to_excel("summary.xlsx")
