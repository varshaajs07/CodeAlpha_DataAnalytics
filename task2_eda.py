import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# STEP 1: Load the dataset
# ---------------------------------------------------------
print("=" * 60)
print("STEP 1: Loading the Titanic dataset")
print("=" * 60)

df = sns.load_dataset("titanic")
print(f"Dataset shape (rows, columns): {df.shape}\n")
print("First 5 rows:")
print(df.head(), "\n")

# ---------------------------------------------------------
# STEP 2: Understand the structure (columns & data types)
# ---------------------------------------------------------
print("=" * 60)
print("STEP 2: Data structure - column names & data types")
print("=" * 60)
print(df.dtypes, "\n")
print("Basic info:")
df.info()
print()

# ---------------------------------------------------------
# STEP 3: Summary statistics
# ---------------------------------------------------------
print("=" * 60)
print("STEP 3: Summary statistics (numeric columns)")
print("=" * 60)
print(df.describe(), "\n")

print("Summary statistics (categorical columns)")
print(df.describe(include=["object", "str", "category", "bool"]), "\n")

# ---------------------------------------------------------
# STEP 4: Check for missing / problematic data
# ---------------------------------------------------------
print("=" * 60)
print("STEP 4: Missing values in each column")
print("=" * 60)
missing = df.isnull().sum()
missing_percent = (missing / len(df) * 100).round(2)
missing_table = pd.DataFrame({"missing_count": missing, "missing_percent": missing_percent})
print(missing_table[missing_table["missing_count"] > 0], "\n")

# ---------------------------------------------------------
# STEP 5: Ask meaningful questions & explore patterns
# ---------------------------------------------------------
print("=" * 60)
print("STEP 5: Answering meaningful questions about the data")
print("=" * 60)

# Q1: What was the overall survival rate?
overall_survival_rate = df["survived"].mean() * 100
print(f"Q1: Overall survival rate: {overall_survival_rate:.2f}%")

# Q2: Did passenger class affect survival?
survival_by_class = df.groupby("pclass")["survived"].mean() * 100
print("\nQ2: Survival rate (%) by passenger class:")
print(survival_by_class)

# Q3: Did gender affect survival?
survival_by_sex = df.groupby("sex")["survived"].mean() * 100
print("\nQ3: Survival rate (%) by gender:")
print(survival_by_sex)

# Q4: How does age relate to survival?
avg_age_by_survival = df.groupby("survived")["age"].mean()
print("\nQ4: Average age by survival outcome:")
print(avg_age_by_survival)

# Q5: Correlation between numeric variables
print("\nQ5: Correlation matrix (numeric columns):")
numeric_df = df.select_dtypes(include=["number"])
correlation = numeric_df.corr()
print(correlation)

# ---------------------------------------------------------
# STEP 6: Detect outliers / anomalies (example: fare)
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 6: Detecting outliers in 'fare' using the IQR method")
print("=" * 60)
Q1 = df["fare"].quantile(0.25)
Q3 = df["fare"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df["fare"] < lower_bound) | (df["fare"] > upper_bound)]
print(f"Number of fare outliers detected: {len(outliers)} out of {len(df)} rows")

# ---------------------------------------------------------
# STEP 7: Save a couple of quick EDA charts
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 7: Saving EDA charts to 'output_images' folder")
print("=" * 60)

import os
os.makedirs("output_images", exist_ok=True)

# Missing data heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Data Heatmap")
plt.tight_layout()
plt.savefig("output_images/eda_missing_data_heatmap.png")
plt.close()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Numeric Features")
plt.tight_layout()
plt.savefig("output_images/eda_correlation_heatmap.png")
plt.close()

print("Saved: output_images/eda_missing_data_heatmap.png")
print("Saved: output_images/eda_correlation_heatmap.png")

print("\nEDA COMPLETE. Review the printed output above and the two charts saved.")
