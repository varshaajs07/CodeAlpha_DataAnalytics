"""
CodeAlpha Data Analytics Internship
TASK 3: Data Visualization

Dataset: Titanic passenger dataset (built into the Seaborn library)
Goal: Turn raw data into clear, insightful charts using
      Matplotlib and Seaborn.

How to run:
    python task3_visualization.py

Requirements:
    pip install pandas matplotlib seaborn
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")
os.makedirs("output_images", exist_ok=True)

# ---------------------------------------------------------
# Load the dataset
# ---------------------------------------------------------
df = sns.load_dataset("titanic")
print(f"Loaded dataset with shape: {df.shape}")

# ---------------------------------------------------------
# CHART 1: Bar chart - Survival count
# ---------------------------------------------------------
plt.figure(figsize=(6, 5))
sns.countplot(data=df, x="survived", hue="survived", palette="Set2", legend=False)
plt.title("Survival Count (0 = Did Not Survive, 1 = Survived)")
plt.xlabel("Survived")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig("output_images/chart1_survival_count.png")
plt.close()
print("Saved chart1_survival_count.png")

# ---------------------------------------------------------
# CHART 2: Bar chart - Survival rate by passenger class
# ---------------------------------------------------------
plt.figure(figsize=(6, 5))
sns.barplot(data=df, x="pclass", y="survived", hue="pclass", palette="Set1", errorbar=None, legend=False)
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig("output_images/chart2_survival_by_class.png")
plt.close()
print("Saved chart2_survival_by_class.png")

# ---------------------------------------------------------
# CHART 3: Bar chart - Survival rate by gender
# ---------------------------------------------------------
plt.figure(figsize=(6, 5))
sns.barplot(data=df, x="sex", y="survived", hue="sex", palette="Set3", errorbar=None, legend=False)
plt.title("Survival Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig("output_images/chart3_survival_by_gender.png")
plt.close()
print("Saved chart3_survival_by_gender.png")

# ---------------------------------------------------------
# CHART 4: Histogram - Age distribution
# ---------------------------------------------------------
plt.figure(figsize=(7, 5))
sns.histplot(df["age"].dropna(), bins=30, kde=True, color="skyblue")
plt.title("Age Distribution of Passengers")
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("output_images/chart4_age_distribution.png")
plt.close()
print("Saved chart4_age_distribution.png")

# ---------------------------------------------------------
# CHART 5: Box plot - Fare by passenger class
# ---------------------------------------------------------
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="pclass", y="fare", hue="pclass", palette="pastel", legend=False)
plt.title("Fare Distribution by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Fare")
plt.tight_layout()
plt.savefig("output_images/chart5_fare_by_class.png")
plt.close()
print("Saved chart5_fare_by_class.png")

# ---------------------------------------------------------
# CHART 6: Pie chart - Passenger distribution by embarkation town
# ---------------------------------------------------------
plt.figure(figsize=(6, 6))
embark_counts = df["embark_town"].value_counts()
plt.pie(
    embark_counts,
    labels=embark_counts.index,
    autopct="%1.1f%%",
    colors=sns.color_palette("Set2"),
    startangle=90,
)
plt.title("Passenger Distribution by Embarkation Town")
plt.tight_layout()
plt.savefig("output_images/chart6_embark_town_pie.png")
plt.close()
print("Saved chart6_embark_town_pie.png")

# ---------------------------------------------------------
# CHART 7: Combined dashboard-style figure (2x2 grid)
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

sns.countplot(data=df, x="survived", hue="survived", ax=axes[0, 0], palette="Set2", legend=False)
axes[0, 0].set_title("Survival Count")

sns.barplot(data=df, x="pclass", y="survived", hue="pclass", ax=axes[0, 1], palette="Set1", errorbar=None, legend=False)
axes[0, 1].set_title("Survival Rate by Class")

sns.histplot(df["age"].dropna(), bins=25, kde=True, ax=axes[1, 0], color="coral")
axes[1, 0].set_title("Age Distribution")

sns.boxplot(data=df, x="pclass", y="fare", hue="pclass", ax=axes[1, 1], palette="pastel", legend=False)
axes[1, 1].set_title("Fare by Class")

plt.suptitle("Titanic Dataset - Mini Dashboard", fontsize=16)
plt.tight_layout()
plt.savefig("output_images/chart7_mini_dashboard.png")
plt.close()
print("Saved chart7_mini_dashboard.png")

print("\nALL CHARTS SAVED in the 'output_images' folder.")
