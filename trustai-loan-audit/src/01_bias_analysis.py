"""
Module 1 - Bias Autopsy  (maps to Exp 2: Detecting Dataset Bias)

Inspects the loan dataset for the bias types Exp 2 covers:
  - class imbalance (good vs bad credit)
  - sampling bias (sex, foreign-worker representation)
  - historical bias (approval rate split by sex / age group)

Saves each chart to outputs/ and prints a short findings summary that
feeds straight into the AIA (docs/03_algorithmic_impact_assessment.md).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.append(str(Path(__file__).parent.parent))
from data.load_data import load_processed

OUT = Path(__file__).parent.parent / "outputs"
OUT.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")


def class_imbalance(df: pd.DataFrame):
    counts = df["target"].value_counts().rename({1: "Good credit", 0: "Bad credit"})
    ax = counts.plot(kind="bar", color=["#4C72B0", "#DD8452"])
    ax.set_title("Class Distribution: Good vs Bad Credit")
    ax.set_ylabel("Number of applicants")
    plt.tight_layout()
    plt.savefig(OUT / "01_class_imbalance.png", dpi=150)
    plt.close()
    return counts


def sampling_bias(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    df["sex"].value_counts().plot(kind="pie", autopct="%1.0f%%", ax=axes[0])
    axes[0].set_ylabel("")
    axes[0].set_title("Applicant Sex Distribution")

    df["foreign_worker"].value_counts().plot(kind="pie", autopct="%1.0f%%", ax=axes[1])
    axes[1].set_ylabel("")
    axes[1].set_title("Foreign Worker Distribution")

    plt.tight_layout()
    plt.savefig(OUT / "02_sampling_bias.png", dpi=150)
    plt.close()


def historical_bias(df: pd.DataFrame):
    approval_by_sex = df.groupby("sex")["target"].mean() * 100

    df["age_group"] = pd.cut(
        df["age"], bins=[0, 25, 40, 60, 100], labels=["<25", "25-40", "40-60", "60+"]
    )
    approval_by_age = df.groupby("age_group", observed=True)["target"].mean() * 100

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    approval_by_sex.plot(kind="bar", ax=axes[0], color="#55A868")
    axes[0].set_title("Approval Rate (%) by Sex")
    axes[0].set_ylabel("% approved (good credit)")

    approval_by_age.plot(kind="bar", ax=axes[1], color="#C44E52")
    axes[1].set_title("Approval Rate (%) by Age Group")
    axes[1].set_ylabel("% approved (good credit)")

    plt.tight_layout()
    plt.savefig(OUT / "03_historical_bias.png", dpi=150)
    plt.close()
    return approval_by_sex, approval_by_age


def missing_values(df: pd.DataFrame):
    return df.isnull().sum().sum()


def main():
    df = load_processed()

    counts = class_imbalance(df)
    sampling_bias(df)
    approval_by_sex, approval_by_age = historical_bias(df)
    n_missing = missing_values(df)

    print("=" * 60)
    print("MODULE 1 - BIAS AUTOPSY FINDINGS")
    print("=" * 60)
    print(f"\nClass distribution:\n{counts.to_string()}")
    print(f"\nMissing values in dataset: {n_missing}")
    print(f"\nApproval rate by sex (%):\n{approval_by_sex.round(1).to_string()}")
    print(f"\nApproval rate by age group (%):\n{approval_by_age.round(1).to_string()}")
    print(f"\nCharts saved to: {OUT.resolve()}")

    gap = approval_by_sex.max() - approval_by_sex.min()
    print(f"\n>> Historical bias gap (sex): {gap:.1f} percentage points")
    print("   This is the number to quote in the AIA risk table.")


if __name__ == "__main__":
    main()
