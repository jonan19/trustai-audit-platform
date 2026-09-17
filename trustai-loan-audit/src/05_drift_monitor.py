"""
Module 5 - Drift & Autonomy Audit  (maps to Exp 7)

Simulates the Module 2 model running in "production" over three
synthetic time windows with an injected population shift (an ageing,
lower-income applicant pool - a realistic scenario as a bank's customer
base changes). Uses the Kolmogorov-Smirnov test to detect data drift on
each numeric feature, then reports whether the model's decisions would
still need human review under a simple human-in/on-the-loop policy.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

sys.path.append(str(Path(__file__).parent.parent))
from data.load_data import load_processed

OUT = Path(__file__).parent.parent / "outputs"
OUT.mkdir(exist_ok=True)

NUMERIC_FEATURES = ["duration", "credit_amount", "age", "installment_commitment"]
DRIFT_ALPHA = 0.05  # KS test significance threshold


def make_window(df: pd.DataFrame, rng: np.random.Generator, shift: float) -> pd.DataFrame:
    """Simulates a later deployment window by ageing the population and
    inflating credit amounts (proportional shift, not a random resample)."""
    w = df.copy()
    w["age"] = w["age"] + shift * 8
    w["credit_amount"] = w["credit_amount"] * (1 + shift * 0.15)
    noise = rng.normal(0, shift * 2, size=len(w))
    w["duration"] = (w["duration"] + noise).clip(lower=1)
    return w


def run_ks_tests(baseline: pd.DataFrame, window: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feat in NUMERIC_FEATURES:
        stat, p_value = ks_2samp(baseline[feat], window[feat])
        rows.append(
            {
                "feature": feat,
                "ks_statistic": round(stat, 4),
                "p_value": round(p_value, 4),
                "drift_detected": p_value < DRIFT_ALPHA,
            }
        )
    return pd.DataFrame(rows)


def autonomy_recommendation(pct_drifted_features: float) -> str:
    if pct_drifted_features == 0:
        return "Human-on-the-loop is sufficient (model may run autonomously; periodic spot-checks)."
    elif pct_drifted_features < 0.5:
        return "Escalate to human-in-the-loop for borderline decisions; retrain scheduled soon."
    else:
        return "Full autonomy withdrawn: require human-in-the-loop review for ALL decisions until retrained."


def main():
    df = load_processed()
    rng = np.random.default_rng(42)

    windows = {"Quarter+1": 0.0, "Quarter+2": 0.6, "Quarter+3": 2.0}
    all_results = {}

    print("=" * 60)
    print("MODULE 5 - DRIFT & AUTONOMY AUDIT")
    print("=" * 60)

    for label, shift in windows.items():
        window_df = make_window(df, rng, shift)
        results = run_ks_tests(df, window_df)
        all_results[label] = results

        pct_drifted = results["drift_detected"].mean()
        print(f"\n--- {label} (simulated shift={shift}) ---")
        print(results.to_string(index=False))
        print(f"Features drifted: {pct_drifted:.0%}")
        print(f"Policy recommendation: {autonomy_recommendation(pct_drifted)}")

    # Plot KS statistic trend across windows per feature
    fig, ax = plt.subplots(figsize=(8, 5))
    for feat in NUMERIC_FEATURES:
        trend = [all_results[w].set_index("feature").loc[feat, "ks_statistic"] for w in windows]
        ax.plot(list(windows.keys()), trend, marker="o", label=feat)
    ax.axhline(0.1, color="red", linestyle="--", linewidth=1, label="Rough drift concern level")
    ax.set_title("Data Drift (KS statistic) Over Simulated Deployment Windows")
    ax.set_ylabel("KS statistic")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "06_drift_trend.png", dpi=150)
    plt.close()

    print(f"\nDrift trend chart saved to: {OUT / '06_drift_trend.png'}")


if __name__ == "__main__":
    main()
