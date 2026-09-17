"""
Module 2 - Fairness + Explainability Dashboard  (maps to Exp 5)

Trains a baseline loan-approval classifier, then audits it with:
  - fairlearn: Demographic Parity Difference, Equalized Odds Difference,
    group-wise selection rate / accuracy / false-positive rate
  - shap: global feature importance + one local explanation for a
    rejected applicant

Saves all charts to outputs/ and prints the metric table used in the
AIA (docs/03_algorithmic_impact_assessment.md).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    false_positive_rate,
    selection_rate,
)
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

sys.path.append(str(Path(__file__).parent.parent))
from data.load_data import load_processed

OUT = Path(__file__).parent.parent / "outputs"
OUT.mkdir(exist_ok=True)

FEATURES = [
    "checking_status", "duration", "credit_history", "purpose", "credit_amount",
    "savings_status", "employment", "installment_commitment", "other_parties",
    "residence_since", "property_magnitude", "age", "other_payment_plans",
    "housing", "existing_credits", "job", "num_dependents", "own_telephone",
    "foreign_worker",
]


def build_pipeline(df: pd.DataFrame) -> Pipeline:
    numeric = [c for c in FEATURES if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in FEATURES if c not in numeric]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric),
        ]
    )
    clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    return Pipeline([("pre", pre), ("clf", clf)])


def train_test(df: pd.DataFrame):
    X = df[FEATURES]
    y = df["target"]
    sensitive = df["sex"]

    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X, y, sensitive, test_size=0.3, random_state=42, stratify=y
    )

    pipe = build_pipeline(df)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    return pipe, X_train, X_test, y_test, y_pred, sens_test


def fairness_audit(y_test, y_pred, sens_test):
    dpd = demographic_parity_difference(y_test, y_pred, sensitive_features=sens_test)
    eod = equalized_odds_difference(y_test, y_pred, sensitive_features=sens_test)

    mf = MetricFrame(
        metrics={
            "accuracy": accuracy_score,
            "selection_rate": selection_rate,
            "false_positive_rate": false_positive_rate,
        },
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=sens_test,
    )

    print("=" * 60)
    print("MODULE 2 - FAIRNESS AUDIT")
    print("=" * 60)
    print(f"\nOverall accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"Demographic Parity Difference: {dpd:.3f}")
    print(f"Equalized Odds Difference:     {eod:.3f}")
    print(f"\nGroup-wise breakdown:\n{mf.by_group.round(3).to_string()}")

    mf.by_group.plot(kind="bar", figsize=(8, 4), title="Fairness Metrics by Sex")
    plt.tight_layout()
    plt.savefig(OUT / "04_fairness_by_group.png", dpi=150)
    plt.close()

    return dpd, eod, mf


def explainability_audit(pipe: Pipeline, X_train, X_test):
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]

    X_train_t = pre.transform(X_train)
    X_test_t = pre.transform(X_test)
    feature_names = pre.get_feature_names_out()

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test_t)

    # shap_values shape depends on sklearn/shap version; normalise to 2D array
    # for the positive class ("good credit").
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values[..., 1]

    plt.figure()
    shap.summary_plot(
        sv, X_test_t, feature_names=feature_names, show=False, max_display=12
    )
    plt.tight_layout()
    plt.savefig(OUT / "05_shap_global_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\nGlobal SHAP importance chart saved: 05_shap_global_importance.png")
    print("(Use this to say WHICH features drive approval overall.)")


def main():
    df = load_processed()
    pipe, X_train, X_test, y_test, y_pred, sens_test = train_test(df)
    fairness_audit(y_test, y_pred, sens_test)
    explainability_audit(pipe, X_train, X_test)
    print(f"\nAll charts saved to: {OUT.resolve()}")


if __name__ == "__main__":
    main()
