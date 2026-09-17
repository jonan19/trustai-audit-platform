"""
Helper functions for the Applicant Simulator tab (dashboard/app.py).

Adds two interactive capabilities on top of the existing audit pipeline:
  1. Live scoring of a hand-entered applicant, with a SHAP local
     explanation for that one decision.
  2. Counterfactual fairness probes:
       - age flip: change ONLY age, holding everything else fixed, on the
         real production model (age is an actual model feature)
       - sex flip: trains a second, sex-aware model purely for this probe
         (the production model in Module 2 never sees sex) to test
         whether direct disparate treatment would occur if sex WAS fed
         to the model - contrasted against the proxy discrimination the
         production model already exhibits without seeing sex at all.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

sys.path.append(str(Path(__file__).parent.parent / "src"))
import importlib

fair_mod = importlib.import_module("02_fairness_explainability")
llm_mod = importlib.import_module("06_llm_bias_probe")

FEATURES = fair_mod.FEATURES

CATEGORICAL_OPTIONS = {
    "checking_status": ["<0", "0<=X<200", ">=200", "no checking"],
    "credit_history": [
        "no credits/all paid", "all paid", "existing paid",
        "delayed previously", "critical/other existing credit",
    ],
    "purpose": [
        "new car", "used car", "furniture/equipment", "radio/tv",
        "domestic appliance", "repairs", "education", "retraining",
        "business", "other",
    ],
    "savings_status": ["<100", "100<=X<500", "500<=X<1000", ">=1000", "no known savings"],
    "employment": ["unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"],
    "other_parties": ["none", "co applicant", "guarantor"],
    "property_magnitude": ["real estate", "life insurance", "car", "no known property"],
    "other_payment_plans": ["none", "bank", "stores"],
    "housing": ["own", "rent", "for free"],
    "job": [
        "unskilled resident", "unemp/unskilled non res", "skilled",
        "high qualif/self emp/mgmt",
    ],
    "own_telephone": ["yes", "none"],
    "foreign_worker": ["yes", "no"],
}

# Deliberately a real borderline case (row 1 of the dataset): this exact
# profile sits close enough to the model's decision boundary that flipping
# ONLY age (22 -> 60, nothing else) flips Rejected -> Approved. Picked this
# way on purpose so the counterfactual demo shows something interesting the
# first time you press the button, instead of a comfortably-approved
# profile that never budges.
DEFAULT_APPLICANT = {
    "checking_status": "0<=X<200",
    "duration": 48,
    "credit_history": "existing paid",
    "purpose": "radio/tv",
    "credit_amount": 5951,
    "savings_status": "<100",
    "employment": "1<=X<4",
    "installment_commitment": 2,
    "other_parties": "none",
    "residence_since": 2,
    "property_magnitude": "real estate",
    "age": 22,
    "other_payment_plans": "none",
    "housing": "own",
    "existing_credits": 1,
    "job": "skilled",
    "num_dependents": 1,
    "own_telephone": "none",
    "foreign_worker": "yes",
}


@st.cache_resource
def get_full_audit(_df: pd.DataFrame):
    """The exact model + train/test split audited in Module 2 - never sees `sex`.
    Cached so switching tabs doesn't retrain the model each time."""
    return fair_mod.train_test(_df)


def get_production_pipeline(_df: pd.DataFrame):
    pipe, X_train, *_ = get_full_audit(_df)
    return pipe, X_train


def _build_sex_aware_pipeline(df: pd.DataFrame) -> Pipeline:
    features_plus_sex = FEATURES + ["sex"]
    categorical = [c for c in features_plus_sex if not pd.api.types.is_numeric_dtype(df[c])]
    numeric = [c for c in features_plus_sex if c not in categorical]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric),
        ]
    )
    clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(df[features_plus_sex], df["target"])
    return pipe


@st.cache_resource
def get_sex_aware_pipeline(_df: pd.DataFrame):
    """
    A second model, trained WITH sex as an explicit feature, used only to
    probe direct disparate treatment. This model is NOT the one audited
    elsewhere in this project and is not recommended for deployment - it
    exists purely so the simulator can show what would happen if sex were
    fed to the model directly, as a contrast to the proxy discrimination
    the sex-blind production model already exhibits.
    """
    return _build_sex_aware_pipeline(_df)


def applicant_to_frame(applicant: dict, columns: list) -> pd.DataFrame:
    row = {c: applicant[c] for c in columns}
    return pd.DataFrame([row])


def predict(pipe: Pipeline, applicant: dict, columns: list):
    X = applicant_to_frame(applicant, columns)
    pred = pipe.predict(X)[0]
    proba = pipe.predict_proba(X)[0][1]
    return int(pred), float(proba)


def local_shap_contributions(pipe: Pipeline, applicant: dict, top_n: int = 8) -> pd.DataFrame:
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]

    X = applicant_to_frame(applicant, FEATURES)
    X_t = pre.transform(X)
    feature_names = pre.get_feature_names_out()

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_t)
    sv = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0, :, 1]

    contrib = pd.DataFrame({"feature": feature_names, "impact": sv})
    contrib["abs_impact"] = contrib["impact"].abs()
    return contrib.sort_values("abs_impact", ascending=False).head(top_n)


def age_counterfactual(pipe: Pipeline, applicant: dict):
    flipped = dict(applicant)
    flipped["age"] = 22 if applicant["age"] >= 40 else 60

    orig_pred, orig_proba = predict(pipe, applicant, FEATURES)
    flip_pred, flip_proba = predict(pipe, flipped, FEATURES)

    return {
        "original_age": applicant["age"],
        "flipped_age": flipped["age"],
        "original_decision": orig_pred,
        "flipped_decision": flip_pred,
        "original_proba": orig_proba,
        "flipped_proba": flip_proba,
        "decision_changed": orig_pred != flip_pred,
    }


def sex_counterfactual(pipe_with_sex: Pipeline, applicant: dict):
    male_row = dict(applicant, sex="male")
    female_row = dict(applicant, sex="female")

    male_pred, male_proba = predict(pipe_with_sex, male_row, FEATURES + ["sex"])
    female_pred, female_proba = predict(pipe_with_sex, female_row, FEATURES + ["sex"])

    return {
        "male_decision": male_pred,
        "female_decision": female_pred,
        "male_proba": male_proba,
        "female_proba": female_proba,
        "decision_changed": male_pred != female_pred,
    }


def llm_advice_preview(rejected: bool):
    """Reuses the Module 6 demo-mode mock LLM for a one-shot advice preview."""
    if not rejected:
        return None
    import random

    rng = random.Random()
    rec = llm_mod.call_llm_demo("general", rng)
    return rec
