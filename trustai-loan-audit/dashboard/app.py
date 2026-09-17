"""
TrustAI Audit Dashboard - visual front-end tying Modules 1, 2 and 5
together (Exp 2 + Exp 5 + Exp 7 in one interactive view).

Run with:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    false_positive_rate,
    selection_rate,
)
from sklearn.metrics import accuracy_score

sys.path.append(str(Path(__file__).parent.parent))
from data.load_data import load_processed

sys.path.append(str(Path(__file__).parent.parent / "src"))
import importlib

fair_mod = importlib.import_module("02_fairness_explainability")

import simulator

st.set_page_config(page_title="TrustAI Audit Dashboard", layout="wide")
st.title("TrustAI Audit Dashboard")
st.caption(
    "Fairness, explainability and drift audit of an AI loan-approval model "
    "- German Credit dataset"
)

df = load_processed()

tab1, tab2, tab3, tab4 = st.tabs(
    ["Bias Overview", "Fairness & Explainability", "Applicant Simulator", "About this audit"]
)

with tab1:
    st.subheader("Class distribution")
    col1, col2 = st.columns(2)
    with col1:
        counts = df["target"].value_counts().rename({1: "Good credit", 0: "Bad credit"})
        st.bar_chart(counts)
    with col2:
        approval_by_sex = df.groupby("sex")["target"].mean() * 100
        st.bar_chart(approval_by_sex.rename("Approval rate (%)"))

    st.subheader("Approval rate by age group")
    df["age_group"] = pd.cut(
        df["age"], bins=[0, 25, 40, 60, 100], labels=["<25", "25-40", "40-60", "60+"]
    )
    approval_by_age = df.groupby("age_group", observed=True)["target"].mean() * 100
    st.bar_chart(approval_by_age.rename("Approval rate (%)"))

with tab2:
    st.subheader("Model fairness metrics")
    with st.spinner("Training model and computing fairness metrics..."):
        pipe, X_train, X_test, y_test, y_pred, sens_test = simulator.get_full_audit(df)

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

    col1, col2, col3 = st.columns(3)
    col1.metric("Overall accuracy", f"{accuracy_score(y_test, y_pred):.1%}")
    col2.metric("Demographic Parity Diff.", f"{dpd:.3f}")
    col3.metric("Equalized Odds Diff.", f"{eod:.3f}")

    st.write("Group-wise breakdown (by sex):")
    st.dataframe(mf.by_group.round(3))

    st.caption(
        "Demographic Parity Difference = 0 means both groups are approved at "
        "the same rate. Equalized Odds Difference = 0 means both groups get "
        "equally accurate treatment. Values above ~0.1 are generally flagged "
        "as a fairness concern worth investigating."
    )

with tab3:
    st.subheader("Score a hypothetical applicant")
    st.caption(
        "Enter an applicant profile, run it through the real audited model, "
        "then test whether the decision would change for reasons that have "
        "nothing to do with creditworthiness."
    )

    pipe, X_train = simulator.get_production_pipeline(df)
    pipe_with_sex = simulator.get_sex_aware_pipeline(df)

    default = simulator.DEFAULT_APPLICANT
    applicant = dict(default)

    def opt_index(field):
        return simulator.CATEGORICAL_OPTIONS[field].index(default[field])

    with st.form("applicant_form"):
        st.info(
            "Pre-filled with a real, deliberately borderline applicant from the "
            "dataset - click **Run audit** as-is first, then try the counterfactual "
            "checks below before changing anything."
        )
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Financial history**")
            applicant["checking_status"] = st.selectbox(
                "Checking account status",
                simulator.CATEGORICAL_OPTIONS["checking_status"],
                index=opt_index("checking_status"),
            )
            applicant["savings_status"] = st.selectbox(
                "Savings account status",
                simulator.CATEGORICAL_OPTIONS["savings_status"],
                index=opt_index("savings_status"),
            )
            applicant["credit_history"] = st.selectbox(
                "Credit history",
                simulator.CATEGORICAL_OPTIONS["credit_history"],
                index=opt_index("credit_history"),
            )
            applicant["existing_credits"] = st.slider(
                "Existing credits at this bank", 1, 4, default["existing_credits"]
            )
            applicant["other_payment_plans"] = st.selectbox(
                "Other payment plans",
                simulator.CATEGORICAL_OPTIONS["other_payment_plans"],
                index=opt_index("other_payment_plans"),
            )

        with c2:
            st.markdown("**Loan details**")
            applicant["purpose"] = st.selectbox(
                "Purpose", simulator.CATEGORICAL_OPTIONS["purpose"], index=opt_index("purpose")
            )
            applicant["credit_amount"] = st.number_input(
                "Credit amount requested",
                min_value=250,
                max_value=20000,
                value=default["credit_amount"],
                step=100,
            )
            applicant["duration"] = st.slider("Duration (months)", 4, 72, default["duration"])
            applicant["installment_commitment"] = st.slider(
                "Installment rate (% of income)", 1, 4, default["installment_commitment"]
            )
            applicant["other_parties"] = st.selectbox(
                "Co-applicant / guarantor",
                simulator.CATEGORICAL_OPTIONS["other_parties"],
                index=opt_index("other_parties"),
            )

        with c3:
            st.markdown("**Personal profile**")
            applicant["age"] = st.slider("Age", 19, 75, default["age"])
            applicant["employment"] = st.selectbox(
                "Employment length",
                simulator.CATEGORICAL_OPTIONS["employment"],
                index=opt_index("employment"),
            )
            applicant["job"] = st.selectbox(
                "Job category", simulator.CATEGORICAL_OPTIONS["job"], index=opt_index("job")
            )
            applicant["housing"] = st.selectbox(
                "Housing", simulator.CATEGORICAL_OPTIONS["housing"], index=opt_index("housing")
            )
            applicant["property_magnitude"] = st.selectbox(
                "Property owned",
                simulator.CATEGORICAL_OPTIONS["property_magnitude"],
                index=opt_index("property_magnitude"),
            )
            applicant["foreign_worker"] = st.selectbox(
                "Foreign worker",
                simulator.CATEGORICAL_OPTIONS["foreign_worker"],
                index=opt_index("foreign_worker"),
            )
            applicant["residence_since"] = default["residence_since"]
            applicant["num_dependents"] = default["num_dependents"]
            applicant["own_telephone"] = default["own_telephone"]

        submitted = st.form_submit_button("Run audit on this applicant", type="primary")

    if submitted:
        pred, proba = simulator.predict(pipe, applicant, simulator.FEATURES)

        st.divider()
        st.subheader("Decision")
        col1, col2 = st.columns([1, 2])
        with col1:
            if pred == 1:
                st.success(f"**APPROVED**  (confidence: {proba:.1%})")
            else:
                st.error(f"**REJECTED**  (confidence: {1 - proba:.1%})")

        with col2:
            contrib = simulator.local_shap_contributions(pipe, applicant)
            fig, ax = plt.subplots(figsize=(6, 3))
            fig.patch.set_alpha(0)
            ax.patch.set_alpha(0)
            text_color = "#B0B3B8"
            colors = ["#55A868" if v > 0 else "#C44E52" for v in contrib["impact"][::-1]]
            ax.barh(contrib["feature"][::-1], contrib["impact"][::-1], color=colors)
            ax.set_xlabel("Impact on approval (SHAP value)", color=text_color)
            ax.set_title("Why the model made this decision", color=text_color)
            ax.tick_params(colors=text_color)
            for spine in ax.spines.values():
                spine.set_color(text_color)
            plt.tight_layout()
            st.pyplot(fig, transparent=True)
            plt.close(fig)

        st.divider()
        st.subheader("Counterfactual fairness checks")

        cf1, cf2 = st.columns(2)

        with cf1:
            st.markdown("**Age flip** *(age is a real model feature)*")
            age_result = simulator.age_counterfactual(pipe, applicant)
            st.write(
                f"Age {age_result['original_age']} -> "
                f"**{'Approved' if age_result['original_decision'] else 'Rejected'}** "
                f"({age_result['original_proba']:.1%} approval confidence)"
            )
            st.write(
                f"Age {age_result['flipped_age']} (everything else unchanged) -> "
                f"**{'Approved' if age_result['flipped_decision'] else 'Rejected'}** "
                f"({age_result['flipped_proba']:.1%} approval confidence)"
            )
            if age_result["decision_changed"]:
                st.error(
                    "Decision changed based on age alone. This is direct evidence "
                    "of age-based disparate treatment in the production model."
                )
            else:
                st.info("Decision did not change - age alone was not decisive here.")

        with cf2:
            st.markdown("**Sex flip** *(production model never sees sex)*")
            sex_result = simulator.sex_counterfactual(pipe_with_sex, applicant)
            st.write(
                f"As male -> **{'Approved' if sex_result['male_decision'] else 'Rejected'}** "
                f"({sex_result['male_proba']:.1%} approval confidence)"
            )
            st.write(
                f"As female (everything else unchanged) -> "
                f"**{'Approved' if sex_result['female_decision'] else 'Rejected'}** "
                f"({sex_result['female_proba']:.1%} approval confidence)"
            )
            if sex_result["decision_changed"]:
                st.error(
                    "A model given direct access to sex WOULD flip its decision here "
                    "- direct disparate treatment. The production model avoids this "
                    "specific failure mode by never seeing sex, but Module 2's "
                    "Equalized Odds Difference (0.112) shows it still discriminates "
                    "indirectly through correlated features - proxy discrimination "
                    "is harder to fix than simply hiding the sensitive attribute."
                )
            else:
                st.info(
                    "Even a sex-aware model doesn't flip on this particular profile - "
                    "try a different applicant profile, disparate treatment is not "
                    "guaranteed on every single case."
                )

        if pred == 0:
            st.divider()
            st.subheader("Follow-up: LLM resource advice (Exp 4 tie-in)")
            advice = simulator.llm_advice_preview(rejected=True)
            st.write(
                f"Since this applicant was rejected, the system would recommend: "
                f"**{advice['name']}** ({advice['cost']})"
            )
            st.caption(
                "Demo-mode mock LLM (see `src/06_llm_bias_probe.py`). Run the full "
                "probe with N=20 repeats per prompt type to see whether this "
                "recommendation is systematically biased toward expensive options."
            )

with tab4:
    st.markdown(
        """
        This dashboard is the visual layer of a larger audit pipeline:

        1. **Bias Autopsy** (tab 1) - dataset-level bias, maps to Exp 2
        2. **Fairness + Explainability** (tab 2) - model-level bias + SHAP, maps to Exp 5
        3. **Applicant Simulator** (tab 3) - live scoring + counterfactual fairness probes
        4. **Algorithmic Impact Assessment** - `docs/03_algorithmic_impact_assessment.md`, maps to Exp 6
        5. **Self-Audit** - `docs/04_self_audit_matrix.md`, maps to Exp 1
        6. **Drift & Autonomy Audit** - `src/05_drift_monitor.py`, maps to Exp 7
        7. **LLM Resource-Recommendation Bias** - `src/06_llm_bias_probe.py`, maps to Exp 4

        See the project `README.md` for the full mapping and how to reproduce
        every result in the final report.
        """
    )
