# Algorithmic Impact Assessment (AIA)

**System audited:** TrustAI loan-approval classifier (Module 2, `src/02_fairness_explainability.py`)
**Framework basis:** UNESCO Recommendation on the Ethics of Artificial Intelligence + IEEE Ethically Aligned Design
**Maps to:** Experiment 6

---

## 1. System Description

An AI-based credit-scoring system that predicts whether a loan applicant has
"good" or "bad" credit risk, using the German Credit dataset (1,000
applicants, 20 attributes including financial history, employment, housing,
and demographic fields). The system would be used by a bank to automate or
assist loan-approval decisions.

## 2. Stakeholders

| Stakeholder | Interest / Exposure |
|---|---|
| Loan applicants | Directly affected by approval/rejection; financial and reputational consequences |
| Bank / lender | Business risk (default rate), regulatory/legal exposure, reputational risk |
| Regulators | Compliance with anti-discrimination and consumer-protection law |
| Model developers | Accountable for design choices, feature selection, monitoring |
| Society | Precedent-setting effect on automated credit access and financial inclusion |

## 3. Data Used

- Source: UCI/OpenML "German Credit" (`credit-g`) dataset.
- Sensitive/protected attributes present: **sex** (derived from `personal_status`), **age**, **foreign_worker** status.
- Data used to train and test a RandomForest classifier (70/30 split, stratified).

## 4. Identified Risks (from Modules 1 & 2 results)

| # | Risk | Evidence | Likelihood | Impact | Overall |
|---|---|---|---|---|---|
| R1 | Historical bias in training data by sex | Module 1: approval rate 72.3% (male) vs 64.8% (female) — 7.5pp gap | High | Medium | **High** |
| R2 | Historical bias by age | Module 1: approval rate ranges from 57.9% (<25) to 77.8% (60+) | High | Medium | **High** |
| R3 | Disparate model treatment across sex | Module 2: Demographic Parity Difference = 0.040; Equalized Odds Difference = 0.112 | Medium | High | **High** |
| R4 | Unequal false-positive rate across sex | Module 2: FPR = 0.848 (female) vs 0.737 (male) — the model is more likely to wrongly flag female applicants as high-risk | Medium | High | **High** |
| R5 | Lack of explainability for individual decisions | Black-box RandomForest; mitigated by SHAP in Module 2, but not enforced by default | Medium | Medium | Moderate |
| R6 | Model degradation as applicant population shifts | Module 5: KS-test drift detected on `age`, `duration`, `credit_amount` from Quarter+2 onward | Medium | High | **High** |
| R7 | Downstream LLM advice to rejected applicants may itself be biased (Cost-as-a-Proxy) | Module 6: average recommended-resource price rank rises from 0.45 (budget-constrained) to 1.35 (unconstrained) | Medium | Low–Medium | Moderate |
| R8 | Lack of meaningful human oversight if deployed at full autonomy | Theoretical — no human-in-the-loop step currently enforced | Medium | High | **High** |

## 5. Mapping to UNESCO Principles

| Risk | UNESCO Principle(s) invoked |
|---|---|
| R1, R2, R3, R4 | *Fairness and Non-Discrimination* — AI systems should not perpetuate bias against groups |
| R5 | *Transparency and Explainability* — decisions affecting individuals should be explainable |
| R6, R8 | *Human Oversight and Determination* — humans must retain meaningful control, especially as system behaviour changes over time |
| R7 | *Responsibility and Accountability* — those deploying AI-driven advice remain accountable for its downstream effects |
| All | *Proportionality* — the level of oversight should match the stakes of the decision (credit access is high-stakes) |

## 6. Mapping to IEEE Ethically Aligned Design Principles

| Risk | IEEE Principle(s) invoked |
|---|---|
| R1, R2, R3, R4 | *Algorithmic bias / A_IS Effectiveness* — systems must be tested and validated to avoid unfair bias |
| R5 | *Transparency* — the basis of a decision should be discoverable |
| R6, R8 | *Human Rights & Well-being* — sustained human control as autonomous systems operate over time |
| R7 | *Accountability* — traceability of who is responsible for AI-driven recommendations |

## 7. Mitigation Measures

| Risk | Proposed mitigation |
|---|---|
| R1, R2 | Rebalance training data (oversampling under-25 and female applicants), or apply reweighting during training |
| R3, R4 | Apply a fairness-constrained learning algorithm (`fairlearn`'s `ExponentiatedGradient` with demographic parity or equalized-odds constraint); set an explicit fairness threshold as a deployment gate |
| R5 | Require a SHAP-based local explanation to be generated and logged for every rejection, made available to the applicant on request |
| R6 | Schedule periodic KS-test drift monitoring (as in Module 5) with automatic alerting; trigger mandatory retraining when >50% of monitored features drift |
| R7 | Add a disclaimer to LLM-generated financial advice; periodically audit LLM outputs for cost-anchoring bias using the Module 6 methodology |
| R8 | Enforce human-in-the-loop review for all rejections and for any decision made while drift alerts are active (see Module 5 policy engine) |

## 8. Overall Impact Classification

**Overall system risk: Moderate–High.**
The dataset carries measurable historical bias, the trained model reproduces
part of that bias (Equalized Odds Difference of 0.112 is above the ~0.1
concern threshold), and the model's reliability is not guaranteed to persist
under population drift. None of these risks are disqualifying, but none
should be deployed without the mitigations above and a standing human
oversight requirement (see Module 4 self-audit and Module 5 drift policy).

## 9. Conclusion

This AIA demonstrates that a technically accurate model (74.3% accuracy) is
not automatically an ethically deployable one. Fairness, explainability, and
long-term monitoring are separate, necessary conditions that must be
evaluated independently of predictive accuracy before an AI system is
allowed to make or influence high-stakes decisions about people's access to
credit.
