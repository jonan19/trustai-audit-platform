# Ethical Self-Audit — TrustAI Audit Platform

**Application audited:** the TrustAI dashboard and pipeline built in this
project (Modules 1, 2, 5, 6), rather than a third-party AI application.
**Maps to:** Experiment 1 — but applied recursively: we turn Exp 1's own
ethical evaluation matrix onto the tool this project produced.

---

## Ethical Evaluation Matrix

| Dimension | Assessment | Evidence / Notes |
|---|---|---|
| **Fairness** | Partially addressed | The system *measures* fairness (Demographic Parity, Equalized Odds, group FPR) but does not automatically *enforce* it — a fairness-constrained retrain (Module 3, R3 mitigation) is recommended, not yet applied by default |
| **Transparency** | Strong | SHAP global + local explanations are generated for every model; the dashboard exposes raw metrics rather than a single opaque "approved/rejected" verdict |
| **Accountability** | Moderate | Code and decisions are traceable (Git-friendly scripts, logged outputs in `outputs/`), but there is no formal sign-off/audit-trail process for who approves a model version for "deployment" |
| **Privacy** | Strong for this context | Uses a public, anonymised benchmark dataset with no real personal data; a production version would need a data-protection/consent review not required here |
| **Safety** | Moderate | Drift monitoring (Module 5) exists, but has no automated circuit-breaker — a human must act on the "full autonomy withdrawn" recommendation; it is advisory, not enforced |
| **Human Oversight** | Moderate | The autonomy policy in Module 5 explicitly recommends human-in-the-loop review under drift, but the current pipeline does not technically block automated decisions when that recommendation fires |

## Strengths

- Combines statistical bias detection, formal fairness metrics, and
  explainability in one auditable pipeline rather than treating them as
  separate, disconnected checks.
- Findings are quantitative and reproducible (fixed random seeds), not just
  qualitative impressions — makes the self-audit falsifiable rather than a
  checklist exercise.
- Drift monitoring extends the audit beyond a one-time snapshot, addressing
  a blind spot most classroom fairness exercises skip entirely.

## Risks / Gaps

- Fairness metrics are *reported* but not *enforced* — a deployment could
  ignore the dashboard's warnings with no technical barrier stopping it.
- No mechanism yet logs *who* viewed a fairness warning and *what* action
  was taken — accountability is procedural, not systemic.
- The LLM bias probe (Module 6) runs in demo/mock mode by default; its
  findings are illustrative of the *method*, not a certified measurement of
  any specific real LLM until run in LIVE mode with an API key.

## Recommendations for Responsible Use

1. Before treating this as production-ready, add an automated gate that
   blocks model deployment if Equalized Odds Difference exceeds an agreed
   threshold (e.g. 0.1).
2. Log every drift-alert event and the human decision taken in response,
   to close the accountability gap identified above.
3. Re-run Module 6 in LIVE mode against the actual LLM your organisation
   plans to use for applicant-facing advice before trusting its output.

## Conclusion

Applying Exp 1's own ethical checklist to the tool built in Modules 2–6
shows the project practices what it audits: it is transparent and
fairness-aware by design, but — like most fairness tooling — stops short of
being self-enforcing. That gap between *measuring* an ethical property and
*guaranteeing* it is itself one of the central findings of AI ethics as a
field, which is a fitting note to end the audit on.
