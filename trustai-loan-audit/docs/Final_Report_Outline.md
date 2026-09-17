# Final Report Outline — TrustAI Audit Platform

Use this as the skeleton for the document you actually submit. Pull the
numbers/charts from `outputs/` and the module scripts as you go — don't
re-derive them by hand.

1. **Title Page** — "TrustAI Audit Platform: A Fairness, Explainability and
   Governance Audit of an AI Loan-Approval System"
2. **Abstract** (150–200 words) — one paragraph summarising the pipeline and
   headline findings (7.5pp approval gap by sex; Equalized Odds Diff. 0.112;
   drift detected from Quarter+2; cost-anchoring bias trend 0.45→1.35)
3. **Introduction** — why loan-approval AI is a good high-stakes case study;
   state which 6 of the 7 lab experiments this project integrates, and why
   Experiment 3 (Deepfake) was excluded (different data modality — media
   forensics vs. tabular decisioning)
4. **Related Frameworks** — UNESCO Recommendation on Ethics of AI, IEEE
   Ethically Aligned Design (brief, cite Exp 1/6 theory sections)
5. **Methodology**
   - 5.1 Dataset (German Credit / OpenML `credit-g`)
   - 5.2 Module 1 — Bias Autopsy (Exp 2)
   - 5.3 Module 2 — Model, Fairness Metrics, SHAP (Exp 5)
   - 5.4 Module 3 — Algorithmic Impact Assessment (Exp 6)
   - 5.5 Module 4 — Self-Audit (Exp 1)
   - 5.6 Module 5 — Drift & Autonomy Audit (Exp 7)
   - 5.7 Module 6 — LLM Cost-as-Proxy Bias Probe (Exp 4)
6. **Results** — insert the charts from `outputs/01`–`06`, the table from
   `outputs/07_llm_bias_probe_results.csv`, and a screenshot of the
   Applicant Simulator's counterfactual result (age 22 → Rejected, age 60 →
   Approved, everything else unchanged) — this single screenshot is the
   most persuasive evidence in the whole report, lead with it if presenting
   live
7. **Discussion** — tie R1–R8 from the AIA (`docs/03_...md`) back to UNESCO/
   IEEE principles; discuss the self-audit findings from `docs/04_...md`
8. **Limitations** — demo-mode LLM probe (no live API key used), synthetic
   drift injection rather than real longitudinal data, single dataset
9. **Conclusion**
10. **References** — reuse the reference lists already present in your
    EXP1–EXP7 `.docx` files (Boddington, Crawford, Kearns & Roth, Russell,
    Loukides/Mason/Patil, UNESCO Recommendation)
11. **Appendix** — full code listing or link to the repo
