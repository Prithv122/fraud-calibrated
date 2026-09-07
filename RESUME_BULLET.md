# Resume Bullets — fraud-calibrated

Form: **action → technical specifics → measured outcome.** Numbers or it doesn't go on the resume.

---

## Bullets

- Built a cost-sensitive credit-default risk model (LightGBM, 30K accounts, 3.5:1 class
  imbalance) with isotonic-calibrated probabilities and a per-customer expected-cost
  decision rule, cutting total misclassification cost 4.85% versus the best single global
  threshold while flagging under half as many customers (53% vs. 99.9%).
- Reduced Expected Calibration Error 85% (0.160 → 0.024) by fitting isotonic regression on
  a held-out validation fold, after identifying that raw gradient-boosted probabilities
  were systematically overconfident enough to distort a cost-based decision threshold.
- Measured model degradation under real (not synthetic) covariate shift across a
  subpopulation split, finding a moderate 0.19 PSI in credit-limit distribution but only a
  0.28-point AUC drop — a finding, not an assumption.
- Explained individual credit decisions with SHAP TreeExplainer, surfacing the most recent
  repayment status as the dominant driver at 2x the influence of any other feature.

## Which roles this supports

- [x] Data Scientist / ML
- [x] Data Analyst / Python Developer
- [ ] AI Engineer (LLM/NLP/CV)
- [ ] Data Engineer

## Keywords this project earns

LightGBM, class imbalance, `scale_pos_weight`, probability calibration (isotonic
regression, Brier score, Expected Calibration Error), cost-sensitive thresholding,
SHAP / model explainability, population stability index (PSI), covariate shift, credit
risk modeling, scikit-learn, pandas.

---

### Bad vs good

❌ "Built a machine learning model to predict customer churn using Python."
✅ "Built a churn classifier on 240k accounts (LightGBM, 1:40 class imbalance) with isotonic calibration and cost-sensitive thresholding, lifting precision@10% from 0.31 to 0.58 over the business's existing rules baseline."

The second one is answerable in an interview. The first invites the question you can't answer.
