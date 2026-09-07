# fraud-calibrated — D6

**Tier:** 2 · **Category:** D — Analytics & data science · **Wave:** 3

Root rules in `../GUIDELINES.md` apply. This file is project-specific only — keep it under 40 lines.

## What this is

Credit default risk done the way a risk team would actually do it: the decision rule is
chosen to minimise **expected money lost**, not to maximise F1. That forces three things
most portfolio projects skip — the scores have to be calibrated probabilities, the cost of
a mistake has to be per-customer rather than a flat ratio, and the threshold has to be
fitted on a fold the test set never sees.

## Stack

Python 3.12 · pandas · scikit-learn · LightGBM · SHAP · matplotlib · argparse CLI.
No services, no API keys, no env vars. Dataset (5.3 MB) is committed at
`data/raw/uci350.zip`, not downloaded on every clone/CI run.

## Acceptance criteria

- [x] Class imbalance handled and the handling *evaluated*, not assumed (3.5:1 here)
- [x] Cost-sensitive threshold, derived from a stated cost model with sensitivity analysis
- [x] Calibration measured (Brier, ECE, reliability curve) before and after
- [x] SHAP global + local explanations
- [x] Drift detection with measured, not only synthetic, distribution shift
- [ ] Ship gate passes (`/ship`) — deliberately not run this session (built-not-shipped, per user request)

## Project-specific notes

- **Dataset:** UCI id 350, "default of credit card clients" (Taiwan, 2005). CC BY 4.0.
  30,000 rows, 22.12% positive. Legacy `.xls`, needs `xlrd`; cached to gzipped CSV after
  the first parse because reading the Excel takes ~6 s.
- **Amounts are NT$.** Do not silently reprint them as USD anywhere.
- **Imbalance is moderate (3.5:1), not extreme.** Say so; do not borrow language from the
  0.2%-positive fraud literature.
- Cost-model constants (LGD 0.75, intervention NT$500, forgone margin 5%) are **assumptions,
  not measurements**. Every result that depends on them must say so.
- Local pytest needs `--basetemp=<scratchpad>/pt`; the sandbox blocks `%TEMP%`.
