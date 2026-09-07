# Interview Prep — fraud-calibrated

**Five questions, five answers.** An unanswered question means this project is not shipped.

---

### Q1. Walk me through the architecture in 90 seconds.

_A:_ It's a credit-default risk pipeline over the UCI "default of credit card clients"
dataset, 30,000 Taiwanese credit accounts, 22% default rate. The Excel source gets cleaned
(fixing an off-by-one column name, folding undocumented category codes) and cached to
CSV. I do a stratified 60/20/20 split — train, valid, test — because I need *two* held-out
folds: one to calibrate the model's probabilities, and a separate one to actually report
results on. I train LightGBM with `scale_pos_weight` for the 3.5:1 imbalance, then fit
isotonic regression on the validation fold to correct the raw scores, which are
overconfident straight out of the tree model. Then — the part that makes this different
from a typical classifier project — I don't pick one threshold. I compute a per-customer
cost of a false negative (letting a defaulter through, scaled by their actual unpaid
exposure) and a false positive (flagging a good customer, scaled by forgone margin plus a
fixed intervention cost), and the flag decision is wherever those two costs cross for
*that specific customer*. Then SHAP for explanations and a real-subpopulation drift check
on top.

### Q2. Why did you choose isotonic calibration over Platt scaling, and why does calibration matter here at all?

_A:_ Calibration matters because my decision rule literally divides a fixed cost by the
predicted probability (`threshold = fp_cost / fn_cost`, and you flag when
`prob >= threshold`). If the probability is wrong — not mis-ranked, wrong in an absolute
sense — the threshold comparison is wrong even if the ranking (AUC) looks great. LightGBM's
raw output is expected to be a probability, but tree ensembles are known to push scores
toward the extremes more than the true rate warrants. I measured it: before calibration,
Brier score 0.168 and Expected Calibration Error 0.160 on the test fold; after isotonic
calibration fit on the *validation* fold (not test — that would leak), Brier drops to
0.134 and ECE to 0.024, an 85% reduction. I chose isotonic over Platt/sigmoid scaling
because Platt assumes the miscalibration is sigmoid-shaped, and I didn't have a reason to
believe that here; isotonic is non-parametric and just needs enough data not to overfit —
with ~4,800 validation-fold positives, that's a reasonable bet, and the calibration-curve
plot confirms the shape isn't a simple sigmoid.

### Q3. What's the weakest part of this, and what would break first under load?

_A:_ Two things. First, honestly: the imbalance here (3.5:1, 22% positive) is moderate,
not extreme — I say that explicitly in the README rather than borrow "fraud detection"
framing from a domain with 100-1000x worse imbalance. Second, the cost-model constants
(75% loss-given-default, NT$500 intervention cost, 5% forgone margin) are assumptions I
made up, grounded in what's directionally reasonable for consumer credit, not measured
from this bank's actual recovery data — because I don't have that data. I say so in the
README and ship a sensitivity table (`fraud-calibrated sensitivity`) so the reader can see
how much the total-cost number moves as those assumptions vary, rather than presenting one
number as if it were audited. Under load/scale, the isotonic calibrator is the first thing
to break — it's a step function with a knot at every distinct validation score, which
doesn't scale gracefully past a few hundred thousand rows; I'd switch to a monotone spline
with a fixed number of knots (see README §7).

### Q4. How do you know it works? What did you measure, and against what baseline?

_A:_ Against a logistic-regression baseline (class-balanced, one-hot encoded): LightGBM
gets AUC 0.775 vs. 0.726, PR-AUC 0.545 vs. 0.507 — a real, non-trivial lift, checked before
I trusted anything downstream. For the cost rule specifically, the baseline that matters
isn't "no model," it's "the best possible *single* global threshold" — because that's what
a team without per-row costing would actually ship. I scanned 200 global cutoffs from
0.001 to 0.99 and took the cheapest one: NT$13.44M total cost on the test fold, flagging
99.9% of customers (at these cost assumptions, the best single cutoff degenerates to
"flag almost everyone," which is itself a finding I kept rather than hid). My per-row rule
costs NT$12.79M — 4.85% cheaper — while flagging only 53% of customers. Both numbers come
from the same test fold, same model, same calibration; only the threshold rule differs.

### Q5. Your drift check splits by age. Isn't that a slightly artificial "drift" — you're not showing me production data changing over time.

_A:_ Correct, and I'd rather own that limitation than dress it up. I don't have a second
time period for this dataset (it's a single 2005 snapshot), so I can't show a true
before/after-shift-in-time comparison. What I did instead was pick a real subpopulation
split already present in the data — customers aged 30 or under versus older — rather than
injecting synthetic noise or an artificial shift, which is what most "drift" sections in
portfolio projects actually do (and which just proves the PSI code runs, not that the
model holds up under a shift that could plausibly happen). The result is honest and a bit
undramatic: LIMIT_BAL shows a moderate PSI of 0.188 between the two age groups (younger
customers get materially smaller credit limits, which makes sense), every other feature is
under 0.06, and model AUC moves by only 0.0028 (0.7776 to 0.7748) across the split. I'd
frame this to a hiring manager as "here's the right *method* for measuring a real
distribution difference," not as "here's proof the model degrades under drift" — because
on this data, it mostly doesn't.

---

## 30-second pitch

Credit-default risk on 30,000 real accounts, 3.5:1 imbalance handled without synthetic
oversampling. The distinguishing piece: instead of one classification threshold, every
customer gets their own break-even point, derived from their actual unpaid exposure and a
stated cost model — and that per-customer rule measurably beats the best possible single
global cutoff (4.85% cheaper, flags under half as many people). Isotonic calibration fixes
a real overconfidence problem in the raw scores (ECE down 85%), SHAP explains individual
decisions, and drift is checked against a real subpopulation, not injected noise.
