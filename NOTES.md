# Build Notes — fraud-calibrated

Working notes: what broke, what you tried, why you chose X over Y.
Not for recruiters — for you, six months from now, in an interview.

---

## Log

### 2026-09-07
- **Tried:** downloading the UCI archive directly and reading the sole zip member.
- **Broke:** nothing at the data layer, but worth flagging — the file inside the zip is
  a legacy BIFF `.xls` (not `.xlsx`), which needs `xlrd`, not `openpyxl`. Confirmed via
  magic bytes (`\xd0\xcf\x11\xe0...`, OLE2 compound file) before writing the parser
  rather than guessing from the extension.
- **Fixed by:** `xlrd>=2.0` added as a dependency; `pd.read_excel(..., header=1)` since
  row 0 is a merged "X1, X2, ..." banner row, not the real header.
- **Learned:** `xlrd` parsing this file takes ~6s. Everything downstream (splits, model
  fits, SHAP, drift, and every CLI subcommand) needs the cleaned frame, so it's cached
  to gzipped CSV after the first parse — the CLI runs are now sub-second on the data
  layer.

- **Tried:** round-tripping the cleaned frame through the CSV cache and asserting the
  reloaded frame equals the freshly-cleaned one, as a test.
- **Broke:** it didn't — a real bug. `clean()` casts the target column to `int8`
  (`df[TARGET] = df[TARGET].astype(np.int8)`), but `pd.read_csv` on the cached file
  infers `int64` by default. `pd.testing.assert_frame_equal` caught the dtype mismatch
  immediately; nothing downstream would have noticed silently (LightGBM/sklearn don't
  care about int8 vs int64), but it's exactly the kind of "worked in dev, subtly
  different after a cache hit" bug that's worth catching with a test rather than luck.
- **Fixed by:** re-casting the target column to int8 immediately after `pd.read_csv` in
  `load()`'s cache-hit branch.
- **Learned:** any "cache to disk, reload later" path needs a round-trip test that
  actually asserts equality, not just "the frame loads and has the right shape."

- **Tried:** SMOTE (imbalanced-learn) as the imbalance-handling approach, since it's the
  first thing most tutorials reach for.
- **Rejected:** `PAY_1`..`PAY_6` are ordinal repayment-status *codes* (-2 = no
  consumption, -1 = paid in full, 0 = revolving credit used, 1-8 = months overdue), not
  continuous measurements. SMOTE's k-NN interpolation would average two customers'
  repayment codes into a value like 1.4 -- a number that doesn't correspond to any real
  repayment status. `scale_pos_weight` reweights the loss instead of inventing rows,
  which sidesteps the whole question of what an interpolated categorical feature means.

- **Tried:** a single global FN:FP cost ratio (the conventional approach), then a
  per-row cost model scaled by each customer's actual exposure.
- **Learned:** the per-row rule is genuinely cheaper (NT$12.79M vs NT$13.44M best global,
  a 4.85% saving) *and* flags dramatically fewer customers (53% vs 99.9%). The reason
  the best global cutoff degenerates to "flag almost everyone": at the default cost
  assumptions (LGD 75%, margin 5%, intervention NT$500), the fp:fn cost ratio for a
  *typical*-exposure customer is well below the population base rate of 22%, so any
  single threshold low enough to make sense for high-exposure customers ends up flagging
  nearly all of them, low-exposure included. This is a real, somewhat uncomfortable
  finding about the cost assumptions, not a bug -- flagged explicitly in the README
  rather than tuned away by picking friendlier cost constants.

- **Tried:** including AGE in the drift PSI table (young vs. older is the split itself).
- **Broke:** PSI on AGE came back as 9.3 -- correctly detecting that the two slices have
  different age distributions, because age *is* the split. Meaningless as a "finding."
- **Fixed by:** excluding AGE from the reported PSI table in `cmd_drift` (it's still a
  valid split key, just not a valid drift signal about itself).

---

## Rejected approaches

| Approach | Why rejected |
|---|---|
| SMOTE / other oversampling for imbalance | Interpolates discrete ordinal repayment codes into values with no real meaning (see log above). |
| Flat FN:FP cost ratio | Ignores that exposure varies from NT$10,000 to NT$1,000,000 across customers; the per-row model is the whole point of this project. |
| Platt (sigmoid) calibration | LightGBM's miscalibration on this data isn't obviously sigmoid-shaped; isotonic's extra flexibility is affordable with ~4,800 validation positives. |
| KernelExplainer for SHAP | Model-agnostic but approximate and slow; TreeExplainer is exact and fast for LightGBM, so there's no tradeoff to make. |
| Synthetic/injected drift | Proves the PSI code runs, not that the model holds up under a shift that could actually happen. |

## Open questions

- [ ] Would a monotone-constrained LightGBM (e.g. non-decreasing risk in PAY_1..PAY_6)
      change the SHAP story, or just make the existing one more defensible to a credit
      committee? Not tried — flagged for a follow-up session, not this project's scope.
- [ ] The isotonic calibrator's out-of-bounds behaviour (`out_of_bounds="clip"`) means a
      test-time probability above every validation probability gets clipped to the
      validation set's max calibrated value. With a 6,000-row validation fold this
      hasn't mattered in practice, but it's the first thing to check if this model were
      ever retrained on a much smaller validation slice.
