---
name: verify-roc
description: Canonical procedure for verifying any BNJetTag accuracy number — recompute AUC from the .npz ROC arrays and check it against what a report claims. Use before quoting, writing, or updating ANY AUC anywhere (RESEARCH.md, reports/, README tables, slides), when asked to "verify results" or "check the numbers", or after a ROC job returns new .npz files.
---

# Verify ROC/AUC — the numbers gate

Rule zero: **recompute, don't trust.** A number may be written into a report only if it was
recomputed from the underlying `.npz` in the current session, or is quoted with its source
file. Usually run by the `results-analyst` agent.

## Environment

Throwaway env only: `pip install --break-system-packages numpy scikit-learn` (add
matplotlib for plots). **Never** activate or import from `.venv-plots/` (vendored, huge).

## Procedure

1. **Locate the arrays:** `qkeras-bitnet-run-2026-06-22/roc-results/*.npz` (era-1 canon:
   `FP32-vanilla`, `W8A8-baseline`, `A8/A6/A4-binary-softmax`; round-5 files arrive via
   `kai-roc-r5.yaml`).
2. **Inspect before computing:** `d = np.load(path); print(d.files, {k: d[k].shape for k in d.files})`.
   Confirm what you actually have (labels vs scores vs precomputed fpr/tpr) and the sample
   count — era-1 ROC-test is n = 222,912; a different n means a different split, say so.
3. **Compute the right AUC for the right era:**
   - **Era 1 (pre-2026-07-01, private dataset):** binary —
     `sklearn.metrics.roc_auc_score(y_true, y_score)` (or `np.trapz` on stored fpr/tpr —
     state which).
   - **Era 2 (round-5+, HLS4ML LHC Jet):** 5-class **macro one-vs-rest** —
     `roc_auc_score(y_onehot, scores, multi_class="ovr", average="macro")`, plus the 5
     per-class AUCs (g/q/W/Z/t). Headline = macro-OvR.
4. **Compare against every place the number is claimed:** `roc-results/roc_auc.md`, the run
   `README.md` headline table, `RESEARCH.md` §5, anything in `reports/` being touched.
   Val AUC and ROC-test AUC are different measurements — check each against its own table,
   never against each other.
5. **Report verbatim:** "claimed 0.7986, recomputed 0.7986 ✓" or "✗ claimed X, got Y,
   Δ=…". Never round a discrepancy away; a mismatch is a finding, not an embarrassment.
6. **Log the outcome** in `.claude/memory/experiment-log.md` (date, files checked, ✓/✗).
   Only after ✓ may `RESEARCH.md` or a report be updated.

## Labeling rules (non-negotiable)

- Every number carries three labels: **metric** (val AUC vs ROC-test AUC), **era**
  (era 1 private 2-class vs era 2 public 5-class), **status** (single-run vs seed-averaged).
- Era-1 ↔ era-2 comparisons are forbidden in any report.
- Round-5 headline rule: seed-averaged AND ROC-tested, baselines = max(original, lr-tuned).

## Plots

Regenerate with `qkeras-bitnet-run-2026-06-22/code/plots/make_results_plots.py`. HEP
convention for ROC axes: x = tagging efficiency (TPR), y = mistag rate (log scale).
