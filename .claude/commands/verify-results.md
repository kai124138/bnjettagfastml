---
description: Recompute AUCs from the .npz files and check them against the report claims.
argument-hint: "[optional: which model, e.g. A4]"
---
Use the **results-analyst** subagent to verify our numbers $ARGUMENTS.

Load the relevant `qkeras-bitnet-run-2026-06-22/roc-results/*.npz`, recompute AUC with
scikit-learn, and compare against **both** tables: the validation-AUC headline (run `README.md`)
and the ROC-test table (`roc-results/roc_auc.md`). Report each as claimed vs recomputed with a
✓ / ✗, flag any discrepancy, and append the outcome to `.claude/memory/experiment-log.md`.
