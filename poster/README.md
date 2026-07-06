# poster/ — FastML26 poster deliverables (2026-07-04)

Produced by a parallel session under hard isolation: everything outside this
directory was treated as read-only; nothing in the results store was modified.

| file | what it is |
|---|---|
| `VERIFICATION.md` | The gate: every poster number re-derived from raw store data; §4 lists what failed and is banned from figures/draft |
| `methods_results_draft.md` | Methods + results narrative, anchored to the submitted abstract, large-model-provisional throughout |
| `GAPS.md` | Numbers the poster wants that the store cannot give, and how to close each |
| `figures/fig1_tradeoff_table.{pdf,svg}` | AUC + EBOPs across FP32/W8A8/W1A8/W1A6/W1A4 (two labeled EBOPs conventions, never mixed) |
| `figures/fig2_roc_per_class.{pdf,svg}` | Per-class ROC, HEP convention (TPR linear, mistag log), trained vs HGQ2 rebuild vs FP32/W8A8 |
| `figures/fig3_dsp_probes.{pdf,svg}` | Whole-probe DSP totals: norm-free binary probes = 0, SubLN-bearing probes carry every DSP |
| `figures/fig4_precision_sweep.{pdf,svg}` | AUC / EBOPs / FFN-block LUT vs activation bits |
| `scripts/verify_gate.py` | Recomputes all AUCs/Δ/corr from raw npz + EBOPs sums → `data/verification_results.json`, `data/roc_curves.npz` |
| `scripts/verify_ebops_analytic.py` | Re-derives the analytic EBOPs table from architecture dims → `data/ebops_analytic_check.json` |
| `scripts/verify_dsp_split.py` | Re-derives the per-function DSP splits from the raw csynth.xml module tables (added 2026-07-05) → `data/dsp_split_check.json` |
| `scripts/make_figures.py` | Renders all four figures from verified values only (raw source cited per constant) |

Regenerate everything:

```
python3 poster/scripts/verify_gate.py
python3 poster/scripts/verify_ebops_analytic.py
.venv-hgq2/bin/python poster/scripts/make_figures.py
```

Palette (validated with the dataviz six-checks validator, light surface, all
PASS): A8 `#2a78d6` · A6 `#199e70` · A4 `#e34948` · SubLN/norm `#4a3aa7` · FP32
ink · W8A8 muted gray; trained = solid, HGQ2 rebuild = dashed, identical
assignments in every figure.
