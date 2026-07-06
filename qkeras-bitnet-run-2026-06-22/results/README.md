# results/ — which results are current (decoder, 2026-07-05)

| Path | What it is | Status |
| --- | --- | --- |
| **`hgq2/`** | **The current results store** (era 2, HGQ2 path). `runs/<config-hash>/` = one dir per pipeline config (extract/calibrate/build/verify/ebops JSONs + synthesis probe dirs with `csynth_report.json`, raw `csynth.xml`, per-module `csynth_modules.json`). Top level: `tradeoff_table.md` (THE table), `constraints_map.md` (what HGQ2+hls4ml can/can't do), `roc_hgq2_overlay.png`, `dashboard.html`, `manifest.json`. Config hashes decode via `../code/hgq2/configs/*.json`. | **CURRENT** |
| `RESULTS.md`, `REPORT.md`, `SESSION_REPORT.md` | The June-22 **era-1** run's data export, per-run report, and session narrative. | frozen |
| `hls_resource_table.md`, `csynth/` | Era-1 QKeras-path synthesis table + raw reports (the original DSP-0 confirmation; era-1 shapes). | frozen |
| `variant_sweep.md`, `ebops.md` | Round-4/5-era analyses (lr tuning; EBOPs conventions). Superseded where the store overlaps. | frozen reference |
| `plots/` | Era-1 figures. | frozen |

Sibling folder `../roc-results/`: the `.npz` ROC arrays every AUC is recomputed from —
root files = era-1, `r5/` = era-2 round 5. **Era-1 and era-2 numbers are never compared.**
