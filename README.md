# BitNet Jet Tagger — run 2026-06-22 · directory & table of contents

A **binary `{−1,+1}` 1-bit BitNet transformer** jet tagger for the CMS Level-1 trigger, trained on NRP
Nautilus and pushed through hls4ml. The thesis: binary weights map to FPGA **LUT/logic instead of DSPs**,
so we measure (a) tagging efficiency vs full-precision baselines and (b) how far activations can be
quantized before efficiency/resources/latency degrade.

This README is the map. **All paths below are relative to this folder.**

---

## Headline results

| model | what it is | val AUC | knob | job file |
| --- | --- | --- | --- | --- |
| **1-bit BitNet** | the model — binary `{−1,+1}` weights, 8-bit activations (W1A8) | **0.7530** | `BN_VARIANT=bitnet` | `code/jobs/training/kai-bn-train-paper-binary-lr15.yaml` |
| vanilla FP32 | full-precision baseline | 0.7703 | `BN_VARIANT=vanilla` | `code/jobs/training/kai-bn-train-vanilla-fp32.yaml` |
| W8A8 | conventional 8-bit baseline | 0.7719 | `BN_VARIANT=w8a8` | `code/jobs/training/kai-bn-train-w8a8.yaml` |
| sweep A8 / A6 / A4 | quantization-aggressiveness axis (canonical softmax, headline-consistent) | 0.7524 / 0.7507 / 0.7437 | `BN_ACT_BITS=8/6/4` | `code/jobs/training/kai-bn-train-paper-binary-sm-a{8,6,4}.yaml` |
| ternary *(appendix)* | `{−1,0,+1}` weights | 0.7685 | `BN_TERNARY=1` | `code/jobs/training/kai-bn-train-paper-ternary.yaml` |
| softmax-free *(appendix)* | `ReLU(QKᵀ)/N` attention | 0.7562 | `BN_SOFTMAX_FREE=1` | `code/jobs/training/kai-bn-train-paper-binary-sffree.yaml` |

**Cost of going 1-bit:** −1.7 AUC points vs FP32 (0.7530 vs 0.7703) for, in firmware, **0 DSP** on the binary
core — exact, precision-independent, and now **confirmed by real Vitis C-synthesis** (see hls table).

**hls4ml firmware → SYNTHESIZED (Vitis HLS 2023.2, run 2026-06-24 on `mulder`):** binary weights type as
`ap_uint<1>` → **DSP = 0 at A8/A6/A4, confirmed in synthesis** (the central claim, in silicon estimates). At the
folded device-fit operating point (RF=256) the binary FFN fits a VU13P at **~25% LUT / ~7% FF / ~1.2% BRAM /
0 DSP**, latency **520 cycles ≈ 1.3 µs @ 400 MHz** (Vitis Fmax ≈ 568 MHz); g++ emulation bit-accurate
(corr = 1.000). Full measured table + raw reports in `results/hls_resource_table.md` (§B) and `results/csynth/`.
*(Honest nuance: BRAM is 0 only for the fully-unrolled RF=1 design; a folded, deployable design parks the binary
weights in ~1.2% BRAM. DSP=0 is the fold-independent structural win.)*

---

## Where each of my four asks is answered

1. **"I want the exact synthesized LUT/FF/DSP/BRAM + latency table."**
   → **`results/hls_resource_table.md`** — **§B now MEASURED** by real Vitis HLS 2023.2 C-synthesis on `mulder`
   (2026-06-24): DSP=0 confirmed, LUT/FF/BRAM/latency filled at A8/A6/A4 (raw JSON in `results/csynth/`). The
   script that produced it is **`code/hls/run_csynth.py`** (hls4ml 1.x); verified runbook in
   **`code/hls/RUN_CSYNTH_ON_VITIS.md`** *(done)*; NRP Job `code/jobs/hls/kai-hls-csynth.yaml` kept for any
   Vitis-equipped cluster.
2. **"Give me the training code — qkeras, what we edited, baseline + vanilla."**
   → code in **`code/training/qkerasModel.py`** (one file; baseline/vanilla/bitnet are the `BN_VARIANT`
   knob, not separate files); the edits are catalogued in **`methods/code_changes.md`**; the Phase-0
   upstream diff is **`code/training/qkerasModel.patch`**.
3. **"Explain why we never retrained the BitNet after the new abstract."**
   → **`methods/abstract_and_training.md`** (the new abstract changed the *comparison set*, not the model;
   timeline shows the binary model predates it and was unchanged).
4. **"Reorganize the folder + a table of contents; archive the outdated, methods → methods, code/results/PNGs → their folders."**
   → this README + the layout below; PNGs in **`results/plots/`**; method docs in **`methods/`**;
   superseded jobs/code in **`archive/`**.

---

## Directory map

```
.
├── README.md                     ← you are here (table of contents)
├── results/                      ← the holy-grail outputs
│   ├── RESULTS.md                  data export: every AUC, the firmware table, derived resource counts
│   ├── REPORT.md                   full per-run detail + reproduce block
│   ├── SESSION_REPORT.md           narrative of how the run went (4 acts + errors/fixes)
│   ├── hls_resource_table.md       ★ ask #1 — the LUT/FF/DSP/BRAM + latency table (§B MEASURED via Vitis csynth)
│   ├── csynth/                      raw Vitis HLS reports: csynth_report_a{8,6,4}_rf256.json (back §B)
│   └── plots/                      4 data/arch diagnostics + 3 results figures (plots/README.md explains each)
│       ├── model_architecture.png
│       ├── data_jet_kinematics.png
│       ├── data_particle_kinematics.png
│       ├── sample_pt_reweighting.png   (sample weights, NOT network weights — see plots/README.md)
│       ├── results_auc_by_variant.png      ★ AUC across precisions (the −1.7 pt cost of going 1-bit)
│       ├── results_quant_axis.png          ★ AUC + FPGA resources vs activation bits A8/A6/A4
│       └── results_csynth_resources.png    ★ measured csynth %VU13P — DSP=0 headline
├── methods/                      ← method-oriented write-ups
│   ├── code_changes.md             ★ ask #2 — exactly what we edited in qkerasModel.py (4 groups)
│   ├── abstract_and_training.md    ★ ask #3 — the abstract / "did we retrain?" explanation
│   ├── hls4ml_findings.md          hls4ml convertibility + firmware findings
│   └── message-to-russell.md       the gradient-flow diagnosis for the upstream author
├── code/                         ← everything runnable
│   ├── CHANGELOG.md               ← code-change ledger: every file we wrote/edited + the hls4ml bug & fix log
│   ├── training/
│   │   ├── qkerasModel.py          ★ ask #2 — THE trainer (all 3 precisions via BN_VARIANT)
│   │   ├── qkerasModel.patch        the Phase-0 upstream data/env/W&B diff
│   │   ├── ROC.py, HLS_qk_Roc_Tracing.py, environment.yml, dataForgeScripts/, util/, README_upstream.md
│   ├── hls/
│   │   ├── run_csynth.py            ★ ask #1 — the script that RAN the Vitis csynth → §B measured
│   │   ├── RUN_CSYNTH_ON_VITIS.md   ★ ask #1 — turnkey runbook for the group's Vitis 2023.2 box
│   │   ├── sweep_precision.py, resource_model.py, convert_probe.py, full_transformer_probe.py, stage_a_fix.py
│   ├── plots/
│   │   └── make_results_plots.py    generates the 3 results_*.png publication figures (matplotlib)
│   └── jobs/
│       ├── training/               the 7 job YAMLs that produced the results table above
│       ├── hls/                    kai-hls-{csynth,sweep,inspect,full}.yaml
│       └── watch_*.sh              log-watchers
└── archive/                      ← superseded / off-thesis (kept for history)
    ├── jobs/                       16 earlier job YAMLs (pre-paper-recipe, smoke tests, dev pods)
    ├── qkerasModel_ste.py          pre-STE intermediate trainer
    └── upstream_samples/           original upstream .root samples
```

---

## Artifacts on the PVC (not in this folder — they're large)

NRP Nautilus, PVC `kai-data`, namespace `cms-ml` (`kubectl --context nautilus -n cms-ml`):

| artifact | path on PVC |
| --- | --- |
| trained models (per run) | `/data/outputs/qk-*/bitnet/noNorm_train_bitnetJetTagModel.h5` |
| the 1-bit headline model | `/data/outputs/qk-paper-binary-lr15/bitnet/…` |
| training logs | `/data/outputs/qk-*/train.log` |
| emitted hls4ml projects | `/data/outputs/hls/binary_ffn_a{8,6,4}_prj` (bit-accurate, NRP) |
| **synthesized csynth reports** | on `mulder` `~/csynth/out/csynth_report_a{8,6,4}_rf256.json`; copied into the repo at `results/csynth/` |
| metrics / learning curves / ROC | **Weights & Biases** — entity `kayamaguchi-uc-san-diego`, project `bnjettag-bitnet` |

---

## Reproduce the headline 1-bit model

```bash
kubectl --context nautilus -n cms-ml apply -f code/jobs/training/kai-bn-train-paper-binary-lr15.yaml
# BN_VARIANT defaults to "bitnet"; expect ~0.7530 (modulo seed noise).
# Swap the job file (table above) for vanilla / W8A8 / sweep / appendix variants.
```

## Conventions & notes
- **Paths** in every doc are relative to this folder.
- **One trainer, one knob:** `qkerasModel.py` produces all precisions via `BN_VARIANT ∈ {bitnet, vanilla, w8a8}`;
  there is no separate baseline/vanilla script.
- **Plots caveat:** `results/plots/sample_pt_reweighting.png` is the per-event training **sample weight**
  (flat-pT reweighting), not the network's learned weights; there are no ROC PNGs (ROC/AUC live in W&B).
- **Credential:** `wandb-api-key.txt` (chmod 600) holds the user's own W&B key for local runs — do not commit
  it anywhere public.
