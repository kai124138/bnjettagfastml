# BNJetTag — Project Context (long-term memory)

Stable facts about the project. Imported by `CLAUDE.md`, so keep it concise.

## The thesis
A 1-bit **BitNet** binary `{−1,+1}`-weight transformer jet tagger for the **CMS Level-1
trigger**. Claim: binary weights map to FPGA **LUT/logic instead of DSPs**, so the binary
core uses **~0 DSP** while tagging efficiency stays close to full precision. Two axes are
measured: (a) tagging efficiency (AUC) vs FP32 / W8A8 baselines, and (b) how far
**activations** can be quantized (A8 → A6 → A4) before efficiency / resources / latency degrade.

## Infrastructure reality (important)
- **Training** runs on **NRP Nautilus** (Kubernetes GPU), submitted via job YAMLs in
  `qkeras-bitnet-run-2026-06-22/code/jobs/training/`.
- **HLS C-synthesis** runs on **`mulder`** (Vitis HLS 2023.2). DSP=0 confirmed 2026-06-24.
- **Do NOT run full training or Vitis synthesis locally / in this environment.** Local work
  = edit code, prep job YAMLs, parse logs, recompute ROC/AUC from `.npz`, plot, update docs.

## Key results — TWO different metrics, do not conflate
> All AUCs below are **era-1** (old private 2-class dataset, runs ≤ round-4). Round-5
> onward trains on the public HLS4ML LHC Jet 5-class data (macro-OvR AUC) and starts a
> fresh table — see `RESEARCH.md` §2/§5. Era-1 ↔ era-2 comparisons are forbidden.
- **Validation AUC** (during training; run `README.md` headline table):
  FP32 0.7703 · W8A8 0.7719 · **W1A8 BitNet 0.7530** · A8/A6/A4 = 0.7524 / 0.7507 / 0.7437 ·
  ternary 0.7685 · softmax-free 0.7562.
- **ROC-test AUC** (`roc-results/roc_auc.md`, n = 222,912):
  FP32 0.8207 · W8A8 0.8283 · A8 0.7986 · A6 0.7984 · A4 0.7886.
- Cost of going 1-bit ≈ **−1.7 AUC pts** vs FP32, for **0 DSP** on the binary core
  (structural, precision-independent).
- Folded device-fit (RF=256): binary FFN on a VU13P ≈ 25% LUT / 7% FF / 1.2% BRAM / 0 DSP,
  latency ≈ 520 cycles ≈ 1.3 µs @ 400 MHz.

## Where things live
- Code: `qkeras-bitnet-run-2026-06-22/code/` → `training/` (`qkerasModel.py`, `ROC.py`,
  `make_roc.py`, `util/`), `hls/` (csynth/probe scripts), `plots/`, `jobs/` (training + hls YAMLs).
- Job knobs: `BN_VARIANT` (bitnet/vanilla/w8a8), `BN_ACT_BITS` (8/6/4), `BN_TERNARY`, `BN_SOFTMAX_FREE`.
- Trained models: `models/` (large binaries, gitignored — see `models/MODEL.md`).
- ROC data: `qkeras-bitnet-run-2026-06-22/roc-results/*.npz`.
- HLS results: `qkeras-bitnet-run-2026-06-22/results/` (`hls_resource_table.md`, `csynth/`).
- Living docs (top level): `00-START-HERE.md` (the map) and `RESEARCH.md` (all verified
  results + status — the doc the PI reads). Frozen deliverables: `reports/` (June-22
  REPORT.md, presentation outline/slides/speaker notes, strategy review) — historical,
  corrected by dated notes only. Run-level: `results/REPORT.md`, `results/RESULTS.md`.
- Workflow playbooks: `.claude/skills/{nrp-training-run, verify-roc, hls-mulder}/SKILL.md`.
- Private learning notes (gitignored): `LEARN/` (Obsidian vault).
- `archive/` — session exports, superseded duplicates, June-22 stray logs. Not load-bearing.

## Ignore (noise — don't read for context)
`.venv-plots/` (vendored deps), `archive/`, `*.zip`, `__pycache__/`, `.DS_Store`.

## Secrets
`qkeras-bitnet-run-2026-06-22/wandb-api-key.txt` holds a real W&B key in plaintext
(rotated 2026-07-01, chmod 600; the NRP secret `kai-wandb` must be updated to match after
any rotation). Never print, commit, or paste it.
