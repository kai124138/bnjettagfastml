# HGQ2 rebuild — change ledger

Running log of every consequential change in this effort. Dated, newest on top.

## 2026-07-04 — SubLN custom-layer extension DONE (keras-v3 → Vitis, C-sim gate passed)
- New `bnhgq2/subln.py`: `PSubLN` keras layer (parameter-free per-token LayerNorm,
  biased var, eps=1e-6, optional `flatten_axes=2` for last-two-axes norm), keras-v3
  handler, `SubLN` hls4ml IR class (new class — built-in `LayerNormalization`/QKeras
  path untouched, asserted in the test), `_produce_kif`/`_request_kif` BitExact
  registrations (output pinned fixed<18, 1+ceil(log2 √(dim−1))> per dim; input request
  fixed<31,15>-equiv), Vivado+Vitis config/function templates deriving internal C++
  types per layer from the final input precision, idempotent `register_subln()`.
- New `hls_templates/nnet_subln.h`: RANGE-REDUCED inverse sqrt — var (+eps) shifted by
  an even power of two onto [1/4,1) via MSB scan, 4096-entry 1/√ table over [1/4,1)
  only, exponent re-applied as an exact half-shift on the product. Replaces the shipped
  `nnet_layernorm.h` scheme whose table only covers var ∈ (eps,1] (raw jet features
  reach var ~1e6). Wide pinned internals (var ≈ ap_ufixed<47..67,·>, prod f=24).
- New `bnhgq2/compat.py`: `apply_hls4ml_compat()` (keras-3.15 `EinsumDense.full_output_shape`
  property restored for hgq2 0.1.9; keras-v3 registry alias
  `hgq.layers.attn.mha.QMultiHeadAttention` → stale registered key, same for Linformer)
  and `patch_project_for_macos()` (std::complex forward-decl → `#include <complex>` in
  the two ap_*_special.h of a WRITTEN project; call between `write()` and `_compile()`).
- New `test_subln.py` — the C-sim acceptance gate. ALL PASS locally (Vitis backend,
  io_parallel, 2048 samples/case, corr target ≥0.9999):
  raw (10,16) var 1e4–1e6 corr 0.999999997 max-err 6.8e-4 · (10,256) 0.999999996/1.0e-3 ·
  (10,1024) 0.999999994/1.1e-3 · (256,) 0.999999996/1.0e-3 · (10,4,64) fa=2
  0.999999996/9.8e-4 · +QDense(4, binary frozen ±1) 0.999999972/2.1e-2 ·
  +QEinsumDense abc,cd->abd (binary frozen) 0.999999975/5.9e-3. No NaN/inf anywhere.
- Gotcha: `convert_from_keras_model(..., bit_exact=...)` kwarg silently OVERWRITES
  `hls_config['Model']['BitExact']` — always pass the kwarg (cost one debug round:
  without it the raw-feature input stayed fixed<18,8> and wrapped).
- csynth (mulder) risks, unvalidated locally: `static constexpr double epsilon` in the
  generated config (C++14 odr corner; swap to a mant/shift pair if the Vitis frontend
  balks), full unroll + complete partition at dim=1024 (long csynth, big priority
  encoder over ~67-bit var word), 42×42-bit squares → deep DSP cascades (SubLN is the
  model's known DSP consumer, but budget unmeasured until csynth).

## 2026-07-04 — gold-model experiments fix the rebuild design (all on r5 val subset n=20k vs verified npz)
- **E1 (architecture proof)**: numpy gold model, *dynamic* act quant (exact QKeras
  semantics): corr **0.999948** vs stored A8 scores, macro-AUC 0.852450 vs 0.852420.
  Reimplementation correct; labels row-aligned exactly.
- **E2 (static substitution)**: naive max-calibrated per-tensor static fixed-point:
  A8 corr 0.969 / ΔAUC −0.005 · A6 0.871/−0.027 · A4 0.493/−0.078. MSE-optimal
  **per-channel** calibration recovers: A8 0.980/−0.0015 · A6 0.895/−0.017 ·
  A4 0.687/−0.017. → calibration policy = mse_per_channel, 8192-jet calib set.
- **E3 (β snap)**: naive pow2 β on all 51 layers is DESTRUCTIVE (A8 corr 0.72,
  −6.3 AUC pts). Mantissa sweep: k=2 → −0.015, k=3 → −0.005, k=4 → −0.003 AUC.
- **E4 (fold-aware β)**: exploiting LN scale-invariance: Wq/Wk β exact via softmax
  input_scaler (exp-LUT fold, free), Wv β dropped (LN-killed), fc1/head_fc1 exact
  via bias→b/β; only 18 residual contributors carry CSD-2 β̃ (≤4.5% err, 2-signed-
  digit → DSP-free). Result: A8 static corr 0.959/ΔAUC −0.0049 · A6 0.897/−0.0168 ·
  A4 0.687/−0.0173. **Design locked**: static per-channel MSE calib + fold-aware
  CSD-2. (Upgrade path if mulder shows affines stay DSP-free at higher precision:
  exact-β QBatchNormalization variant recovers ~+0.003 at A8.)
- A4 note: its loss is the static-quant substitution itself (dyn+fold corr 0.987) —
  a trained-static-quant (HGQ2 QAT) run would be the proper fix; logged as future work.
- hgq2/hls4ml source deep-dive (agent, all claims EXECUTED): binary pin =
  KBI(k0=1,b0=1,i0=1,SAT_SYM,frozen) passes ±1 bit-identically, reports 1 bit to
  EBOPs; QSoftmax folds input_scaler into exp LUT; PE+bias fold into QEinsumDense
  bias table (bias_axes='td'); keras LayerNormalization UNCONVERTIBLE 3 ways →
  custom PSubLN extension (in progress); QMultiHeadAttention has 2 hls4ml-1.3.0
  compat bugs (registry key module path + keras-3.15 full_output_shape removal) —
  patched in bnhgq2/compat.py; full MHA+FFN block verified BIT-EXACT (max diff 0.0)
  through Vitis-backend C-sim at RF 1 and 2.

## 2026-07-04 — session start: scaffold + infrastructure
- Scouted repo + HGQ2/hls4ml web state (7-agent sweep; findings in
  `.claude/memory/research-log.md` 2026-07-04 entry).
- Local env built: `.venv-hgq2` (py3.12, hgq2 0.1.9, hls4ml 1.3.0, keras 3.15, tf 2.21).
- All 8 round-5 checkpoints fetched from W&B → `models/r5/<key>/bitnet/*.h5` (76.9 MB each).
- Era-2 val split (Zenodo 3602260, 1.14 GB) downloaded → `data/` (project root).
- Confirmed by h5py inspection: r5 `.h5` holds all latent FP32 kernels/biases AND the folded
  positional-encoding constant (serialized in `model_config` → TFOpLambda add `y` kwarg) —
  the port needs no TF-2.11 environment.
- Round-6-small training YAMLs (D32/H4/L2/FFN64, era-2, PVC-free) generated + server-dry-run
  validated: `code/jobs/training/variants/kai-bn6s-*.yaml`. **NOT launched** — permission
  gate requires Kai's approval for new GPU jobs. See decisions.md 2026-07-04.
- Architecture-target decision + binary-pinning verify-first policy + β-fold plan:
  decisions.md 2026-07-04 (Decisions 1–4).
