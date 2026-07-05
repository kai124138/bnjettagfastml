# HGQ2 rebuild — change ledger

Running log of every consequential change in this effort. Dated, newest on top.

## 2026-07-05 (later) — per-instance csynth reports in the store; poster gaps 1/2/7 closed
- A parallel poster session (poster/) froze figures against the store and left a gap
  list (poster/GAPS.md); items addressed to the live session are now closed:
- **Gap 1 — per-function DSP attribution is store-backed.** Raw `csynth.xml` fetched
  from mulder into `results/hgq2/runs/b224a8ea/<probe>/` for all four finished probes;
  new `parse_csynth_modules.py` emits `csynth_modules.json` (per-Module resources +
  latency) beside each. Numbers confirm exactly what the ledger had quoted:
  head_fc2_rf32 → SubLN 112 / binary dense 0 / CSD-2 affine 0 (top 112);
  rf256 Resource → dense 256 / SubLN 14 (top 270); subln_rf1 → SubLN 1792.
- **New per-module fact from attn_core_rf1**: each act×act einsum (QKᵀ and attn·V,
  10×10×8×32 = 25,600 MACs) synthesized to **25,600 DSP — exactly 1 DSP per MAC**;
  the softmax module itself is only 10 DSP. The "~51k real multipliers" statement is
  now precise: 51,200 einsum DSPs + ~800 top-level/softmax glue = 52,000.
- **Gap 7 — head count.** `"n_heads": 8` read directly off the `model_config` h5
  attribute of all three r5 large checkpoints (s1 / a6-s1 / a4-s1). Era-2 large = H8, E=32.
- fetch_mulder_reports.sh now also pulls csynth.xml + regenerates csynth_modules.json
  on every fetch, and probe_attn_core_rf64 added to its PAIRS (fetch-ready when it lands).
- rf64 folded attention core still synthesizing on mulder (monitor armed; no crash).

## 2026-07-05 — attention core SYNTHESIZED (the historically-excluded piece)
- **probe_attn_core_rf1** (large-model core: scores QKᵀ → stable table softmax with
  β_qβ_k/√d in the exp LUT → attn·V; T=10, H=8, E=32; 16-bit input grids, RF=1
  fully spatial): **synthesized on Vitis 2023.2** — 31 cycles @ II=1, est. 1.81 ns,
  LUT 4,271,510 (247% VU13P) / FF 9.47M / **DSP 52,000 (423%)** / BRAM 720.
  Reading: the act×act core converts and synthesizes natively now (EinsumDense
  blocker closed), and its cost at the spatial extreme quantifies WHY attention is
  the piece weight-binarization cannot touch — every one of the ~51k act×act products
  is a real multiplier (no weights to binarize). At 0.65% of the model's MACs it is
  disproportionately DSP-expensive per MAC.
- Folded variant probe_attn_core_rf64 (multiplier_limit = total/64) shipped + csynth
  launched — the deployable-point row. For the SMALL (D32/H4/E8) model the core is
  8× fewer MACs; folded, trivial.

## 2026-07-04 (late) — DSP-0 core VERIFIED in csynth on the HGQ2 path; artifact-contamination bug fixed
- **probe_bitlinear_head_fc2_rf32** (pure ±1 + CSD-2 affine, Strategy=Latency, real
  csynth on mulder): total 194,012 LUT / 116,346 FF / **112 DSP** / 100 cyc / II 32 /
  est. 2.03 ns. Per-function split: binary dense **0 DSP** (23,788 LUT adder trees) ·
  CSD-2 affine **0 DSP** (285 LUT) · SubLN **112 DSP** = 100% of the probe's DSPs.
  → The v4 factoring is validated end-to-end: the DSP-0 binary core reproduces on the
  HGQ2-native path when weights are compile-time constants (Latency) — closing the
  Resource-ROM regression found earlier today.
- **Artifact-contamination bug caught & fixed**: pack_for_mulder tarred the project dir
  *including* a previously-fetched csynth_report.json → the stale v1 report rode inside
  the v3lat tarball to mulder and was fetched back masquerading as v3lat's result
  (numbers bit-identical to v1 exposed it; the remote build was genuinely Latency and
  still synthesizing). Fixes: tar now `--exclude csynth_report.json`; the fetch script
  regenerates the json only from a real csynth.xml and deletes stale ones. Bogus row
  purged from the store/table/dashboard.
- Still in flight on mulder: probe_attn_core_rf1 (attention core, heavy elaboration),
  probe_bitlinear_{v3lat,a6lat,a4lat}_rf256 (Latency elaboration of 65k-MAC layers is
  known-slow; the head_fc2 probe already settles the DSP question at small shape).
- **UPDATE (late evening): ALL THREE big-shape Latency probes FAILED identically** —
  `HLS 200-1715 problem during source synthesis` after ~4 h each: v3lat (A8), a6lat (A6),
  a4lat (A4). The Vitis frontend crash on the fully-unrolled 65k-MAC Latency dense is
  **precision-independent** — the big-shape Latency wall is structural, exactly the
  intractability prior art documented. Negative result recorded in constraints_map.md
  ("Latency at big shapes" row): small shapes → Latency/DSP-0 csynth-verified (head_fc2);
  big shapes → Resource + (future) true 1-bit weight-type emission in the HGQ2 frontend,
  or cite the QKeras-path per-shape DSP-0 numbers. Only attn_core still running.

## 2026-07-04 — session close: verified state + what's in flight
- **Results-analyst verification PASS** (experiment-log 2026-07-04 ✓): all 27 core
  quantities recomputed independently from raw arrays match to full float precision.
- RESEARCH.md updated: §2 status, new §6′ (HGQ2 path), §8 provenance rows.
- Dashboard (read-only store view) generated + published as a Claude artifact;
  regenerate anytime with `generate_dashboard.py`.
- **csynth still RUNNING on mulder at close** (fetch with `./fetch_mulder_reports.sh`):
  probe_attn_core_rf1 (A8 attention core — THE previously-excluded piece; C-sim corr 1.0
  already proven), probe_bitlinear_v3lat_rf256 + a6lat + a4lat (pure-±1 Latency-strategy
  denses — the expected DSP-0-core validation + the LUT-vs-precision axis). Latency-
  strategy elaboration of 65k-MAC layers is slow; expect hours, not minutes.
- **NOT launched (needs Kai)**: round-6-small training (the directive's literal D32/L2
  era-2 model — no trained checkpoint exists at that scale).
  Launch: `code/jobs/training/variants/launch_r6s_staged.sh` (5 PVC-free jobs,
  server-dry-run validated). Once its W&B checkpoints exist, the same pipeline runs it
  end-to-end (new configs + `run_stage.py all`), and the SMALL model can be synthesized
  as ONE monolith per precision — the deployable-scale table the large model can't give.

## 2026-07-04 — v4: DSP regression caught by per-instance csynth breakdown; affine refactor
- **STOP-grade catch**: probe_bitlinear (v3 factoring: in-weight ±β̃ CSD-2) synthesized
  at RF=256 Resource with **DSP=256 in the "binary" dense** (SubLN only 14). Root
  cause: Resource strategy stores weights in BRAM/ROM → runtime operands → the Vitis
  ≤2-signed-digit constant rule NEVER applies. The QKeras path was safe because its
  weights were literally 1-bit. **The binary-DSP-0 claim requires weights ≤2 bits in
  the datapath, regardless of the constant's digit count.**
- **v4 factoring**: ALL denses carry pure ±1 kernels (2-bit operands → mux/negate);
  the 2L+2 explicit β̃·z+b sites become frozen `QBatchNormalization` affines (ε=0,
  var=1 → HW scale = γ = CSD-2 exactly, compile-time constants). Biases move into
  the affines (input_proj: (bias+PE)/β̃ stays as the dense's (T,D) table). Every
  bias/γ quantizer is an explicit wide frozen SAT grid — v3's silent WRAP-default
  bias quantizers were ALSO costing fidelity.
- **v4 verify (A8, full 260k)**: gate1 corr 0.9589, macro-AUC **0.84933 vs trained
  0.85510 (Δ −0.00578)** — better than v3 (−0.00642); gate2 vs gold corr 0.9937 →
  **0.9978** after the bias-quantizer fix (residual = float-order noise; softmax-
  table and stream-margin ablations both negative).
- A6 v3-semantics verify (math identical to v4), full 260k: gate1 corr 0.8907,
  AUC **0.82150 vs trained 0.83943 (Δ −0.01794)**.
- EBOPs (HGQ2-native, trace_minmax): A8 = **634,685,224** vs analytic HGQ-v1
  convention 530.4M (results/ebops.md) — +19.7%, the accumulator term; conventions
  reported side by side, never mixed.
- SubLN folded probe (inside bitlinear probe): DSP=14, LUT ≈ 170k — LUT-heavy
  (unoptimized wide internals + full unroll), timing 3.03 ns > 2.5 target. Levers:
  narrow diff_t/var_t, fold the norm. Deferred; reported as-is.

## 2026-07-04 — A8 rebuild VERIFIED end-to-end (keras side) + first real csynth
- **v1 → v3 debug trail** (each a real measured failure):
  v1 gate1 AUC 0.500 (garbage) — root cause: every unconfigured HGQ2 datalane
  quantizer defaults to WRAP+uncalibrated (QEinsum inputs, QSoftmax exp/inv grids,
  QGAP). v2 fix (explicit frozen SAT grids + QSoftmax table configs) hit an
  UPSTREAM hgq 0.1.9 bug: `enable_iq=False` crashes multi-input layers
  (`_iqs_confs` typo in QLayerBaseMultiInputs.__init__). v3 workaround: exact-
  passthrough frozen grids on the einsum streams (identity on the integer grid,
  EBOPs stays live) + per-block calibrated score/stream ranges.
- **v3 verify (era2-large-w1a8, full 260k val)**: gate1 HGQ2↔trained corr 0.9578,
  macro-AUC **0.84868 vs ref 0.85510 (Δ −0.0064)** — right at the gold-model
  prediction (0.8475). gate2 HGQ2↔gold corr 0.9937 — residual attributed to the
  table-based softmax vs gold's exact float softmax (ablation pending).
- **First csynth on mulder (HGQ2 path)**: probe_subln_rf1 (dim 256, io_parallel,
  II=1): LUT 165,695 (9.6%) · FF 151,297 · **DSP 1,792** · BRAM 0 · 36 cycles @
  est. 1.818 ns. The range-reduced SubLN at II=1 spends ~7 DSP/lane on the 42-bit
  variance squares — the norm remains the model's DSP consumer (context: old
  LayerNorm census 1,049 DSP TOTAL for 51 instances but folded at RF=256; these
  are different operating points). Lever if needed: narrow diff_t/var_t, or fold.
- probe_bitlinear_rf256 (real block-0 Wo weights + SubLN, RF=256 Resource):
  local C-sim corr 0.9999992. Shipped to mulder with probe_attn_core.

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
