# FastML26 poster — Methods & Results draft

**Status: LARGE-MODEL-PROVISIONAL.** Every number in this draft is measured on the
*large* model (D=256, 8 blocks, FFN 1024 — 6.37 M parameters), reported as
**per-shape / per-probe synthesis measurements**, because the fully-spatial large
model composes to **515.8% of a VU13P's LUTs** and does not fit one device. No
deployable single-model synthesis result exists yet; that table arrives with
**r6-small** (D32/L2, job YAMLs validated, not yet launched). Nothing below may be
presented as a deployed full-model result.

Every number here passed the raw-data verification gate — `poster/VERIFICATION.md`.
Anything the store could not reconcile is absent (see `poster/GAPS.md`).

---

## Anchor (abstract, as submitted)

The poster's claims trace to the submitted abstract: a binary-weight, BitNet-style
1-bit transformer jet tagger whose {−1,+1} matrix multiplications map onto FPGA
LUT/logic fabric rather than scarce DSPs, evaluated on (a) tagging efficiency vs
vanilla FP32 and W8A8 baselines and (b) how aggressively activations can be
quantized (A8 → A6 → A4) before efficiency, resources, or latency degrade.

---

## Methods

### Model and task *(large-model-provisional)*

A BitNet-style transformer jet tagger: 8 pre-norm blocks (multi-head self-attention
+ FFN, both with binary {−1,+1} weights), D=256, FFN 1024, followed by global
average pooling and a two-layer head; 5 output logits. Data: the public HLS4ML LHC
Jet 150-particle dataset (Zenodo record 3602260), 5 classes (gluon, light quark, W,
Z, top). Input: the **top 10 constituents by pT × 16 features = 160 inputs**.

**Input-truncation defense.** The top-10-by-pT truncation follows advisor guidance
(2026-07-01): a small fixed input matched to the L1 latency/resource envelope, with
the input-size axis (`BN_N_PART`) an explicitly deferred study — stated future
work, not a gap. Attention's act×act score cost is O(N²) in constituents; at N=10
it is 409,600 MACs/jet = **0.65% of the model's 63.4 M MACs**, so the non-binary
attention core stays negligible by construction.

### BitNet binarization (training)

Weights binarize by AbsMean (BitNet, arXiv:2310.11453): α = mean(W),
W_c = W − α, β = mean|W_c|, W_q = sign(W_c), applied as W_q·β with a
straight-through estimator against FP32 latent weights. Activations quantize
per-token symmetric absmax at b ∈ {8, 6, 4} (A8/A6/A4), also with an STE.
Normalization is a parameter-free sub-layer LayerNorm (SubLN) inside every
BitLinear. Training: Adam, categorical cross-entropy on logits, early stopping;
all era-2 runs share one recipe (round-5).

### HGQ2 binary-pinned rebuild (the hls4ml-facing model)

The trained checkpoints are rebuilt in HGQ2 0.1.9 / hls4ml 1.3.0 so that what is
synthesized is bit-defined fixed-point, not a TF emulation:

- **Binary pinning.** Kernels enter as ±1 through a frozen 1-bit quantizer
  (`KBI(k0=1, b0=1, i0=1, SAT_SYM)`), passing the trained signs through
  bit-identically; weights are 2-bit ±1 operands in the datapath.
- **Static activation quantization.** BitNet's *dynamic per-token* activation
  scaling has no static fixed-point equivalent, so the rebuild substitutes
  **frozen per-channel grids, MSE-calibrated** on 8,192 jets. The substitution is
  measured, not assumed free — it is the dominant term in the A4 fidelity gap
  (Results).
- **Fold-aware β.** The per-tensor BitNet β is never left as a runtime multiply:
  Q/K β fold exactly into the softmax exp-table input scale; V's β is dropped
  exactly (the following norm is scale-invariant); fc1/head_fc1 β fold into the
  bias; only the 2L+2 residual-contributing sites carry an approximate β̃ snapped
  to **2 signed digits (CSD-2, ≤4.5% error)**, held in a separate frozen affine
  layer — deliberately *not* in the weight values (see the Resource-strategy
  finding below).
- **Custom SubLN hls4ml extension.** Keras LayerNormalization does not convert on
  this stack, so the rebuild ships a custom layer end to end: a `PSubLN` keras
  layer, a new `SubLN` hls4ml IR type with bit-exact-flow registrations, Vitis
  templates, and an `nnet_subln.h` kernel using a range-reduced inverse sqrt
  (even-power-of-two shift onto [1/4, 1), 4096-entry table, half-shift back) —
  C-sim correlation ≥ 0.9999999 against Keras across widths 16–1024 and input
  variances up to 10⁶.
- **hls4ml conversion.** Vitis backend, io_parallel, bit-exact flow; attention
  score core (QKᵀ, softmax, A·V) expressed natively via QEinsum + table-based
  QSoftmax (C-sim corr 1.0 at A8). Weight-handling strategy matters: with
  **Resource** (weights in BRAM/ROM) any in-weight scale becomes a runtime operand
  and Vitis infers real multipliers; with **Latency** (weights inlined as
  constants) the pure-±1 datapath synthesizes multiplier-free. This is a measured
  finding, not a convention (Results, DSP).

### Verification protocol

Every AUC was recomputed from raw score/label arrays (identical 260,000-jet
held-out split for every model — label arrays bit-identical across files); EBOPs
totals were re-summed from per-layer payloads and the analytic table re-derived
from architecture dimensions; resource/latency numbers were read from raw Vitis
csynth reports, with probe composition confirmed from the generated firmware.
Procedure and rulings: `poster/VERIFICATION.md`.

---

## Results

### Tagging efficiency across the quantization axis *(large-model-provisional)* — Fig. 1, Fig. 2

Era-2 ROC-test macro-OvR AUC, n = 260,000:

| model | trained | HGQ2 rebuild | Δ |
|---|---|---|---|
| FP32 (vanilla) | 0.8765 | — | — |
| W8A8 | 0.8642 | — | — |
| **W1A8** | 0.8551 | **0.8493** | −0.0058 |
| **W1A6** | 0.8394 | **0.8222** | −0.0173 |
| **W1A4** | 0.7346 | **0.7115** | −0.0231 |

Going binary costs **−2.1 AUC points vs FP32** (0.8765 → 0.8551) and −0.9 vs W8A8
at trained A8. The hls4ml-facing rebuild preserves that to within −0.6 points at
A8 (score correlation 0.959). The activation axis is the story the abstract
promised: A8 → A6 costs 1.6 points (trained), A6 → A4 costs 10.5 — **the knee is
at A6**. The rebuild gap grows as activations narrow (−0.006 / −0.017 / −0.023)
because the static-grid substitution for BitNet's dynamic per-token scaling bites
harder at low bits; the store's ablations attribute most of the A4 gap to exactly
this substitution, making trained-static-quant (HGQ2 QAT) the identified fix —
future work. Per-class ROC overlays (HEP convention, log mistag axis) in Fig. 2:
the rebuild tracks the trained curve family across all five classes, with the
visible spread concentrated in the W/Z discrimination at low mistag.

### DSP: the binary core is free; the norm is the entire cost — Fig. 3

Whole-probe Vitis HLS 2023.2 csynth totals on the VU13P (each row is one complete
synthesis; no per-function attribution is needed for this claim):

- Binary FFN block (fc1 256→1024 → ReLU → fc2, **no norm in the probe**), RF=256:
  **0 DSP at A8, A6, and A4** — with LUT 440,882 / 429,098 / 415,259 (25.5% →
  24.0% of the device), 520 cycles, II=256, 400 MHz met.
- SubLN alone (dim 256, fully parallel, II=1): **1,792 DSP** (14.6% of the
  device's 12,288), 36 cycles.
- SubLN + binary dense + CSD-2 affine, folded (Latency, RF=32): **112 DSP for the
  whole three-layer chain**.
- The same chain under Resource at RF=256: **270 DSP** — the ROM-trap negative
  result: weights stored in BRAM become runtime operands and the ≤2-signed-digit
  constant rule never applies. The deployable recipe is therefore pure-±1
  datapath weights + the scale in a separate constant affine.

Two raw corroborations sharpen the dichotomy: per-shape probe DSP is **identical
at A8/A6/A4** for every layer shape (the DSP block does not scale with activation
precision, while LUT does), and the composed 51-layer census puts the model's
entire DSP footprint at **1,049 = 8.5% of a VU13P** at every precision. The
abstract's structural claim — binary matmul maps to LUT/logic, not DSPs — holds in
real synthesis; what remains on DSPs is normalization, which is a known, bounded
engineering target (narrower internals, folding, or norm elimination).

### Compute: EBOPs — Fig. 1, Fig. 4

HGQ2-native convention (accumulator term in): **650.4 M / 501.2 M / 352.2 M** for
the rebuilt W1A8 / W1A6 / W1A4. In the separate analytic convention (bit-product
sum, no accumulator — the only convention in which all five models are defined):
FP32 64.95 G, W8A8 4.06 G, W1A8 530.4 M — binary weights buy **7.65× fewer EBOPs
than W8A8** and 122× below FP32. The two conventions are never mixed in any table
or figure.

### Latency *(large-model-provisional)*

The folded binary FFN block runs 520 cycles (1.3 µs at 400 MHz, II=256, timing met
with margin — estimated clock 1.76 ns). Composing per-shape worst-case latencies
along the critical path gives a fully-spatial streamed **upper bound of 23,409
cycles ≈ 58.5 µs at 400 MHz** for the large model (attention score core excluded
from that composition). This is an upper bound on an intentionally oversized
model, not a trigger-latency claim: the L1-scale statement belongs to r6-small.

### What does NOT exist yet (stated on the poster)

- **No deployable single-FPGA result:** fully-spatial, the large model is 515.8%
  of a VU13P's LUTs. The deployable-scale, single-monolith-per-precision table is
  blocked on r6-small training (jobs validated, awaiting launch approval).
- **No whole-model HGQ2 hls4ml conversion/C-sim:** rebuild fidelity is verified at
  the score level (Δ, corr above); the end-to-end converted model has not been
  run. Attention-core csynth was still in flight on mulder at freeze.
- **Big-shape Latency synthesis is structurally intractable** (Vitis frontend
  fails on fully-unrolled 65k-MAC denses after ~4 h, precision-independent) —
  documented negative result; big shapes fold under Resource, where DSP-0 on the
  HGQ2 path additionally needs true 1-bit weight-type emission (bounded next
  engineering item; the QKeras-path probes already prove DSP-0 at Resource/RF=256
  for those shapes).

---

*All numbers: poster/VERIFICATION.md (raw-data recomputation, 2026-07-04). Figures:
poster/figures/fig1–fig4 (vector PDF/SVG). Gaps and unverifiable attributions:
poster/GAPS.md.*
