# FastML26 poster — Methods & Results draft

**Status: LARGE-MODEL-PROVISIONAL.** Every number in this draft is measured on the
*large* model (D=256, 8 blocks, FFN 1024, ≈6.4 M parameters), reported as
**per-shape / per-probe synthesis measurements**, because the composed large model
(one instance per layer, each internally folded at RF=256) needs **515.8% of a
VU13P's LUTs** and does not fit one device. No deployable single-model synthesis
result exists yet; that table arrives with **r6-small** (D32/L2, job YAMLs
validated, not yet launched). Nothing below may be presented as a deployed
full-model result. **Additionally, the QKeras-path synthesis artifacts carry
era-1-shape provenance** (see the era note in Results): AUC/EBOPs numbers are
era-2 (16-feature, 5-class); the QKeras csynth builds are the era-1 shape variant
of the same architecture.

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

A BitNet-style transformer jet tagger: 8 pre-norm blocks (8-head self-attention
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
  to **2 signed digits (CSD-2, ≈4.5% max error)**, held in a separate frozen affine
  layer — deliberately *not* in the weight values (see the Resource-strategy
  finding below).
- **Custom SubLN hls4ml extension.** Keras LayerNormalization does not convert on
  this stack, so the rebuild ships a custom layer end to end: a `PSubLN` keras
  layer, a new `SubLN` hls4ml IR type with bit-exact-flow registrations, Vitis
  templates, and an `nnet_subln.h` kernel using a range-reduced inverse sqrt
  (even-power-of-two shift onto [1/4, 1), 4096-entry table, half-shift back) —
  C-sim correlation ≥ 0.9999999 against Keras across widths 16–1024 and input
  variances up to 10⁶.
- **hls4ml conversion.** Vitis backend, io_parallel, bit-exact flow — exercised
  **per probe / per block**; the whole-model conversion has not been run (Results,
  last section). The attention score core (QKᵀ, softmax, A·V) expresses natively
  via QEinsum + table-based QSoftmax (C-sim corr 1.0 at A8). Weight-handling
  strategy matters: with
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

Era-2 ROC-test macro-OvR AUC, n = 260,000. **Seed convention: all trained
references are the single seed-1 run (s1)**, matching the store's tradeoff table;
second seeds exist and reconcile (A8 0.8452, A6 0.8220, A4 0.7312). Seed spread is
not negligible: at A6 it is 1.7 points — as large as the A8→A6 step itself — and
the W1A6 rebuild Δ (−0.017) is of the same order as trained seed noise at A6.

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
promised: A8 → A6 costs 1.6 points (trained, s1; 2.3 on s2), A6 → A4 costs 10.5 —
**efficiency is largely retained through A6 and collapses by A4** (with three
sampled precisions the collapse can only be located between A6 and A4, not finer). The rebuild gap grows as activations narrow (−0.006 / −0.017 / −0.023)
because the static-grid substitution for BitNet's dynamic per-token scaling bites
harder at low bits; the store's ablations attribute most of the A4 gap to exactly
this substitution, making trained-static-quant (HGQ2 QAT) the identified fix —
future work. Per-class ROC overlays (HEP convention, log mistag axis) in Fig. 2:
the rebuild tracks the trained curve family across all five classes, with the
visible spread concentrated in the W/Z discrimination at low mistag.

### DSP: the binary matmul is free; in the dense+norm stack, the norm carries the cost — Fig. 3

**Era note (applies to every QKeras-path synthesis number in this draft).** The
QKeras-path csynth artifacts — the binary FFN block, the per-shape probes, the
composed census, and the latency composition — were synthesized 2026-06-24/26 from
the **era-1 shape variant** of this architecture (14-feature input projection,
single-logit head). 49 of the 51 layer instances, including every FFN and
attention-projection shape, are shape-identical in the era-2 model; the two that
differ are input_proj (14→256 vs 16→256) and the head output (256→1 vs 256→5).
The DSP conclusions are structural (weight-value- and activation-precision-
independent) and are corroborated on the era-2 HGQ2 path below; the composed
LUT/latency totals are quoted as **era-1-shape measurements** — no era-2-shape
full composition has been synthesized (that is the r6-small deliverable).

Whole-probe Vitis HLS 2023.2 csynth totals on the VU13P, with per-function splits
re-derived from the raw per-instance module tables (`csynth.xml`, fetched into the
store 2026-07-05 — VERIFICATION.md §5):

- Binary FFN block (fc1 256→1024 → ReLU → fc2, **no norm in the probe**; QKeras
  path, era-1-shape build, block shape era-identical), RF=256: **0 DSP at A8, A6,
  and A4** — with LUT 440,882 / 429,098 / 415,259 (25.5% → 24.0% of the device),
  520 cycles, II=256, 400 MHz met (est. 1.76 ns).
- SubLN alone (HGQ2 path, era-2 build; dim 256, fully unrolled, II=1): **1,792
  DSP** (14.6% of the device's 12,288), 36 cycles.
- SubLN + binary dense (256→5) + CSD-2 affine, folded (HGQ2 path, Latency,
  RF=32): **112 DSP for the whole three-layer chain — and the module table
  attributes all 112 to the SubLN: binary dense 0 DSP (23,788 LUT of adder
  trees), CSD-2 affine 0 DSP (285 LUT).**
- **The ROM trap (negative result, different build — not a controlled strategy
  flip):** an earlier factoring that carried β̃ *inside the weight values* (v3;
  SubLN + 256→256 dense, Resource, RF=256, no separate affine) synthesized to
  **270 DSP = 256 in the "binary" dense + 14 in the folded SubLN** (est. 3.04 ns,
  missing the 2.5 ns target). Weights stored in BRAM become runtime operands, so
  the ≤2-signed-digit constant rule never applies. Lesson: keep the datapath
  weights pure ±1 and the scale in a separate constant affine — csynth-proven
  DSP-free at small shape under Latency; making it hold at the big-shape folded
  point (Resource) still requires true 1-bit weight-type emission in the HGQ2
  frontend (the bounded next engineering item).

Two raw corroborations sharpen the dichotomy: per-shape probe DSP is **identical
at A8/A6/A4** for every layer shape (the DSP block does not scale with activation
precision, while LUT does), and the composed 51-instance census (dense+norm stack
only — **attention score core excluded** — each instance folded at RF=256,
era-1-shape build) puts that stack's DSP at **1,049 = 8.5% of a VU13P** at every
precision. The abstract's structural claim — binary matmul maps to LUT/logic, not
DSPs — holds in real synthesis. Within the dense+norm stack the residual DSP cost
is the normalization: measured per-function on the HGQ2 probes (all 112 of the
folded chain's DSPs sit in SubLN), and consistent at census level with the DSPs'
precision-independence and LN-width scaling (the per-instance attribution for the
QKeras-path census itself still awaits its per-module reports — GAPS.md).

**The attention score core — the piece binarization cannot touch, now measured**
*(large-model-provisional)*. The weightless act×act core (QKᵀ → softmax → attn·V;
no weights, so nothing to binarize) synthesized at the fully-spatial extreme
(II=1): **31 cycles at est. 1.812 ns, but 4.27 M LUT (247% of the device) and
52,000 DSP (423%)** — each of the two einsums costs **exactly 1 DSP per MAC**
(25,600 each; softmax 10, remaining 790 in top-level glue). This is the measured
quantification of why the binary-weight win targets the linear layers, and why
keeping N small (top-10 truncation, act×act = 0.65% of MACs) is the architectural
defense: the core's cost is real but its share is bounded by construction. A
folded (RF=64) variant was still synthesizing at freeze; no deployable-point
number for this core is quoted yet.

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
along the critical path (one instance per layer, each internally folded at RF=256
— *not* a fully-unrolled design) gives an **upper bound of 23,409 cycles ≈ 58.5 µs
at the 2.5 ns target** for the era-1-shape large model (attention score core
excluded from that composition). Two caveats: the input-projection stage on that
path closed timing at 3.71 ns, not 2.5 ns, so the composed bound is not
timing-closed at 400 MHz as synthesized; and this is an upper bound on an
intentionally oversized model, not a trigger-latency claim — the L1-scale
statement belongs to r6-small.

### What does NOT exist yet (stated on the poster)

- **No deployable single-FPGA result:** composed one-instance-per-layer (each
  folded at RF=256), the large model is 515.8% of a VU13P's LUTs. The
  deployable-scale, single-monolith-per-precision table is blocked on r6-small
  training (jobs validated, awaiting launch approval).
- **No era-2-shape full composition:** the QKeras-path census and latency
  composition are era-1-shape builds (era note above); the era-2 shapes differ in
  2 of 51 instances. No number was invented to bridge this — the era-2-shape
  synthesis arrives with r6-small or a re-run of the shape probes.
- **No whole-model HGQ2 hls4ml conversion/C-sim:** rebuild fidelity is verified at
  the score level (Δ, corr above); the end-to-end converted model has not been
  run. The attention core is measured only at the fully-spatial extreme (rf1);
  its folded deployable point (rf64) was still synthesizing at freeze, and the
  composed 23,409-cycle latency bound still excludes it.
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
