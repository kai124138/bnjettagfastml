# Research Log — BNJetTag

Append-only. **Newest entries on top.** Every claim needs a source URL.
Written mainly by the `physics-researcher` agent.

Format:

```
## YYYY-MM-DD — <question / topic>
- Finding. (source: https://…)
- Finding. (source: https://…)
- Relevance to us: …
```

---
<!-- new entries go below this line, newest first -->

## 2026-07-04 — HGQ2, EBOPs, hls4ml/da4ml integration, and binary-weight pinning API (scoping the HGQ2 rebuild)

**Question:** For rebuilding BNJetTag in HGQ2 (successor to calad0i/HGQ): exact package/version/
install, quantized-layer API (Dense/MHA/norm/activation), how to pin a weight quantizer to hard
binary `{-1,+1}` and freeze bitwidths, EBOPs semantics/API, hls4ml version compatibility and
MHA/softmax/LayerNorm conversion status, da4ml's role, and any published HGQ2+attention+hls4ml
end-to-end example.

**Primary sources used:** GitHub (calad0i/HGQ2), PyPI JSON API (HGQ2, quantizers), the official
HGQ2 docs (calad0i.github.io/HGQ2), fastmachinelearning.org/hls4ml docs (hgq.html, release_notes,
autodoc for `hls4ml.converters.keras_v3.hgq2`), and — highest confidence, full PDF read — **Chang
Sun's "Tutorial on HGQ and hls4ml" (Caltech/PMA, NGT Tutorial, 2025-11-10)**, fetched from
https://indico.cern.ch/event/1593461/contributions/6715101/attachments/3170139/5635793/material.pdf
(47 slides, read verbatim with the Read tool — this is the single richest source found and is
cited heavily below as "[Sun tutorial, slide N]").

---

### 1. Package identity, version, requirements

- **PyPI name:** `HGQ2` (import as `hgq`). **Install:** `pip install HGQ2`.
  (source: https://pypi.org/project/HGQ2/, https://github.com/calad0i/HGQ2, accessed 2026-07-04)
- **Latest version (PyPI JSON, accessed 2026-07-04):** `0.1.9` (dev builds tagged e.g.
  `0.1.9.dev16+g24f3b5f85` on the docs site, i.e. still pre-1.0/actively developed).
  (source: https://pypi.org/pypi/HGQ2/json, https://calad0i.github.io/HGQ2/)
- **Python requirement:** `>=3.10`. **Core deps:** `keras>=3.11`, `quantizers>=1.2.2`, `tqdm`.
  **Optional/test deps:** `da4ml>=0.6`, `hls4ml>=1.2`, `jax>=0.4.20`, `jaxlib>=0.4.20`.
  (source: https://pypi.org/pypi/HGQ2/json, accessed 2026-07-04)
- **Keras/backends:** built on **Keras v3**, supporting **all three backends: JAX, TensorFlow,
  PyTorch** [Sun tutorial, slide 4: "Keras v3 based, supporting all jax/tensorflow/torch
  backends"]. **JAX is the recommended backend for training** — "strongly recommended using the
  jax (preferred) or tensorflow (less preferred) backends for model training with the native
  `model.fit` API for the best GPU utilization"; fake-quantization is many cheap elementwise ops
  and XLA `jit_compile` fuses them for a large speedup; native `torch` training loop works but is
  not as optimized, and `torch.compile`/dynamo support is "not yet available" [Sun tutorial,
  slides 4, 17]. **Do `import keras`, not `from tensorflow import keras`** [slide 46].
  (source: Sun tutorial slides 4, 17, 46, accessed 2026-07-04)
- **This is the direct successor to `calad0i/HGQ`** (now archived/legacy: "Legacy High Granularity
  Quantization 1 — Please use HGQ2 instead"). (source: https://github.com/calad0i/HGQ, accessed
  2026-07-04)
- **hls4ml version needed:** HGQ2 native support landed in **hls4ml v1.2.0 ("hyacinth"),
  released 2025-11-03** — release notes: "Distributed Arithmetic strategy implementations for
  Dense, Conv1/2D, and EinsumDense, and HGQ2 support." **v1.3.0 ("iris"), released 2026-03-20**
  added further fixes ("hgq2 homogeneous quant fix", "Da custom layer"). **LayerNorm support for
  the Vivado backend was also added in v1.2.0.** (source:
  https://fastmachinelearning.org/hls4ml/intro/release_notes.html, accessed 2026-07-04)

---

### 2. Quantized layer API and how quantization is applied

- **Layer classes confirmed** (from HGQ2 docs index + hls4ml's HGQ2 converter autodoc, which is
  the more authoritative list since it enumerates exactly what hls4ml can ingest):
  `QDense`, `QDenseT`, `QBatchNormDense`; `QConv1D/2D/3D`, `QConvT1D/2D`;
  **`QMultiHeadAttention`** (module path `hgq.layers.attn.mha.QMultiHeadAttention`) and
  **`QLinformerAttention`** (`hgq.layers.attn.linformer.QLinformerAttention`, linear-attention
  variant); `QEinsumDense`, `QEinsumDenseBatchnorm`, and the generic `QEinsum`
  (`hgq.layers.ops.einsum.QEinsum`); `QSoftmax` (`hgq.layers.softmax.QSoftmax`); pooling variants
  (`QMaxPooling1/2/3D`, `QAveragePooling1/2/3D`, global versions); `QBatchNormalization`;
  activation quantizers `QUnaryFunctionLUT` / `QAffinedUnaryFunctionLUT`
  (`hgq.layers.activation.QUnaryFunctionLUT`); `QGRU`, `QSimpleRNN`.
  (source: https://calad0i.github.io/HGQ2/,
  https://fastmachinelearning.org/hls4ml/autodoc/hls4ml.converters.keras_v3.hgq2.html, accessed
  2026-07-04)
- **hls4ml's HGQ2 converter (`hls4ml.converters.keras_v3.hgq2`) registers exactly these handler
  classes** — i.e. this is the ground truth for what's actually convertible today:
  `QEinsumHandler`, **`QMultiHeadAttentionHandler`** (and `QLinformerAttentionHandler`, which
  subclasses it), `QPoolingHandler`, **`QSoftmaxHandler`**, `QUnaryLUTHandler`. `QDense`/`QConv`
  are handled by the generic (non-HGQ2-specific) Keras-v3 converters since they're structurally
  ordinary Keras layers with quantizer-wrapped weights. (source:
  https://fastmachinelearning.org/hls4ml/autodoc/hls4ml.converters.keras_v3.hgq2.html, accessed
  2026-07-04)
- **Yes — native MHA exists and is bit-accurate.** HGQ2 docs / search summaries state the
  MultiHeadAttention layer supports "bit-accurate softmax and scaled dot-product attention," and
  the paper **arXiv:2510.24784 ("Sub-microsecond Transformers for Jet Tagging on FPGAs") is
  literally the paper that "add[ed] multi-head attention and linear attention support to
  hls4ml"** (verbatim from its abstract) — i.e. HGQ2's `QMultiHeadAttention` +
  `QMultiHeadAttentionHandler` is the mechanism by which mainline hls4ml gained MHA support, not
  a side fork. (source: https://arxiv.org/abs/2510.24784, accessed 2026-07-04; already logged
  2026-07-02 with Table 1 numbers, this session confirms it's the hls4ml-MHA-support paper itself)
- **Quantization is applied at layer INPUT, not output** (opposite convention from QKeras, which
  quantizes outputs): "Quantization is applied at the input of `Q-` layers, instead of the
  outputs like QKeras. This ensures all values engaged in non-trivial operations are quantized."
  Output quantizer (`oq`) can be switched on per-layer if needed; input quantizer (`iq`) is on by
  default. If mixing in a raw QKeras layer, you must add an explicit quantizer after it. [Sun
  tutorial, slides 9, 14, 31]
- **Model-building rule:** use `hgq.layers` as much as possible; **never call `keras.ops.matmul`
  / `a @ b` directly on Keras tensors, and never use `lambda`** — both break `da4ml`/`hls4ml`
  conversion. Any layer with a non-trivial multiply must be a quantized layer or hand-built from
  quantized primitives — **"even MHA can be constructed this way!"** (i.e., MHA is itself just
  `QEinsumDense` + `QSoftmax` + quantized matmuls under the hood). `QEinsumDense` is called out as
  the general tool for building custom complex ops; `QEinsumDenseBatchnorm` is preferred over
  composing `QEinsumDense`+`QBatchNormalization` separately because of batchnorm folding
  stability. [Sun tutorial, slides 15, 31, 46]

---

### 3. Quantizer parametrization (kbi/kif) — the exact "PIN to bits=N" API

- **Fixed-point numbers have 3 attributes** (this is the entire quantizer vocabulary in HGQ2):
  `i` = integer bits, `f` = fractional bits, `k`/`keep_negative` = sign bit (0 or 1). **Total
  width = k + i + f.** Representable range = `[-k·2^(i-1), 2^(i-1) - 2^-f]`, step `2^-f`. Both `i`
  and `f` may be negative (e.g. `f=-3` for large-magnitude-only numbers) as long as total bits
  ≥ 1. [Sun tutorial, slide 2 — this is the foundational fixed-point convention hls4ml/HGQ use
  throughout, including `ap_fixed<W,I,signed>` on the C++ side]
- **Two quantizer parametrizations, selected via `default_q_type` / `QuantizerConfig` first
  positional arg:**
  - **`'kbi'`** — `keep_negative, bitwidth (b, EXCLUDING sign), integers (i)`. This is the
    **default**.
  - **`'kif'`** — `keep_negative, integers (i), fractional (f)`.
  - Rule of thumb: if `overflow_mode='WRAP'`, use `'kif'` for **activation-like** (`datalane`)
    quantizers; for `SAT`/`SAT_SYM` the choice barely matters. [Sun tutorial, slide 10]
- **Quantizers are configured in exactly 4 "places" (`place=` kwarg):** `datalane` (the
  activation-like input/output quantizer), `weight` (kernel), `bias`, `table` (LUT quantizer for
  `QUnaryFunctionLUT`/table-based ops). `datalane` is "Activation-like" (bounds not known at
  synthesis time — value range must be **calibrated post-training** if `WRAP` is used, since
  `WRAP` is *not enforced during training*, only estimated from data seen); the other three are
  "Weight-like" (known at synthesis time, so no calibration step needed). [Sun tutorial, slide 10,
  11]
- **Config API objects:** `QuantizerConfigScope` (context manager, hierarchical — `place='all'`
  sets a global default, more specific `place=` scopes override it) and `QuantizerConfig`
  (per-layer override object) and `LayerConfigScope` (`enable_ebops=True`, `beta0=...`). Verbatim
  example from the tutorial [Sun tutorial, slide 13]:
  ```python
  with (
      QuantizerConfigScope(place='all', overflow_mode='SAT_SYM'),
      QuantizerConfigScope(place='datalane', default_q_type='kif', overflow_mode='WRAP'),
      LayerConfigScope(enable_ebops=True, beta0=1e-5),
  ):
      oq_conf = QuantizerConfig('kif', 'datalane', fr=fr, ir=ir)
      model = keras.Sequential([
          QDense(64, activation='relu'),
          QDense(32, activation='relu'),
          QDense(32, activation='relu'),
          QDense(5, enable_oq, oq_conf=oq_conf),
      ])
  ```
- **Overflow modes (3):** `WRAP` (drop MSBs, 0 overhead, corrupts result on overflow — used
  during training for `datalane` so the network self-calibrates its integer-bit range), `SAT`
  (saturate, high overhead), `SAT_SYM` (saturate, symmetric lower bound `-k·(2^i - 2^-f)` instead
  of `-k·2^i`). **Rounding modes (3):** `TRN` (truncate/floor, 0 overhead), `RND` (round, ties up),
  `RND_CONV` (round, ties-to-even, "still negligible" overhead vs `SAT`). [Sun tutorial, slide 8]

---

### 4. Pinning to hard 1-bit binary `{-1,+1}` and freezing bitwidths (the key question for us)

**Two distinct routes exist — this is the load-bearing nuance for BNJetTag's rebuild:**

**(a) The generic fixed-point route (`kbi`/`kif`, frozen, not truly `{-1,+1}`).** HGQ2's own
"Bonus: Emulate QKeras" slide gives the canonical freeze recipe [Sun tutorial, slide 16,
verbatim]:
- Turn ON the output quantizer (`oq`), turn OFF the input quantizer (`iq`) for `Q-` layers (to
  match QKeras's output-side convention).
- If `alpha=1` (no learned scale): **set `trainable=False`, `overflow_mode='SAT'`,
  `round_mode='RND_CONV'`, and set the desired bitwidths via the quantizer's initial-value kwargs
  `k0, i0, f0`** (the `0`-suffixed kwargs are the *initial, and — with `trainable=False` — final*
  values of `k`, `i`, `f`; this is the literal "freeze the bitwidth" mechanism: initialize to the
  target width and disable the gradient that would otherwise let HGQ2 learn/shrink/grow it).
- To emulate `auto_pow2` "correctly": set quantizer type to `kbi` for all, set `k0='signed'`, set
  the bitwidth constraint `bc=hgq.constraints.Constant(b)` (from `hgq.constraints` — also has
  `Min`, `Max`, `MinMax` for bounding rather than pinning a learned bitwidth), `overflow_mode=
  'WRAP'`, `round_mode='RND_CONV'`.
- **Caveat for true `{-1,+1}` binary via this route:** a `kbi`/`kif` fixed-point quantizer with
  `k=1` (signed) and total width forced down to its minimum still represents a 2-valued or
  3-valued set that **includes 0** by construction (fixed-point always spans a symmetric-about-
  zero range at a given step) — it is not guaranteed to collapse to the literal *bipolar, no-zero*
  `{-1,+1}` set that BitNet's `sign()` function produces. Freezing `i0=1, f0=0, k0=1` gives a
  1-bit-plus-sign representation whose exact representable set depends on the overflow/round mode
  and needs to be checked numerically before assuming it matches BitNet's `{-1,+1}` — **flag for
  the ml-engineer to verify empirically** (e.g. print the quantizer's representable value set)
  before treating "kbi frozen to 1 bit" as equivalent to our current QKeras `binary()` quantizer.
- (source: Sun tutorial, slide 16, accessed 2026-07-04)

**(b) The dedicated `Binary`/`binary_quantize` route (the `quantizers` package, HGQ2's own
runtime-quantizer dependency) — the closer match to BNJetTag's actual thesis.** The separate
`quantizers` PyPI package (by the same author, a required HGQ2 dependency, `quantizers>=1.2.2`)
ships **dedicated non-fixed-point quantizer functions**: `binary_quantize()` — described as
**"Maps to {-1,1} with 0 to -1"** — alongside `ternary_quantize()`, plus the general
`get_fixed_quantizer()`/`FixedQ` (fixed-point) and `float_quantize()`/`MinifloatQ` (minifloat).
**Important flag: the PyPI listing itself describes the binary quantizer as a "preliminary
implementation"** — i.e. not yet a fully mainline/first-class path through HGQ2's own
`QuantizerConfig`/`kbi`/`kif` abstraction (which only documents `'kbi'`/`'kif'` as the two
`default_q_type` choices, with no `'binary'`/`'ternary'` string type surfaced in the tutorial's
config API). **Practical read:** achieving BNJetTag's literal hard `{-1,+1}` BitNet weight in
HGQ2 most likely means either (i) freezing a `kbi` quantizer to its minimum width per route (a)
above and empirically confirming the value set, or (ii) wiring in `quantizers.binary_quantize`
directly as a custom weight quantizer on a `QDense`/`QEinsumDense` layer (bypassing the default
`QuantizerConfig` machinery) — **this is a code-level design decision for `ml-engineer`, not
something we can resolve from documentation alone; the "preliminary" label on `binary_quantize`
means it should be tested for hls4ml/da4ml conversion compatibility before committing to it.**
(source: https://pypi.org/project/quantizers/, accessed 2026-07-04)

- **General "don't touch precision" warning:** hls4ml runs **model-wide precision propagation**
  (a pass derived from "quantized interval arithmetic": every variable carries `[low, high,
  step]`, propagated through every op) automatically for HGQ2/QKeras models — "Don't touch the
  precision configuration for hls4ml... No need to tune them in most cases." Code:
  `bit_exact.py` in hls4ml. [Sun tutorial, slides 31, 32]

---

### 5. EBOPs — exact HGQ2 formula (a refinement vs. our 2026-06-29 HGQ-v1 log entry) and API

- **HGQ2's own tutorial gives the formula slightly differently from the original HGQ paper's Eq.5
  (already logged 2026-06-29) — HGQ2 explicitly ADDS an accumulation term, unlike the original
  HGQ, which claimed accumulation was only "implicitly counted":**
  $$\text{EBOPs} = \sum_{\times} bw_i \cdot bw_j \;+\; \sum_{+} \max(bw_i, bw_j)$$
  "for all multiplication operations in the network, where $bw_i$ and $bw_j$ are the bitwidths of
  the two operands. **In HGQ2, we also add EBOPs for ADD/SUB**, as well as a very rough estimation
  of the table size for lookups into the calculation." [Sun tutorial, slide 6 — this is a
  **correction/refinement to our 2026-06-29 log entry**, which (reading the original HGQ v1 paper,
  arXiv:2405.00645) stated accumulation was NOT independently added; HGQ2 explicitly changes this
  by adding a `max(bw_i,bw_j)` term per addition, matching the "adder tree" intuition from the
  BOPs/UNIQ literature we logged that same day]
- **Resource mapping unchanged in spirit:** "If `da4ml` is NOT used, on Xilinx FPGAs, EBOPs ≈
  LUT + 55×DSP with hls4ml synthesized models" — same empirical coefficient (55) as HGQ v1,
  validated across three benchmark families (Jet Classifier, SVHN Classifier, Muon Tracker) shown
  spanning EBOPs ≈ 10² to 10⁵ with tight linear correlation on a log-log plot. **If `da4ml` IS
  used, EBOPs instead upper-bounds LUT alone** (no DSP term, since da4ml/distributed-arithmetic
  designs use zero DSP by construction). [Sun tutorial, slide 6; also stated in the
  fastmachinelearning.org/hls4ml/advanced/hgq.html doc page, accessed 2026-07-04]
- **API:** enable via `LayerConfigScope(enable_ebops=True, beta0=...)` at model-build time — this
  is a **training-time-only mechanism**: `beta0`/`beta` weights an EBOPs regularization term added
  to the loss ($\mathcal{L} = \mathcal{L}_0 + \beta \cdot \text{EBOPs} + \gamma \cdot \sum
  \text{bit-widths}$), and gradients flow from both the task loss (pushes bitwidth up) and the
  EBOPs term (pushes bitwidth down) into each quantizer's `f`/`b` parameter (a genuinely
  differentiable quantity — the tutorial derives $\partial L/\partial f$ explicitly). It is
  **recommended to schedule `beta` dynamically via `BetaScheduler` from `hgq.utils.sugar`** rather
  than fixing it. [Sun tutorial, slides 5, 14] **EBOPs can also be READ post-hoc on a trained (or
  converted) model** — "If the model is trained with HGQ, check EBOPs to estimate resource usage"
  before committing to a synthesis run [Sun tutorial, slide 45] — i.e. the number is a model
  attribute you can query without re-running training, though the *reason* it exists as a useful
  proxy only holds if the model was actually trained with the EBOPs regularizer switched on (an
  un-regularized model's EBOPs is just a descriptive count of $\sum bw_i \cdot bw_j$, still
  computable, just not something training optimized against).

---

### 6. hls4ml integration mechanics — strategies, reuse_factor, and a genuinely new (to us) finding
about WHEN binary/low-bit weights avoid DSPs entirely — direct, mechanistic confirmation of our
own "binary → 0 DSP" thesis

- **hls4ml offers 4 synthesis strategies for the core constant-matrix-vector-multiply (CMVM)
  operation** (Vitis backend): `Latency` (default; unroll everything, `reuse_factor` = target II,
  "weights may exist nowhere in binary form" i.e. can be fully constant-folded into logic),
  `Resource` (pipeline via `reuse_factor`, weights go to BRAM, one processing element reused),
  `resource_unrolled` (Vitis-only; like Resource but exploits structured pruning), and
  **`distributed_arithmetic` (Vitis + OneAPI only; implemented by the external `da4ml` library;
  only `reuse_factor=1` supported; "No DSP usage in general, similar or lower LUT usage than
  Latency")**. [Sun tutorial, slides 21–23]
- **THE key mechanistic fact for our DSP=0 claim, stated explicitly and generally (not just for
  our specific binary case) — Vitis HLS itself decides not to instantiate a multiplier whenever a
  weight's minimal signed-digit representation has ≤ 2 nonzero "digits":** "When the multiplication
  to that weight can be represented as a single addition/subtraction operation ≡ only two bits in
  the minimal signed representation of the weight. As 0-bit weight is trivial, 1-bit weight is
  simply a wire/negation, they also do not use multiplier. **Hence when the number of digits is
  ≤ 2, Vitis HLS 2025.1 (likely also older versions) do not use multipliers, and reuse_factor will
  not be useful for those operations.**" [Sun tutorial, slide 38 — this directly explains, at the
  Vitis-HLS-internals level, WHY our binary `{-1,+1}` FFN synthesizes to DSP=0: a `{-1,+1}` weight
  is *by definition* a 1-signed-digit value, i.e. exactly the "wire/negation, no multiplier" case]
- **Quantitatively, this is NOT a rare edge case** — the tutorial gives a table (assuming
  uniformly-distributed signed weights) of what fraction of weights at a given total bitwidth $W$
  have ≤2 signed digits: **100% at $W \le 4$ bits, 87.5% at $W=5$, 68.8% at $W=6$, 50.0% at
  $W=7$, dropping to 34.4%/22.7%/14.5%/9.0%/5.5% at $W=8/9/10/11/12$** — and notes realistic
  (Gaussian-ish, not uniform) weight distributions push these ratios even higher. **Takeaway
  stated directly: "Most of the low-latency NN for L1T are LUT heavy due to the low number of
  digits in the weights"** — i.e. DSP-avoidance via the digit-count effect is the *general*
  mechanism across the whole low-bitwidth L1-trigger NN literature, not something special to
  hard binary weights specifically — **our `{-1,+1}` case is simply the W=1 extreme end of a
  continuum that already mostly avoids DSPs by W≈6-7 bits.** [Sun tutorial, slide 39, 40] This
  refines (does not contradict) our existing thesis statement in `project-context.md` — worth
  citing this Vitis-internals mechanism explicitly next time we write up the DSP=0 result, rather
  than treating it as unique to us.
- **`reuse_factor` behavior is more subtle than its name implies:** it sets the II (initiation
  interval) target for a CMVM operation, but does NOT guarantee that operation reuse ⟹ hardware
  multiplier reuse — "There might be no multiplier used the first place." Concretely: with a 2×2
  kernel with all weights having >2 signed digits, `reuse_factor=2` failed to be honored at all
  ("the compiler refused to comply with the `II=2` constraint") in one of the tutorial's toy
  examples; with weights that DO have ≤2 digits, some multiplies become subtractors even at
  `reuse_factor=1`. **Rule of thumb given:** increasing `reuse_factor` roughly divides DSP usage
  but leaves LUT usage unchanged-to-slightly-increased (muxing overhead); **`II=1` (reuse_factor=1)
  almost always gives the lowest LUT consumption.** [Sun tutorial, slides 33–40]
- **Recommended strategy when using HGQ(2):** prefer `distributed_arithmetic` over `Latency` "in
  general for both LUT and latency"; **`Resource` strategy is "usually not a good idea"** with
  HGQ; prefer `resource_unrolled` over `Resource` when pruning is present. Prefer tuning a
  `parallelization_factor` (kernel-level reuse) before touching `reuse_factor`/`strategy`, since
  "kernel-level muxing/demuxing is usually cheaper than multiplier-level muxing/demuxing," and
  multi-layer dataflow pipelining is "a bit problematic on the current hls4ml+vitis hls backend"
  for overall latency (though resource/II are usually fine). [Sun tutorial, slide 44]
- **Overall workflow (explicit diagram):** Define/train model with HGQ2 → convert with hls4ml
  (Keras-3 frontend) → HLS synthesis (Vitis/Vivado, Quartus/OneAPI, or Catapult backend) → RTL
  synthesis/place-and-route → FPGA. `da4ml` sits as an alternate/parallel path directly off the
  HGQ2 model (producing an adder-graph representation), either feeding its own RTL output or
  feeding into hls4ml's `distributed_arithmetic` strategy. [Sun tutorial, slide 29]
- **Bit-exactness is fp32-emulated, not literally proven,** but "with fp32 emulation, the
  mismatches should be negligible and occur very rarely — i.e., few in millions with less than
  0.1% relative change in values," provided inputs and all non-trivial-op inputs (table lookups,
  multiplies) are quantized. [Sun tutorial, slide 30]

---

### 7. da4ml — what it is, relation to HGQ2, when to use it instead of hls4ml directly

- **Paper:** Sun et al., "da4ml: Distributed Arithmetic for Real-time Neural Networks on FPGAs,"
  **arXiv:2507.04535** (2025), also published as **ACM TRETS**, DOI 10.1145/3777387. (source:
  https://arxiv.org/abs/2507.04535, https://dl.acm.org/doi/10.1145/3777387, accessed 2026-07-04)
- **What it is:** a standalone library implementing **constant-matrix-vector-multiplication
  (CMVM) via Distributed Arithmetic** — i.e., it replaces a matrix-vector multiply (where the
  matrix is known at synthesis time — exactly the "weight-like" case in HGQ2's own vocabulary)
  with an **optimized adder-graph** (shift-and-add network with common-sub-expression
  elimination), rather than instantiating multipliers. "Distributed arithmetic is a technique for
  implementing fixed-point multipliers in hardware via shift-and-add operations... an adder graph
  structure, optimized by heuristics to minimize the number of adders weighted by their bit-width,
  including common sub-expression elimination." (source: WebSearch summary of arXiv:2507.04535,
  accessed 2026-07-04)
- **Relation to HGQ2:** it is a separate PyPI package (`da4ml>=0.6`, an optional HGQ2 test/runtime
  dependency), **not part of HGQ2 itself**, but "seamlessly" integrated: HGQ2 models can be fed
  directly to `da4ml` as an alternative to hls4ml, OR `da4ml` can be invoked **as hls4ml's
  `distributed_arithmetic` strategy** for the Vitis/OneAPI backends (i.e. da4ml is the engine
  behind that one hls4ml strategy option). [Sun tutorial, slides 4, 6, 21–23, 28, 29]
- **When people use it instead of / alongside hls4ml directly:** (a) when `reuse_factor=1` is
  required/acceptable — DA strategy only supports `reuse_factor=1`; (b) whenever DSP usage must be
  literally zero and pure-hls4ml `Latency`/`Resource` strategies aren't reliably avoiding DSPs
  (DA "usually" gives similar-or-lower LUT than Latency AND zero DSP by construction, vs.
  Latency's DSP usage depending on the accidental digit-count of the trained weights); (c) it can
  also be used **fully standalone** to emit synthesizable RTL directly, bypassing HLS C-synthesis
  entirely — useful when HLS synthesis time/RAM cost is a bottleneck (the tutorial separately
  warns "HLS synthesis is expensive... you may need days and hundreds of GiB of RAM"). [Sun
  tutorial, slides 21–23, 28, 29, 45]
- **Quantitative claims (already partly logged; reconfirmed):** da4ml/DA strategy "can reduce
  on-chip resources by up to a third for realistic, highly quantized neural networks," "usually
  reduce[s] up to 30% of the LUTs and all DSPs used compared to traditional latency strategy CMVM
  kernels." (source: WebSearch summary of arXiv:2507.04535, accessed 2026-07-04 — not yet
  independently PDF-verified, treat the specific "30%"/"a third" figures as provisional pending a
  direct PDF read)
- **Relevance to BNJetTag specifically:** since our binary `{-1,+1}` weights are already the
  degenerate 1-signed-digit case that Vitis HLS itself turns into wire/negation with **zero
  multiplier instantiation regardless of strategy** (§6 above), da4ml/DA's main value-add for us
  would likely be on the **A8/A6/A4 activation-quantized, non-binary layers** (attention
  projections, LayerNorm scale/shift, softmax normalization) where weights are NOT forced to
  `{-1,+1}` and Latency-strategy DSP usage is not guaranteed to be zero — **worth a follow-up
  resource comparison (`ml-engineer`/`results-analyst`) of Latency vs. distributed_arithmetic
  strategy specifically on those non-binary sublayers**, not on the already-binary FFN core.

---

### 8. Published end-to-end HGQ2 + attention/transformer + hls4ml examples found

- **arXiv:2510.24784, "Sub-microsecond Transformers for Jet Tagging on FPGAs"** (already logged
  2026-07-02 with full Table 1) is now confirmed to be **the paper that added native MHA/Linformer
  support to hls4ml** via HGQ2's `QMultiHeadAttention`/`QLinformerAttention` — this is the
  strongest published "HGQ2 + attention + hls4ml, end-to-end, synthesized" example we found,
  reporting real Vitis-HLS LUT/latency/DSP numbers (DSP=0 across every HGQ-trained
  configuration) on the same public HLS4ML LHC jet dataset we use. (source:
  https://arxiv.org/abs/2510.24784, abstract text fetched 2026-07-04)
- **Chang Sun's own tutorial (this session's primary source)** is itself an unpublished-but-
  citable (indico-hosted) worked walkthrough of the whole HGQ2→hls4ml→da4ml pipeline with toy
  CMVM synthesis examples (RTL screenshots of actual Vitis HLS output for 2×2 and cascaded dense
  layers) — useful as a teaching reference but not a peer-reviewed jet-tagging result in itself.
  URL: https://indico.cern.ch/event/1593461/contributions/6715101/attachments/3170139/5635793/material.pdf
  (2025-11-10, Caltech/PMA, NGT Tutorial).
- **Adjacent but NOT HGQ2/hls4ml:** found two newer (2026) FPGA-transformer-jet-tagging papers
  while searching, neither of which uses the HGQ2/hls4ml stack, so they don't answer this
  question but are worth flagging as parallel efforts: **JetFormer, arXiv:2601.17215** ("A
  Scalable and Efficient Transformer for Jet Tagging from Offline Analysis to FPGA Triggers,"
  submitted 2026-01-23) — uses **"custom extensions to the Allo high-level synthesis framework,"
  not hls4ml**, reports matching ParT accuracy within 0.7% on JetClass with 37.4% fewer FLOPs, and
  "JetFormer-tiny" variants synthesized for sub-µs FPGA triggers (source:
  https://arxiv.org/abs/2601.17215, accessed 2026-07-04 — not independently PDF-read, title/
  abstract only); and **"Reconfigurable Computing Challenge: Transformer for Jet Tagging on Versal
  AI Engines," arXiv:2606.17500** — targets Versal ACAP AI Engines, not classic HLS4ML-Vitis FPGA
  fabric (source: https://arxiv.org/pdf/2606.17500, accessed 2026-07-04, title only, not
  PDF-read). Also noted in passing: **HGQ-LUT, arXiv:2604.22293** — "Fast LUT-Aware Training and
  Efficient Architectures for DNN Inference," stated to be integrated into
  `github.com/calad0i/HGQ2` itself as a new "LAT" (lookup-table-aware training) mode claiming
  "state-of-the-art hardware efficiency while accelerating training by over 100 times on modern
  GPUs" (source: WebSearch summary of arXiv:2604.22293, accessed 2026-07-04 — not PDF-read, flag
  for a dedicated follow-up read since it's directly inside our target library).

---

### Key open items / flags for follow-up (handing to lead-pm / ml-engineer)

1. **Binary-pinning mechanism is genuinely ambiguous from docs alone** (§4) — needs an empirical
   check by `ml-engineer`: does a `kbi` quantizer frozen to its minimum width (`trainable=False`,
   `k0`/`i0`/`f0` or `b0` set) actually produce the value set `{-1,+1}` (no zero), or does it
   produce `{-1,0}`/`{-1,0,+1}`? If not exactly `{-1,+1}`, the dedicated (but "preliminary")
   `quantizers.binary_quantize()` may be required instead, with its own hls4ml/da4ml conversion
   compatibility unverified.
2. **da4ml's "30%"/"up to a third" resource-reduction figures are WebSearch-summarized, not
   PDF-verified** — recommend a direct read of arXiv:2507.04535 before quoting these numbers in
   `RESEARCH.md`.
3. **HGQ-LUT (arXiv:2604.22293)** claims to already be merged into `calad0i/HGQ2` and offers
   >100× training speedup via a LUT-aware training mode — worth a dedicated read given it's
   inside our exact target library and directly relevant to training-time cost on NRP.
4. **EBOPs formula changed between HGQ v1 (arXiv:2405.00645, no explicit accumulator term — our
   2026-06-29 log) and HGQ2 (explicit `+ Σ max(bw_i,bw_j)` accumulator term per the Sun tutorial)**
   — if we quote an EBOPs number after the HGQ2 rebuild, cite the HGQ2 tutorial formula, not the
   HGQ-v1 paper formula, since they now differ.

## 2026-07-02 — Literature scan: BitLinear/binary-transformer hardware + the EBOPs comparison table for the HLS4ML LHC jet dataset

**Question:** (1) What's new/adjacent to BitNet in binary-transformer hardware since our library
was built (2026-06-28)? (2) What do published trigger-oriented jet taggers report on the same
public HLS4ML LHC jet dataset? (3) Build an EBOPs/BOPs-vs-accuracy comparison table anchored on
HGQ (arXiv:2405.00645), the direct axis for comparing our ~6.4M-param W1A8 binary transformer.

**Method note:** Numbers below marked "[PDF-read]" were extracted by directly reading the paper
PDF with the Read tool (highest confidence — verbatim). Numbers marked "[WebFetch]" came from
arXiv-HTML summarized by WebFetch's secondary model; where I could cross-check by fetching twice
independently, the accuracy/%/latency figures were stable across fetches, but one internal
inconsistency surfaced (see HGQ flag below) — treat any single absolute count from WebFetch with
caution until confirmed against the PDF directly.

---

### Goal 1 — BitNet-adjacent / binary-transformer hardware, newer than our library

- **ViT-1.58b: Mobile Vision Transformers in the 1-bit Era**, arXiv:2406.18051 (June 2024).
  Applies BitNet b1.58-style **ternary {−1,0,+1}** `BitLinear` layers to a ViT, activations kept
  at 8-bit. Reports CIFAR-10 and ImageNet-1k results "comparable accuracy to full-precision ViT"
  with large memory/compute reduction; ViT-1.58b-L's training loss tracks full-precision ViT-L
  closely. (source: https://arxiv.org/abs/2406.18051, https://arxiv.org/html/2406.18051v1,
  accessed 2026-07-02)
  — Relevance: closest published analog to our W1/W1.58-weight, A8-activation recipe, but for
  vision, not particle physics; no FPGA synthesis reported (this is an accuracy-only paper).
- **BinaryAttention: One-Bit QK-Attention for Vision and Diffusion Transformers**,
  arXiv:2603.09582 (2026). Binarizes the **Q/K attention pathway itself** to 1 bit, not just the
  linear-layer weights — directly relevant to our own open question of whether/how far attention
  score computation can be binarized (we currently exclude the attention core from HLS synthesis
  because EinsumDense isn't hls4ml-supported; see RESEARCH.md §6 caveat). (source:
  https://arxiv.org/pdf/2603.09582, accessed 2026-07-02; found via WebSearch, not yet PDF-read —
  flag for a follow-up deep read if we revisit attention-core binarization)
- **BiViT: Exploring Binary Vision Transformers** (OpenReview / arXiv, ~2023-2024) and the
  related **Bi-ViT: Pushing the Limit of Vision Transformer Quantization**: both binarize weights
  *and* activations to 1 bit (full W1A1, more aggressive than our W1A8/A6/A4 axis). BiViT reports
  75.6% ImageNet top-1 with a binarized Swin-S backbone via cross-layer binarization that
  decouples attention vs. MLP quantization. (source: OpenReview id lXBzOtKn20t; ResearchGate
  summaries of both papers, accessed 2026-07-02 — **not independently PDF-verified, treat top-1
  number as provisional**)
  — Relevance: shows W1A1 (full binary) ViTs are being pushed in vision; useful upper-bound
  context for how aggressive activation quantization can go before it's "too much," complementing
  our own A8→A6→A4 sweep (we have not tried A1/A2).
- **BiBERT: Accurate Fully Binarized BERT**, arXiv:2203.06390 (ICLR 2022). Fully binarizes BERT
  (weights + activations); reports it "exceeds 1-1-1-bit BinaryBERT by 20.4% accuracy on average"
  on GLUE and achieves "56.3× and 31.2× savings on FLOPs and model size" vs. full precision.
  (source: https://arxiv.org/abs/2203.06390, accessed 2026-07-02)
  — Relevance: the NLP-domain precedent for "binarizing attention is the hard part" — BiBERT's
  main contribution (Bi-Attention structure + Direction-Matching Distillation) exists specifically
  because naive full binarization of attention degrades badly, echoing why we've kept attention
  scores in higher precision and treat attention-core binarization as future work.
- **Co-Designing Binarized Transformer and Hardware Accelerator for Efficient End-to-End Edge
  Deployment**, arXiv:2407.12070 (2024). A binarized-transformer + custom accelerator co-design
  (not FPGA/hls4ml specifically — appears to target a custom ASIC/accelerator flow), with an
  "energy efficient matrix multiplication decomposition" to cut binarized-matmul cost further.
  (source: https://arxiv.org/pdf/2407.12070, accessed 2026-07-02 — found via WebSearch only, not
  PDF-read; flag for deeper read if we want cross-accelerator (non-hls4ml) comparisons)
  — Relevance: another binary-transformer-on-hardware paper confirming the "binary matmul → no
  multiplier" hardware argument outside the HEP/hls4ml ecosystem.
- **FBPT: A Fully Binary Point Transformer**, arXiv:2403.09998. Binarizes a point-cloud
  transformer (point clouds ≈ our permutation-invariant particle-constituent sets). Not HEP, but
  the closest architectural analog we found to "binary transformer over an unordered set of
  points/particles." (source: https://arxiv.org/html/2403.09998v1, accessed 2026-07-02 — title/
  abstract only, not deep-read)
- **BEExformer: A Fast Inferencing Binarized Transformer with Early Exits**, arXiv:2412.05225 —
  binarized transformer + early-exit for latency reduction; adjacent but early-exit doesn't map
  cleanly onto a fixed-latency L1-trigger pipeline. (source: https://arxiv.org/pdf/2412.05225,
  accessed 2026-07-02, title/abstract only)
- None of the above (except the two HEP-adjacent FPGA papers under Goal 2/3) report BOPs or
  EBOPs; the EBOPs framing appears to be specific to the hls4ml/HGQ lineage, not yet adopted by
  the vision/NLP binary-transformer literature.

---

### Goal 2 / 3 — Trigger-oriented jet taggers on the HLS4ML LHC jet dataset, and the EBOPs table

**Dataset identity check:** the "HLS4ML LHC Jet dataset (150 particles)" is the same public
benchmark across all papers below: Pierini, Duarte, Tran, Freytsis, Zenodo record (Jan 2020),
underpinned by arXiv:1804.06913 (original hls4ml paper) and arXiv:1908.05318 (JEDI-net); 5
classes (gluon g, light quark q, W, Z, top t), balanced. **Two different input conventions
appear in the literature** — this matters for comparing to us:
  - **16 high-level physics features per jet** (not per-particle) — used by the original hls4ml
    MLP, QKeras/AutoQKeras, and HGQ. This is a *jet-level* feature vector, fully different in
    kind from constituent/particle-level input.
  - **Per-particle constituent input** (various numbers of particles, various numbers of
    features per particle) — used by JEDI-net (interaction network over constituents),
    the Sub-microsecond Transformers paper (10871/8/16/32/64 particles × **3 features**: pT, η,
    φ), and us (**10 particles × 16 features = 160 inputs**). **Our 16-features-per-particle
    input is richer than the 3-features-per-particle used in the closest transformer comparison
    paper** — direct accuracy comparison to that paper is confounded by this input difference,
    not just by our binary weights.

**A. HGQ (arXiv:2405.00645) — the EBOPs anchor paper, jet-tagging (16 jet-level features) Table I**
[WebFetch, cross-checked across 2 independent fetches of the arXiv-HTML — accuracy/%/latency
figures agreed both times]:

- Baseline architecture: "a 4-layer fully connected neural network" (the classic hls4ml/
  AutoQKeras jet-tagger, 16 jet-level features in, three hidden layers, 5-way softmax out).
  Metric reported is **classification accuracy (%), not AUC** — different metric than our
  macro-OvR AUC reporting; do not directly equate the numbers below to our AUC without
  re-deriving accuracy from our confusion matrix, or vice versa.
- Target device (per WebSearch of the paper's stated FPGA): **Xilinx XCVU9P**, clock period
  implied 5 ns (200 MHz) — LUT-percentage figures below are consistent with the VU9P's ~1.18M
  LUT total (48,321 LUT / 4.09% ≈ 1.18M ✓); **the DSP-percentage figures did NOT independently
  reconcile against VU9P's known ~6,840 DSP48E2 count (1,826 DSP quoted as "56.0%" implies a
  device total of only ~3,260 DSPs)** — flagging this as an unresolved extraction inconsistency,
  possibly a WebFetch transcription artifact or a partial-device/pragma-limited DSP budget in the
  paper. **Use the % figures below with normal confidence; do not trust the absolute DSP/LUT
  counts in parentheses without confirming against the primary PDF.**
  (source: https://arxiv.org/abs/2405.00645, https://ar5iv.labs.arxiv.org/html/2405.00645,
  accessed 2026-07-02)

| Model | Accuracy | Latency | DSP % | LUT % | Notes |
| --- | --- | --- | --- | --- | --- |
| BF (baseline full, ~ FP-ish) | 74.4% | 9 cc (45 ns) | 56.0% | 4.09% | unquantized-ish baseline |
| BP (baseline pruned) | 74.8% | 14 cc (70 ns) | 7.7% | 1.49% | pruned baseline |
| BH | 73.2% | 14 cc (70 ns) | 1.3% | 1.34% | another baseline variant |
| Q6 (uniform 6-bit QKeras) | 74.8% | 11 cc (55 ns) | 1.8% | 3.36% | |
| QE (AutoQKeras, energy-optimized) | 72.3% | 11 cc (55 ns) | 1.0% | 0.77% | AutoQKeras [Coelho 2021, 2006.10159] |
| QB (AutoQKeras, bits-optimized) | 71.9% | 14 cc (70 ns) | 1.0% | 0.95% | |
| **HGQ-1** | **76.4%** | 6 cc (30 ns) | 0.50% | 0.53% | HGQ Pareto point, highest-accuracy |
| HGQ-2 | 75.9% | 4 cc (20 ns) | 0.09% | 0.27% | |
| HGQ-3 | 75.0% | 4 cc (20 ns) | 0.07% | 0.13% | |
| HGQ-4 | 73.9% | 3 cc (15 ns) | 0.00% | 0.05% | |
| HGQ-5 | 72.5% | 2 cc (10 ns) | 0.00% | 0.04% | |
| **HGQ-6** | **71.0%** | 2 cc (10 ns) | 0.00% | 0.02% | HGQ Pareto point, lowest-resource |

- **Important nuance found, correcting the task brief's premise:** HGQ's Table I does **not**
  tabulate literal EBOPs counts per model — the reported Pareto axis is measured resource
  utilization (DSP%, LUT%) on the VU9P after synthesis, and the paper's Figure III plots
  "accuracy versus resource consumption" using the empirical **EBOPs ≈ LUT + 55×DSP** combination
  (already logged 2026-06-29) rather than raw EBOPs on an axis. EBOPs is used as the **training-
  time regularization target** (the β-controlled loss term), not a reported per-model output
  metric in the results table. So: to compare our binary transformer's EBOPs to HGQ's Pareto
  front, the correct move is to convert our synthesized LUT/DSP into their same
  LUT+55×DSP resource proxy, not to look for a literal "EBOPs" column in their table.
  (source: same as above)
- HGQ-1 through HGQ-6 come from **a single training run with a ramping β** (the EBOPs
  regularization strength), i.e. one HGQ training run traces the whole Pareto front — not 6
  separately-trained models. (source: https://ar5iv.labs.arxiv.org/html/2405.00645, WebFetch,
  accessed 2026-07-02)
- HGQ headline claim (abstract, PDF-verbatim): "orders of magnitude reduction in resource
  consumption and latency while maintaining accuracy" — consistent with HGQ-1 (76.4%, the
  *highest* accuracy in the whole table) using **0.50% DSP / 0.53% LUT**, i.e. ~2-100× fewer
  resources than every baseline (BF 56.0%/4.09%, Q6 1.8%/3.36%) while scoring *higher* accuracy
  than all of them. (source: https://arxiv.org/abs/2405.00645, accessed 2026-07-02)

**B. Sub-microsecond Transformers for Jet Tagging on FPGAs (arXiv:2510.24784, NeurIPS 2025 ML4PS
workshop) — [PDF-read directly, full text, Table 1 verbatim]:**

- Authors: Laatu, Sun, Cox, Gandrakota, Maier, Ngadiuba, Que, Luk, Spiropulu, Tapper.
  FERMILAB-PUB-25-0779-CMS-LDRD. (source: PDF at
  `papers/jet-tagging-transformers/2510.24784_submicrosecond_transformers_jet_tagging_fpga.pdf`,
  also https://arxiv.org/abs/2510.24784)
- **Dataset:** the same public HLS4ML LHC jet dataset (their refs [33,34] = Pierini/Duarte/Tran/
  Freytsis Zenodo record + Coleman et al. calorimetry paper), **620,000 train / 260,000 test
  jets**, 5 classes (g/q/W/Z/t), balanced. **Input: per-particle, 3 features (pT, η, φ) per
  particle**, sequence length swept over **(8, 16, 32, 64) particles**, sorted by pT, no
  positional encoding — a Set-Transformer-style encoder-only model with **single-head** vanilla
  MHA, or Linformer linear attention (K/V projected to a lower dim than n).
- **Precision:** HGQ-trained (per-parameter bitwidth, gradient-based), attention layers
  constrained to ≥1 bit, other layers unconstrained bitwidth (can go to 0, i.e. pruned). All
  models trained to a **fixed target EBOPs of 350,000** ("roughly one Super Logic Region of the
  target XCU250 chip"), enforced via a PID controller on β.
- **Target device:** Xilinx **XCU250** (note: NOT VU13P — different device than our VU13P
  target). Tools: Vitis HLS + Vivado.
- **Table 1 results (verbatim from the PDF):**

| Model | Particles | Accuracy | Latency | LUT (k) | II (clk) | DSP |
| --- | --- | --- | --- | --- | --- | --- |
| Multi-Head Attention | 8 | 66.3% | 104 ns | 246k | 1 | 0 |
| Multi-Head Attention | 16 | 72.3% | 98 ns | 279k | 1 | 0 |
| Multi-Head Attention | 32 | 77.0% | 83 ns | 180k | 1 | 0 |
| Multi-Head Attention | 64 | 77.9% | 44 ns | 47k | 1 | 0 |
| Linformer | 8 | 66.3% | 110 ns | 230k | 1 | 0 |
| Linformer | 16 | 72.8% | 103 ns | 246k | 1 | 0 |
| Linformer | 32 | 78.4% | 140 ns | 267k | 1 | 0 |
| **Linformer** | **64** | **79.8%** | **78 ns** | **202k** | 1 | **0** |
| Deep Sets (HGQ) | 8 | 64.7% | 49 ns | 177k | 1 | 0 |
| Deep Sets (HGQ) | 16 | 70.1% | 53 ns | 205k | 1 | 0 |
| Deep Sets (HGQ) | 32 | 77.4% | 53 ns | 256k | 1 | 0 |
| Deep Sets (HGQ) | 64 | 79.4% | 44 ns | 191k | 1 | 0 |
| MLP Mixer [Sun et al. 2503.03103] | 16 | 71.7% | 68 ns | 75k | 1 | 0 |
| MLP Mixer | 32 | 78.0% | 62 ns | 63k | 1 | 0 |
| MLP Mixer | 64 | 79.7% | 72 ns | 159k | 1 | 0 |
| Deep Sets (QKeras) [Odagiu et al. 2402.01876] | 8 | 64.0% | 95 ns | 386k | 3 | 626 |
| Deep Sets (QKeras) [2402.01876] | 16 | 69.4% | 115 ns | 747k | 3 | 555 |
| Deep Sets (QKeras) [2402.01876] | 32 | 75.9% | 130 ns | 903k | 2 | 434 |
| Deep Sets M (QKeras) [Weitz et al. NAC/MetaML, 2501.05515] | 8 | 65.1% | 110 ns | 130k | 3 | 548 |
| Deep Sets L (QKeras) [2501.05515] | 8 | 66.6% | 135 ns | 337k | 3 | 2,458 |

- **Per-class AUC (ROC curves, Fig. 2, PDF-verbatim legend values)** for Linformer and MHA —
  note this is **per-class one-vs-rest AUC, not macro-averaged in the paper**; the macro-average
  in the row below is **our own arithmetic mean of their 5 per-class numbers**, not a number the
  paper states directly:

  | Particles | Model | g | q | W | Z | t | macro-avg (ours, computed) |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | 8 | Linformer | 0.853 | 0.886 | 0.917 | 0.900 | 0.929 | 0.897 |
  | 16 | Linformer | 0.896 | 0.904 | 0.947 | 0.935 | 0.946 | 0.926 |
  | 32 | Linformer | 0.930 | 0.919 | 0.968 | 0.963 | 0.962 | 0.948 |
  | 64 | Linformer | 0.941 | 0.921 | 0.972 | 0.968 | 0.964 | 0.953 |
  | 8 | MHA | 0.854 | 0.886 | 0.916 | 0.900 | 0.929 | 0.897 |
  | 64 | MHA | 0.930 | 0.911 | 0.963 | 0.956 | 0.954 | 0.943 |

- **Key qualitative finding stated by the authors:** every HGQ-trained model in their Table 1
  (MHA, Linformer, Deep Sets-HGQ, MLP-Mixer) achieves **DSP = 0** at every input length, while
  every uniformly-quantized QKeras baseline uses hundreds to **thousands** of DSPs (up to 2,458
  for "Deep Sets L (QKeras)" at just 8 particles) — this is the direct precedent for our own
  "binary/HGQ-style quantization → 0 DSP" structural claim, but achieved via *per-parameter*
  learned bit-widths (some going to 0/pruned) rather than a hard binary `{−1,+1}` constraint like
  ours. (source: Table 1, arXiv:2510.24784)
- **No BOPs or EBOPs value is reported per-model in Table 1** — EBOPs=350,000 is a fixed
  *training target* applied identically to all their own HGQ-trained models (MHA, Linformer,
  Deep Sets-HGQ), not a per-model output metric, and it is **not applied or reported** for the
  QKeras/MLP-Mixer baseline rows (those are uniformly-quantized, not HGQ-trained). (source: §3,
  arXiv:2510.24784)

**C. JEDI-net (arXiv:1908.05318) per-class AUC and param count:**
- JEDI-net (interaction network over constituents) reported AUC 0.9529 (g) / 0.9301 (q) / 0.9739
  (W) / 0.9679 (Z) / 0.9683 (t), with **~34k trainable params** — our own arithmetic macro-average
  of these five is **0.9586**. **Caveat: these numbers came from a secondary compilation table (a
  later paper's comparison table citing JEDI-net), not a direct read of JEDI-net's own tables —
  flag for verification against the original 1908.05318 PDF before quoting as a JEDI-net
  citation.** (source: WebSearch summary, accessed 2026-07-02 — not yet independently confirmed)
  — Relevance: **34k params is ~190× smaller than our 6.375M-param model**, illustrating how much
  smaller the published hls4ml-dataset benchmarks typically are (see takeaways below).

**D. LogicNets (arXiv:2004.03021)** — confirmed the paper exists and includes a "Jet Substructure
Classification" (i.e. this same dataset) case study mapping quantized neurons directly to LUTs as
truth tables, but **exact accuracy/LUT/latency numbers were not confirmed** (search results did
not surface the table; not independently PDF-read this session). **Not quoted — flag for a
follow-up direct PDF read if LogicNets numbers are needed for a table.** (source:
https://arxiv.org/abs/2004.03021, accessed 2026-07-02)

**E. JEDI-linear (arXiv:2508.15468)** — newer (2025) linear-complexity interaction-network
successor to JEDI-net, explicitly HGQ/quantization-aware and DSP-free. Abstract-level numbers
only (not PDF-read): "3.7 to 11.5× lower latency" and "up to 6.2× lower LUT usage" than prior
GNN designs, "less than 60 ns latency," "eliminating the need for DSP blocks entirely," higher
accuracy than baselines — **no absolute accuracy/AUC or param count confirmed this session.**
(source: https://arxiv.org/abs/2508.15468, accessed 2026-07-02)

**F. AutoQKeras / QKeras (arXiv:2006.10159)** — already in our library
(`papers/hls4ml-fpga-triggers/2006.10159_autoqkeras_heterogeneous_quantization.md`); its two
named jet-tagging configurations (Energy-optimized "QE" and Bits-optimized "QB") are the same
models HGQ's Table I compares against (72.3% / 71.9% accuracy — see table A above). No new
numbers found beyond what's already logged via the HGQ table.

**G. Not investigated this session (would need dedicated follow-up):** MetaML the *conference*
paper (Que et al., FPL 2023 — distinct from the MLST 2025 "Neural Architecture Codesign" journal
paper, arXiv:2501.05515, which we do have numbers for transitively via Table 1 above); Deiana et
al. "Applications and Techniques for Fast Machine Learning in Science" (arXiv:2110.13041) and
"FastML Science Benchmarks" (arXiv:2207.07958) — both confirmed to exist and be relevant
community reviews, but not mined for jet-tagging-specific numbers this session; SymbolNet /
symbolic-regression taggers — not found under that name in this search pass, flag for a
targeted follow-up search ("SymbolNet arXiv jet tagging FPGA").

---

### Takeaways — where would our ~6.4M-param W1A8 binary transformer sit?

1. **Every published model in the tables above is dramatically smaller than ours.** JEDI-net:
   ~34k params. The HGQ jet-tagging baseline (4-layer FCNN, 16 features): almost certainly O(1k–
   10k) params (3 hidden layers of 64/32/32 on 16 inputs ≈ a few thousand weights; not explicitly
   stated this session, flag for exact count). The sub-microsecond transformers paper's own
   models: unspecified in the paper, but their LUT counts (47k–903k depending on config) and the
   fact that a **fixed target EBOPs of 350,000** binds all of their own HGQ-trained models
   suggests parameter counts and MAC counts far below a 6.375M-param, 8-layer/8-head/FFN-1024
   transformer. Our model is at least ~2 orders of magnitude larger than any published model on
   this exact dataset that we found numbers for.
2. **On the EBOPs/resource axis, "far smaller and far cheaper" published models still land at
   DSP=0 and sub-1% LUT** (HGQ-6: 71.0% accuracy at 0% DSP / 0.02% LUT of a VU9P; the transformer
   paper's Linformer-64: 79.8% accuracy at 0 DSP / 202k LUT of an XCU250). Our folded RF=256
   binary FFN block alone is ~25% LUT of a VU13P (RESEARCH.md §6) — i.e., **our single FFN block
   already uses more LUT than these entire published end-to-end jet taggers**, which is expected
   given our model is ~100-1000× larger, but it means a literal "our EBOPs vs. their EBOPs"
   overlay would show us far off to the right (high-resource) end of every published Pareto
   front. The comparison that will actually be informative is **accuracy delta per unit EBOPs
   added going from their scale to ours** — not "are we Pareto-optimal against these."
3. **The closest architectural comparison (sub-microsecond transformers, 2510.24784) uses 3
   features/particle vs. our 16 features/particle** and jet-level-feature papers (HGQ, AutoQKeras)
   use 16 *jet-level* (not per-particle) features — **no published number in this table is an
   apples-to-apples input match to our 10×16 constituent representation.** Any accuracy
   comparison we make once round-5 numbers land must state this input-representation difference
   explicitly, not just the precision difference.
4. **Metric mismatch is real and multi-way:** HGQ/AutoQKeras report **accuracy (%)**; the
   sub-microsecond transformers paper reports **accuracy (%) + per-class AUC** (not macro-AUC);
   JEDI-net reports **per-class AUC**; we report **macro one-vs-rest AUC**. Every cross-paper
   comparison number in this log that isn't explicitly our own arithmetic macro-average is a
   *different metric* than our headline number — do not drop any of these into RESEARCH.md as a
   same-axis comparison without recomputing on a common metric.
5. **The "binary weights → 0 DSP" structural claim has direct, multiple, independent published
   precedent** on this exact dataset and adjacent ones: every HGQ-style (learned per-parameter
   bit-width, including hard 0/1-bit) model in both table A and table B hits DSP=0, while every
   uniformly-quantized QKeras baseline uses hundreds-to-thousands of DSPs. Our contribution is
   not "DSP=0 is achievable" (well-established) but **applying a hard {−1,+1} constraint
   specifically (not learned/soft bit-width) to a much larger transformer** and reporting the
   resulting AUC/EBOPs/latency trade at that scale — worth stating precisely this way rather than
   claiming DSP=0 itself is the novel finding.

---

### New/adjacent BitNet-on-hardware papers worth a deeper read (one-line relevance each)

- **BinaryAttention** (arXiv:2603.09582) — binarizes the Q/K attention pathway itself; directly
  relevant to our open "attention-core binarization" question (currently excluded from our HLS
  synthesis, RESEARCH.md §6/§7 Q4).
- **ViT-1.58b** (arXiv:2406.18051) — closest published BitNet-b1.58-style {−1,0,+1}-weight, A8
  transformer outside HEP; accuracy-only (no FPGA numbers), but a useful "does BitLinear + A8
  hold up" reference point outside our own domain.
- **BiBERT** (arXiv:2203.06390) — the canonical "binarizing attention is the hard part" NLP paper;
  its Bi-Attention + distillation fixes are a possible source of ideas if we ever try to push
  attention itself into `{−1,+1}`.
- **Co-Designing Binarized Transformer and Hardware Accelerator** (arXiv:2407.12070) — binary
  transformer + custom accelerator co-design outside the hls4ml ecosystem; useful for
  cross-checking whether the "binary matmul → no DSP-equivalent multiplier" argument holds on
  non-FPGA/non-hls4ml accelerators too.
- **JEDI-linear** (arXiv:2508.15468) — newest (2025) DSP-free, HGQ-trained interaction network on
  this exact dataset; worth a full PDF read since it's the most current directly-competing
  trigger-tagger design philosophy (learned bit-width vs. our hard binary).

## 2026-06-29 — BOPs and EBOPs: precise definitions, formulas, and resource mapping

**Question:** What are the exact, citable formulas for BOPs and EBOPs, how do they treat b_w=1
binary weights, and how does EBOPs map empirically to on-chip LUT/DSP for hls4ml designs?

---

### 1. BOPs (Bit-Operations) — origin and exact formula

**Originating paper:** Baskin et al., "UNIQ: Uniform Noise Injection for the Quantization of
Neural Networks," arXiv:1804.10969 (ACM Trans. Comput. Syst. 37, 2021).
(source: https://arxiv.org/abs/1804.10969, accessed 2026-06-29)

**Key claim in UNIQ:** "We propose a novel complexity metric of number of bit operations
performed (BOPs), and show that this metric has a linear relation with logic utilization and power."
The metric is designed to generalise FLOPs to heterogeneous-precision (mixed-bit) networks.

**BOPs definition — per primitive operation:**
- Multiply of a b_i-bit number by a b_j-bit number: **b_i * b_j BOPs**.
- Add of two numbers of widths b_i and b_j: **max(b_i, b_j) + 1 BOPs**
  (the result requires one extra bit beyond the wider operand).

**Per-layer formula — convolutional layer (from UNIQ §4.2, and reproduced in Ps and Qs Eq. 8):**

$$\text{BOPs}_\text{conv} = m n k^2 \bigl( b_a b_w + b_a + b_w + \log_2(n k^2) \bigr)$$

where m = output channels, n = input channels, k = filter height/width,
b_a = activation bit-width, b_w = weight bit-width.

**Term decomposition:**
- $m n k^2 \cdot b_a b_w$ — the multiply cost: each of the $m n k^2$ MACs costs $b_a b_w$ BOPs.
- $b_a + b_w$ — accumulator input overhead: each addition operand is the output of one multiply,
  which is $b_a + b_w$ bits wide.
- $\log_2(n k^2)$ — adder-tree depth: accumulating $n k^2$ partial products widens the
  accumulator by $\lceil \log_2(n k^2) \rceil$ bits.
Together: accumulator bit-width $b_o = b_a + b_w + \log_2(n k^2)$.

**Per-layer formula — dense (fully-connected) layer:**
Substituting k=1 and treating n as fan-in (inputs) and m as outputs:

$$\text{BOPs}_\text{dense} = m n \bigl( b_a b_w + b_a + b_w + \log_2 n \bigr)$$

**Pruning extension (Ps and Qs, Eq. 8; source: Baskin 2021 cited there):**

$$\text{BOPs}_\text{dense,pruned} = m n \bigl[ (1 - f_p) b_a b_w + b_a + b_w + \log_2 n \bigr]$$

where $f_p$ is the fraction of weights set to zero (pruned). Without pruning ($f_p = 0$) this
reduces to the standard dense formula above.
(source: https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2021.676564/full,
accessed 2026-06-29; Ps and Qs, Banerjee et al., Frontiers in AI 2021, arXiv:2102.11289)

**Dominant term:** The $m n b_a b_w$ multiply-cost term is quadratic in bit-widths; the
$b_a + b_w + \log_2 n$ accumulator/adder terms add only a linear correction. Many papers
(including HAWQ-V3, Yao et al. ICML 2021, arXiv:2011.10680; and Bayesian Bits, van Baalen et al.
NeurIPS 2020, arXiv:2005.07093) drop the accumulator terms and use the simplified form:

$$\text{BOPs}_\text{dense} \approx \text{MACs} \cdot b_w \cdot b_a = m n \cdot b_w \cdot b_a$$

HAWQ-V3 (arXiv:2011.10680, §3.4) states: "The total Bit Operations for calculating a layer:
$G_i^{(b_i)} = b_{w,i} \cdot b_{a,i} \cdot \text{MAC}_i$."
(source: https://ar5iv.labs.arxiv.org/html/2011.10680, accessed 2026-06-29)

Bayesian Bits (arXiv:2005.07093) uses BOPs(l) = MACs(l) * b_w * b_a (no accumulator term).
(source: https://arxiv.org/abs/2005.07093, accessed 2026-06-29)

**FLAG — ambiguity on accumulator terms:** The original UNIQ formula includes the full
accumulator width $b_a + b_w + \log_2(n k^2)$; many subsequent works (HAWQ-V3, Bayesian Bits,
van Baalen) drop it and use only $b_a b_w$ per MAC. The simplified form dominates in practice
because the accumulator term is small for typical b_w, b_a (e.g., 8x8: multiply term = 64
BOPs/MAC vs. accumulator correction of ~8+8+7=23 for n=128). For BNJetTag reporting, if we use
the simplified form BOPs = MACs * b_w * b_a we must state so and cite HAWQ-V3 / Bayesian Bits;
if we use the full UNIQ form we cite UNIQ arXiv:1804.10969.

---

### 2. EBOPs (Effective Bit-Operations) — the HGQ definition

**Source paper:** Chang Sun, Thea K. Aarrestad, Vladimir Loncar, Jennifer Ngadiuba, Wayne Luk,
Maria Spiropulu et al., "HGQ: High Granularity Quantization for Real-time Neural Networks on
FPGAs," arXiv:2405.00645 (2024).
(source: https://arxiv.org/abs/2405.00645, accessed 2026-06-29)
HTML version: https://arxiv.org/html/2405.00645v2 (accessed 2026-06-29)

**Why EBOPs instead of BOPs:**
"A common metric for estimating on-chip resource usage in FPGAs is Bit Operations (BOPs)
[63]. BOPs quantify the resource consumption by counting the number of bits involved in all
operations performed during the network's forward pass. [...] To address this discrepancy and
offer a more precise estimation of on-chip resource usage, we propose a novel metric,
Effective Bit Operations (EBOPs)."

**Equation (5) — exact definition:**

Let $\mathcal{M} = \{(i,j)_n\}$ be the set of ALL multiplication operations in the network,
where operands have bit-widths $b_i$ and $b_j$. Then:

$$\boxed{ \text{EBOPs} = \sum_{(i,j) \in \mathcal{M}} b_i \cdot b_j }$$

This sums $b_i \cdot b_j$ over every single multiply in the network (every scalar multiply
in every matrix-multiply or convolution).

**Two critical modifications vs BOPs:**

(a) **Constant (weight) bit-width redefinition:** "For computing EBOPs, the bitwidth used for
constants is not the declared bitwidth, but the number of bits enclosed by non-zero bits in
binary form." Example verbatim: "a weight '001xx1000' will be counted as 4 bits instead of 8
bits." If multiple weights share a multiplier in a partially-unrolled design, "the bitwidth of
that weight group is defined by the number of bits enclosed by the most and least significant
non-zero bits in that weight group."

(b) **Accumulator / addition cost deliberately excluded:** "EBOPs effectively count only the
BOPs conducted during multiplicative processes in a network with the modified bitwidth
definition." The paper states that "the accumulation of N shifted numbers, each of b bits, [is]
count[ed] as N * b EBOPs" — meaning the adder-tree cost IS included, but through the
multiply-sum, not separately. The accumulation operation's contribution is "implicitly counted"
via the multiply terms themselves. This is the key design choice: addition/accumulation is NOT
independently added as a $b_a + b_w + \log_2 n$ term; instead, the sum over all multiplies
already captures the full cost.

**EBOPs → resource empirical mapping (Section V.1 / Figure II):**

$$\text{EBOPs} \approx \#\text{LUT} + 55 \times \#\text{DSP}$$

This is the empirical linear fit across multiple benchmark tasks synthesized with
io_type=io_parallel (unrolled / parallel IO) in hls4ml. The coefficient 55 means one DSP "costs"
approximately 55 LUTs in terms of EBOPs. Models synthesized with stream IO have higher actual
resource consumption because of additional buffering.

**Why EBOPs correlates better than BOPs with real resources:**
BOPs counts all bit-level operations uniformly; on FPGA, operations involving very sparse
constants (many zeros in binary) cost less because short-circuit carry logic eliminates the
zero-bit multiplications. EBOPs's "effective constant bitwidth" rule (non-zero-bit span only)
captures this hardware behaviour directly.

**B_w = 1 (binary {-1,+1}) interpretation in EBOPs:**
The paper does NOT have explicit special-casing for b_w=1. Under the constant-bitwidth rule:
- A weight of +1 in binary is "1" = 1 non-zero bit -> b_i = 1.
- A weight of -1 in two's complement is typically all ones, but as a *signed* 1-bit quantity it
  still encodes in 1 effective bit.
Therefore EBOPs per binary-weight multiply = b_i * b_j = 1 * b_a (where b_a is activation
bit-width, e.g. 8). This is the minimum possible cost per MAC and is why the binary model is
efficient: its EBOPs count is exactly b_a times the number of MACs, vs. b_w * b_a for a W8A8
model (8x reduction for same MACs).

**FLAG — accumulator ambiguity in EBOPs for b_w=1:**
When b_w=1, the multiply b_i * b_j = 1 * b_a = b_a per scalar. The adder-tree still widens
the result by log2(fan_in) bits, but EBOPs intentionally excludes this term (accumulator
"implicitly counted" claim). This means EBOPs slightly underestimates the actual FPGA adder
cost for the binary network. For the purposes of BNJetTag, this is acceptable and consistent
with how HGQ/EBOPs is applied in the field.

---

### 3. BOPs/EBOPs with b_w = 1 binary weights — full analysis

Under **simplified BOPs** (MACs * b_w * b_a):
- Binary W1A8: BOPs = MACs * 1 * 8 = 8 * MACs.
- W8A8 baseline: BOPs = MACs * 8 * 8 = 64 * MACs.
- **Ratio: 8x fewer BOPs for the binary model at the same number of MACs.**

Under **full UNIQ BOPs** (including accumulator):
$$\text{BOPs}_\text{W1A8} = mn(1 \cdot 8 + 8 + 1 + \log_2 n) = mn(9 + 1 + \log_2 n)$$
vs.
$$\text{BOPs}_\text{W8A8} = mn(64 + 8 + 8 + \log_2 n) = mn(80 + \log_2 n)$$
The ratio narrows somewhat from 8x to roughly 10/(80+log2n) at the low end, but the binary
model is still enormously cheaper.

Under **EBOPs** (HGQ, b_w=1 effective):
- Per scalar MAC: $b_i \cdot b_j = 1 \cdot b_a$.
- Summed over layer: $\text{EBOPs}_\text{layer} = mn \cdot 1 \cdot b_a = mn \cdot b_a$.
- vs W8A8: $\text{EBOPs}_\text{W8A8} = mn \cdot 8 \cdot b_a$.
- **8x EBOPs reduction at same MACs.**

On hardware: for b_w=1 the multiply becomes a sign-select (0 DSP), and the EBOPs formula
correctly predicts DSP=0 because there is no real multiplier at all — any multiply by ±1 is
implemented as conditional negation in LUT logic. hls4ml uses ap_uint<1> for 1-bit weights,
which physically cannot use the DSP block's multiplier input (DSPs require multi-bit operands).
This is confirmed in BNJetTag's own csynth results (DSP=0 for binary FFN, 2026-06-24 synthesis).
(source: BNJetTag project results, qkeras-bitnet-run-2026-06-22/results/hls_resource_table.md)

The UNIQ paper confirms the physics: "reducing both bitwidths by a factor of two reduces the
amount of required hardware by roughly a factor of four" — this is the quadratic scaling of the
b_w * b_a term.
(source: https://ar5iv.labs.arxiv.org/html/1804.10969, accessed 2026-06-29)

---

### 4. Efficiency argument: grow the model at equal BOPs/EBOPs

**BitNet b1.58 paper (arXiv:2402.17764) frames this directly:**
"13B BitNet b1.58 is more efficient, in terms of latency, memory usage and energy consumption,
than 3B FP16 LLM."
(source: https://arxiv.org/html/2402.17764v1, accessed 2026-06-29)

"BitNet b1.58 saves 71.4 times arithmetic operations energy consumption for matrix multiplication
on 7nm chips." This is grounded in the b_w * b_a energy scaling of integer MAC units.

**Quantitative BOPs version of the argument (not explicitly stated in BitNet papers but
derivable from BOPs formula):** A W1A8 model with N_1 MACs has:
BOPs_1 = N_1 * 1 * 8 = 8 N_1. A W8A8 model with N_8 MACs has BOPs_8 = 64 N_8.
At equal BOPs budget: 8 N_1 = 64 N_8 => N_1 = 8 N_8. The binary model can have **8x more
MACs** (and therefore ~8x more parameters in a linear layer) at the same BOPs cost. This is
the "grow the model" argument.

**Ps and Qs (arXiv:2102.11289) explicitly uses BOPs as the Pareto axis:**
Figure 5 in Ps and Qs plots accuracy vs. BOPs, showing Pareto fronts for quantization+pruning
combinations. This is the standard framing in the hls4ml / HEP-ML community.
(source: https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2021.676564/full,
accessed 2026-06-29)

**HGQ (arXiv:2405.00645) uses EBOPs as the regularisation axis** during training:
"HGQ [...] EBOPs [used] as a regularization term in the loss function, controlled by a
hyperparameter β, which allows the user to control the trade-off between accuracy and resource
usage during training." This is the analogous axis: accuracy vs. EBOPs, not accuracy vs. FLOPs.

---

### 5. Summary of key citations

| Item | Citation | URL |
|------|----------|-----|
| BOPs original definition | Baskin et al., "UNIQ," arXiv:1804.10969, ACM TOCS 2021 | https://arxiv.org/abs/1804.10969 |
| BOPs simplified (MACs*bw*ba) | van Baalen et al., "Bayesian Bits," NeurIPS 2020, arXiv:2005.07093 | https://arxiv.org/abs/2005.07093 |
| BOPs simplified (HAWQ-V3 form) | Yao et al., "HAWQ-V3," ICML 2021, arXiv:2011.10680 | https://arxiv.org/abs/2011.10680 |
| BOPs with pruning, Eq.8 | Banerjee et al. "Ps and Qs," Frontiers AI 2021, arXiv:2102.11289 | https://arxiv.org/abs/2102.11289 |
| **EBOPs definition (Eq.5)** | **Sun et al. "HGQ," arXiv:2405.00645, 2024** | **https://arxiv.org/abs/2405.00645** |
| EBOPs ≈ LUT+55*DSP mapping | Sun et al. "HGQ," Fig. II / §V.1 | https://arxiv.org/abs/2405.00645 |
| Efficiency = grow 1-bit model | Ma et al. "BitNet b1.58," arXiv:2402.17764 | https://arxiv.org/abs/2402.17764 |
| Accuracy-vs-BOPs Pareto axis | Banerjee et al. "Ps and Qs," arXiv:2102.11289 | https://arxiv.org/abs/2102.11289 |

## 2026-06-28 — Built the BNJetTag paper library + reference-code

- Assembled a **26-paper rich-brief library** under `papers/`, in four areas: BitNet &
  1-bit/ternary, jet tagging & transformers, hls4ml & FPGA triggers, QAT & binary-NN
  foundations. Index: `papers/README.md`. PDFs fetch via `papers/download_papers.sh`
  (this sandbox can't reach arxiv.org; arxiv egress blocked).
- arXiv IDs confirmed via web search. Anchors: BitNet 2310.11453; BitNet b1.58 2402.17764;
  BitNet b1.58 2B4T report 2504.12285 (source: https://arxiv.org/abs/2504.12285); ParT
  2202.03772; hls4ml 1804.06913; AutoQKeras 2006.10159; GarNet 2008.03601; Ultrafast jet
  classification on FPGAs 2402.01876 (source: https://arxiv.org/abs/2402.01876).
- Cloned best reference code into `reference-code/` (snapshots, .git stripped):
  google/qkeras, jet-universe/particle_transformer (+ pretrained ParT/ParticleNet weights),
  hqucms/weaver-core, fastmachinelearning/hls4ml, kyegomez/BitNet, microsoft/BitNet.
- Best HuggingFace artifacts (pull via `reference-code/download_huggingface.sh`):
  microsoft/bitnet-b1.58-2B-4T {packed, -bf16, -gguf}
  (source: https://huggingface.co/microsoft/bitnet-b1.58-2B-4T), jet-universe/sophon,
  datasets jet-universe/jetclass + jetclass2 (source: https://huggingface.co/datasets/jet-universe/jetclass).
- Relevance: closes the literature gap for the thesis (binary weights → ~0 DSP; A8/A6/A4
  activation axis) and provides runnable QAT / ParT / hls4ml references.
