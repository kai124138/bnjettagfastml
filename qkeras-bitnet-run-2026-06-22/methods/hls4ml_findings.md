# hls4ml conversion — support matrix & resource-estimation findings

Captured 2026-06-23 to back the REPORT.md hls4ml section. Versions: **hls4ml v1.3.0**
(transformer features since **v1.2.0**, 2025-11). Backends: Vivado, Vitis, oneAPI,
Catapult, Quartus, Libero.

## 1. Binary/ternary `QDense` (the dominant primitive) — CONVERTIBLE
- QKeras `QDense` with `binary()`/`ternary()`/`quantized_bits` maps to hls4ml IR
  quantizers (`BinaryQuantizer` → {−1,1}). Binary weights map to **LUT/logic, NOT DSP**.
- **Anchor:** Ngadiuba, Di Guglielmo, Duarte et al. 2020, *Compressing deep neural
  networks on FPGAs to binary and ternary precision with hls4ml*
  ([arXiv:2003.06308](https://arxiv.org/abs/2003.06308) / MLST 2020). Their jet-tagging
  BNN/TNN: **0% DSP, ~1–8% LUT, 0% BRAM** (vs 57% DSP for the fixed-point baseline).
- **Caveats:** (a) our activations are 8/6/4-bit (not 1-bit), so a MAC is a binary-weight ×
  multi-bit-activation = **sign-select + popcount adder tree in LUTs** — still DSP-free, but
  more LUTs than a true W1A1 XNOR net. (b) Use `binary(alpha=1)` or power-of-two scale
  (`auto_po2`); a generic float `alpha` reintroduces a real multiplier.

## 2. LayerNormalization (SubLN) — CONVERTIBLE but fragile
- IR layer `LayerNormalization` exists; **Vivado/Vitis, io_parallel only** (PR #1109/#1110).
  Inverse-sqrt via a LUT with log-distributed inputs. io_stream unsupported; sensitive to
  feature-dim/variance range. d_model=256 is within range but is the riskiest piece.

## 3. Attention — softmax-free is the MORE convertible path
- Built-in MultiHeadAttention + softmax landed **v1.2.0 via the HGQ2 frontend, Vitis only**
  (ref design [arXiv:2409.05207](https://arxiv.org/abs/2409.05207): transformer on VU13P,
  ~2 µs, 6/10-bit). **It requires softmax** (optimized k-op LUT pipeline).
- The standalone QKeras-MHA PR (#1163) is **not merged**; MHA is HGQ2-native.
- **Our softmax-free ReLU(QKᵀ)/N is not a drop-in built-in**, BUT the two batched matmuls
  (QKᵀ, scores·V) are exactly **`EinsumDense`/`Einsum`** (IR layers present, #1424). Path:
  assemble attention as `EinsumDense` + plain `ReLU` + constant-scale, **avoiding the softmax
  layer entirely**. If the auto-converter won't assemble it, use the **Extension API**
  (https://fastmachinelearning.org/hls4ml/advanced/extension.html). → Removing softmax makes
  attention *easier* to convert, not just cheaper.

## 4. Resource numbers WITHOUT Vivado/Vitis — NOT natively possible
- hls4ml `profiling` only profiles **numerical ranges/weights** for precision tuning; it does
  **not** emit LUT/FF/DSP/BRAM. Real per-layer numbers need `hls_model.build()` → csynth →
  Vivado/Vitis (which NRP does not have).
- Surrogate predictors (architecture-only): **rule4ml** ([arXiv:2408.05314](https://arxiv.org/abs/2408.05314)),
  **wa-hls4ml** ([arXiv:2511.05615](https://arxiv.org/abs/2511.05615), 680k synthesized nets,
  ~10–30% err) — but trained mostly on Dense/Conv, so rough for attention/LayerNorm/binary.
- **HLS pre-synth LUT estimates are overstated ~3–10×** vs post-logic-synthesis → do not
  headline pre-synth LUTs; report MAC counts + DSP=0 (structural) + literature-anchored LUT%.

## 5. Analytical anchors / rules of thumb
- Binary/ternary jet-tagger: **0% DSP, ~1–8% LUT, 0 BRAM, O(100 ns)** ([2003.06308], Tables 4/6).
- hls4ml jet-tagging baseline: Duarte et al. 2018, [JINST 13 P07027](https://doi.org/10.1088/1748-0221/13/07/P07027)
  ([arXiv:1804.06913](https://arxiv.org/abs/1804.06913)); QKeras heterogeneous quant: Coelho et al.
  [arXiv:2006.10159](https://arxiv.org/abs/2006.10159).
- Binary MAC ≈ **~b LUTs** (b = activation bits) for conditional-negate + add; true 1×1 XNOR is
  sub-LUT. Fixed-point DSP MAC = **1 DSP**.
- Popcount/adder tree: summing N b-bit terms ≈ N−1 adders ≈ **O(N·log₂N) LUTs**. For a 256→256
  binary Dense the cost is the 256-input popcount trees, not multipliers.
- Surrogate empirical form: `LUTs ≈ α·N_MAC + β·N_ADD + γ·N_ACT`.

## Plan implied by the above
- **Convert for real (no Vitis needed to validate dataflow + numerics):** binary `QDense`
  projections + FFN + head as a QKeras model → `convert_from_keras_model` → inspect config
  (binary quantizers, DSP-free) → `compile()` (g++, bit-accurate emulation, no Vitis).
- **Scope as remaining firmware work:** softmax-free attention via `EinsumDense`+ReLU (or an
  Extension-API custom layer), and LayerNorm validation on our exact shapes.
- **Resources:** analytical per-component model (`code/hls/resource_model.py`), anchored to
  [2003.06308]; flag the need for a real csynth pass for final numbers.
