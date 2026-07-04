# Constraints map — what each layer type can/can't do through HGQ2 (0.1.9) + hls4ml (1.3.0)

Written 2026-07-04 during the HGQ2 rebuild of the era-2 BitNet transformer. Every row was
**established by execution** (local C-sim = compile()+predict on the Vitis backend, or a
failed conversion with the exact error), not by reading docs. Purpose: future model designs
start from this table instead of trial-and-error. Source of each finding: `code/hgq2/LEDGER.md`.

## Verdict legend
✅ converts + bit-exact · 🟡 converts with caveats · 🛠 needs the custom-layer recipe · ❌ blocked

| Construct | Verdict | Detail |
|---|---|---|
| **Binary {−1,+1} weights** | ✅ | `KBI(k0=1, b0=1, i0=1, RND, SAT_SYM, trainable=False)` passes ±1 kernels through bit-identically and reports **1 bit** to EBOPs. Representable set is {−1,0,1}; 0 is unreachable for pinned ±1 kernels (values outside ±0.5 would be needed). `b0=0` is a trap: representable set collapses to {0}. |
| **BitNet per-tensor β scale** | 🟡 | Arbitrary float β as `QuantizerConfig(scaler=β)` breaks bit-exactness (weight_t inflates to fixed<24,0>) and risks DSP inference on the constant multipliers. **Power-of-2 β is free but too lossy end-to-end** (−6.3 AUC pts at A8 across 51 layers, measured). The working recipe is **fold-aware**: Q/K β exact via the softmax `input_scaler` (exp-LUT fold, free); V β dropped exactly (next LN is scale-invariant); fc1/head_fc1 β exact via bias→b/β (next LN kills the scale); only residual contributors (input_proj, Wo, fc2, head_fc2 = 2L+2 sites) carry a CSD-2 β̃ (2-signed-digit, ≤4.5% err, DSP-free by the Vitis ≤2-digit rule, bit-exact). Measured cost at A8: −0.003 AUC vs exact β. |
| **Static activation quantizers** | ✅ | `KIF(k0=1, i0, f0, RND_CONV, SAT, trainable=False)`. RND_CONV = ties-to-even matches TF/numpy. Per-element heterogeneous i/f is native (io_parallel wires). **Per-channel MSE-optimal calibration beats max-calibration** decisively at low bits (A4: ΔAUC −0.078 → −0.017, measured). |
| **Dynamic per-token activation scaling (BitNet training-time quant)** | ❌ | Unimplementable in static fixed-point — no hls4ml/HGQ2 equivalent exists. Any rebuild substitutes static grids; measure the substitution, don't pretend it's free (A8 −0.005, A6 −0.017, A4 −0.017 AUC at per-channel-MSE calibration). |
| **Datalane defaults (the WRAP trap)** | 🟡 | Every unconfigured HGQ2 datalane quantizer defaults to **WRAP + trainable, uncalibrated**. At inference this silently wraps out-of-range values → garbage outputs with no error (cost one full debug round: AUC 0.50). Rule: **every quantizer in the graph gets an explicit frozen config**, or run `trace_minmax` calibration for WRAP ones. |
| **LayerNormalization (keras)** | ❌→🛠 | Blocked three verified ways in keras-v3/hls4ml-1.3.0 (no v3 handler; v2 fallback needs axis=(2,) and learnable γ/β; BitExact has no `produce_kif`). Also the stock `nnet_layernorm.h` inv-sqrt table only covers var ∈ (eps,1]. **Fixed via the custom-layer recipe**: `PSubLN` keras layer + keras-v3 handler + new `SubLN` IR + `_produce_kif`/`_request_kif` + Vitis templates + `nnet_subln.h` with range-reduced 1/√ (even-power-of-two shift onto [1/4,1), 4096-entry table, half-shift back). C-sim corr ≥ 0.9999999 for dims 16/256/1024, input var 0→1e6. Table-based ⇒ near-exact, not bit-exact. |
| **QMultiHeadAttention (native)** | 🟡 | Works and is bit-exact through hls4ml, BUT two 0.1.9/1.3.0 compat bugs need shims (registry key module path; keras-3.15 removed `EinsumDense.full_output_shape`) — `bnhgq2/compat.py`. Its internal structure is fixed: no hook to insert a norm before the output projection (SubLN-inside-Wo cannot be expressed) → for this model, compose from the same primitives instead. Key projection never gets a bias (hardwired). |
| **QEinsum (act×act contractions)** | 🟡 | The attention-core enabler (scores=QKᵀ, ctx=attn·V) — converts to the Einsum IR, **io_parallel + Latency strategy only** (RF acts via `multiplier_limit`). Upstream bug: `enable_iq=False` crashes `build()` on multi-input layers (`_iqs_confs` typo) — use explicit exact-passthrough frozen SAT grids instead (identity on the integer streams, keeps EBOPs live). |
| **QSoftmax** | 🟡 | Table-based stable softmax (exp LUT + inv LUT), bit-exact through hls4ml. `input_scaler` folds an arbitrary constant into the exp table for FREE — the sanctioned place for score scales (β_qβ_k/√d). Its exp/inv INPUT quantizers are WRAP datalane defaults (same trap) — set frozen SAT grids sized from calibrated score ranges; negative `f0` is legal and right for coarse-but-covering grids (score gaps ≫ step). Table sizes = 2^bits of the input grids — keep ≤ 12 bits. |
| **QEinsumDense** | ✅ | The workhorse. Head split/merge lives in the equation (`btd,dhe->bthe`, `bthe,hed->btd`) — no Reshape needed. `bias_axes` spanning non-channel axes gives quantized additive constant TABLES. |
| **Positional encoding (additive constant)** | ✅ | keras `Embedding` is blocked under BitExact (no produce_kif). Fold the PE table into the input projection's `bias_axes='td'` bias: exact, quantized, zero extra ops. (This model's PE is an untrained folded constant — extracted from the checkpoint's `model_config` JSON, not from weights.) |
| **Residual Add (float)** | ✅ | keras `Add` via the merge handler; BitExact carries exact growing widths. No broadcasting in `nnet_merge.h` — shapes must match exactly. |
| **GlobalAveragePooling1D** | 🟡 | `QGlobalAveragePooling1D` converts + bit-exact; non-power-of-2 sequence length (T=10) makes 1/T binary-infinite → very wide accumulator fraction. Acceptable at the head; avoid in hot paths. `QSum`/`QMeanPow2` have no keras-v3 handler. |
| **ReLU** | ✅ | plain keras layer, standard support. |
| **Rescaling / constant Multiply** | ❌ | no keras-v3 handler, no v2 fallback (without da4ml installed). Use `QBatchNormalization` (frozen per-channel affine, bit-exact) or fold into quantizer scalers / softmax input_scaler / bias tables. |
| **3-D QDense** | 🟡 | silently re-parsed as PointwiseConv1D — use QEinsumDense for 3-D instead (explicit, predictable). 2-D QDense is fine. |
| **hls4ml `bit_exact` kwarg** | 🟡 | `convert_from_keras_model(..., bit_exact=None)` silently OVERWRITES `hls_config['Model']['BitExact']` — always pass `bit_exact=True` explicitly. |
| **macOS local C-sim** | 🟡 | patch `ap_{int,fixed}_special.h` (std::complex fwd-decl → `#include <complex>`) AFTER `write()`, then `_compile()` (plain `compile()` re-writes and undoes it). Linux/mulder unaffected. |

## Where DSPs can hide (for the DSP-savings claim)
1. **SubLN**: variance + inverse-sqrt path — the model's known DSP consumer (old
   fixed<32,16> LayerNorm census: 1,049 DSP = 100% of model DSPs). The new range-reduced
   `nnet_subln.h` has 42×42-bit squares → measure, don't assume (csynth pending).
2. **Arbitrary-constant multiplies**: any const with >2 signed CSD digits may infer a DSP
   (Vitis rule, verified in literature). Keep every hardware constant ≤2 signed digits
   (power-of-2 or CSD-2) — β̃ snapping exists exactly for this.
3. **Wide accumulators** (GAP's 1/10, exact einsum streams) — adders, not multipliers:
   LUT cost, not DSP. But watch synthesis.
4. The binary matmul cores themselves: DSP=0 **structurally** (1-bit weights are
   wire/negation) — confirmed in prior csynth at A8/A6/A4 and re-checked per new probe.
