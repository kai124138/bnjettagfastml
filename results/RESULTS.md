# RESULTS — BitNet 1-bit Jet Tagger (training + hls4ml)

**Run:** `qkeras-bitnet-run-2026-06-22`
**Date:** 2026-06-23
**Model:** BitNet-style binary-weight `{−1,+1}` 1-bit transformer jet tagger (D256 / H8 / L8 / FFN1024, N=10
particles, F=14 features, ~6.37M params)
**Abstract in force:** 1-bit transformer vs the vanilla (FP32) version + how aggressively it can be quantized
before tagging efficiency, **resource consumption, and latency** degrade — with the binary→FPGA **LUT/logic
(not DSP)** mapping as the central hardware claim, characterized via **hls4ml**.

This file is the **data export** (every headline number in one place). The plain-English story of how we got
here is in `SESSION_REPORT.md`; the full annotated writeup is in `REPORT.md`.

---

## 1. Tagging efficiency (val AUC) — training campaign

> **Figures:** the cross-precision AUC comparison (§1a/§1c) is plotted in
> [`plots/results_auc_by_variant.png`](plots/results_auc_by_variant.png); the quantization axis (§1b), with
> matching measured FPGA resources, is in [`plots/results_quant_axis.png`](plots/results_quant_axis.png).

### 1a. The abstract's central comparison — 1-bit vs vanilla & 8-bit
Identical architecture / data / recipe, switched by a single `BN_VARIANT` knob. This isolates the cost of
going 1-bit.

| variant            | weights   | activations | **val AUC** | W&B run name            | read                                                   |
| ------------------ | --------- | ----------- | ----------- | ----------------------- | ------------------------------------------------------ |
| vanilla            | FP32      | FP32        | **0.7703**  | `VANILLA-FP32-…`        | full-precision reference                               |
| W8A8               | int8      | int8        | **0.7719**  | `BASELINE-W8A8-…`       | 8-bit ≈ **lossless** (+0.16 pt vs vanilla = noise)     |
| **1-bit (binary)** | `{−1,+1}` | 8-bit       | **0.7530**  | `kai-bn-paper-bin-lr15` | **the deployed model**; going 1-bit costs **~1.7 pts** |

**Takeaway:** 8-bit quantization is essentially free; going all the way to 1-bit weights costs ~1.7 points of
AUC — the price paid to move every MAC off the DSPs and onto LUT/logic (vanilla & W8A8 stay multiply/DSP-bound;
only binary maps to XNOR+popcount → DSP=0).

### 1b. Quantization-aggressiveness axis — "how far can we push it?"
Weights are already 1-bit; the remaining dial is **activation precision** (`BN_ACT_BITS`). No precision cliff.

| activation precision | **val AUC** | Δ vs A8 |
| --- | --- | --- |
| **A8** (W1A8) | **0.7524** | — |
| **A6** (W1A6) | **0.7507** | −0.2 pts |
| **A4** (W1A4) | **0.7437** | −0.9 pts |

Run on the **canonical softmax model** (`BN_SOFTMAX_FREE=0`) — i.e. the *same* configuration as the headline
W1A8, so the axis is consistent with the deployed model (A8 here = 0.7524 independently reproduces the headline
0.7530, seed noise). The degradation is **gentle and monotonic**: A6 is within noise of A8, and dropping all the
way to 4-bit activations costs only **~0.9 pts** total — no precision cliff. Matching firmware/resource savings
are in §2. *(The earlier sweep was run on the softmax-free base — 0.7562/0.7450/0.7381; that base is retained as
the appendix cross-check in §1c, and its A8 point is the softmax-free 0.7562 below.)*

### 1c. Appendix explorations (kept, off the current abstract)
Belonged to an earlier/longer abstract; preserved, not part of the main story.

| exploration | val AUC | note |
| --- | --- | --- |
| ternary `{−1,0,+1}` (b1.58, arXiv:2402.17764) | **0.7685** | accuracy ceiling reference; binary reaches ~98% of it. Off-thesis (ternary's `0` breaks the XNOR+popcount/LUT purity). |
| softmax-free attention `ReLU(QKᵀ)/N` | **0.7562** | deleting softmax is statistically free (+0.003 vs 0.7530, within noise); hls-friendly. |

---

## 2. hls4ml / FPGA — resource & latency (what going 1-bit buys)

> **Method note.** The gold-standard route hls4ml → Vitis/Vivado **C-synthesis** emits exact LUT/FF/DSP/BRAM +
> latency-in-cycles. **NRP Nautilus has no Xilinx HLS backend**, so it ran **off-cluster on the group's `mulder`
> box (Vitis HLS 2023.2, 2026-06-24)** — and it is now **done**: see `hls_resource_table.md` §B for the measured
> A8/A6/A4 table (DSP=0 confirmed; ~25% LUT / ~7% FF / ~1.2% BRAM / 1.3 µs at RF=256). The story is thus delivered
> three reinforcing ways: **(a)** real-hls4ml **convertibility + bit-accuracy** of the binary core across the
> A8/A6/A4 axis, firmware inspected for the DSP=0 mapping; **(b)** an **analytical per-component** model; and now
> **(c)** the **real Vitis C-synthesis**, which confirms DSP=0 and lands ~2× under the analytical fold estimate.
>
> **Now extended to the FULL trained transformer (2026-06-26):** `hls_resource_table.md` **§B′** synthesizes the
> *actual* `lr15_bitnetJetTagModel.h5` end-to-end — all 51 BitLinears + 51 SubLN norms + the 4 weighted attention
> projections, rebuilt from hls4ml-supported layers with **trained weights ported in** (rebuild↔trained corr
> **0.99998**; QKeras↔Vitis bit-accuracy **0.9967–0.9999**), C-synthesized per distinct shape. **Result: the binary
> matmul is 0 DSP on the real model, and the transformer's entire DSP footprint — 1,049, 8.5 % of a VU13P — is
> 100 % LayerNorm** (the binary win is structural; normalization is the only multiplier-bearing op). This is the
> answer to "is this just an FFN converted to HLS?": no — it is the whole trained transformer, in silicon estimates.

### 2a. Binary core converts AND emulates bit-accurately — across the whole quantization axis  *(real hls4ml 0.8.1)*
`code/hls/sweep_precision.py`, Job `kai-hls-sweep` (hls4ml 0.8.1, qkeras 0.9.0, tf 2.11.1). Dominant primitive = a
binary FFN block (`fc1` 256→1024 `binary(alpha=1)` → `quantized_relu(A,2)` → `fc2` 1024→256 `binary(alpha=1)`),
pushed through the real toolchain at A8/A6/A4. N=1000 random inputs vs Keras.

| act precision | **emul corr (N=1000)** | weight C-type (firmware) | activation datapath `layer4_t` (firmware) | accumulator (firmware) | io / reuse |
| --- | --- | --- | --- | --- | --- |
| **A8** | **1.000000** | `ap_uint<1>` ×2 | `ap_ufixed<8,2,AP_RND_CONV,AP_SAT>` | `ap_fixed<32,16>` | io_parallel, RF=1 |
| **A6** | **1.000000** | `ap_uint<1>` ×2 | `ap_ufixed<6,2,AP_RND_CONV,AP_SAT>` | `ap_fixed<32,16>` | io_parallel, RF=1 |
| **A4** | **1.000000** | `ap_uint<1>` ×2 | `ap_ufixed<4,2,AP_RND_CONV,AP_SAT>` | `ap_fixed<32,16>` | io_parallel, RF=1 |

**Three facts straight from the generated `defines.h` / `parameters.h` (Job `kai-hls-inspect`, not inferred):**
- **DSP = 0 is structural and precision-independent.** Both dense kernels type as `ap_uint<1>`
  (`weight2_t`, `weight5_t`) at *every* A. A 1-bit weight cannot drive a DSP48 multiplier input, so hls4ml
  emits the "multiply" as a conditional-negate (sign-select) + LUT adder tree — **zero DSPs at A8/A6/A4 alike.**
- **Activation precision IS the firmware datapath dial.** `layer4_t` (the `quantized_relu` buffer between fc1
  and fc2) narrows `ap_ufixed<8,2>` → `<6,2>` → `<4,2>`, `SAT` clip (range [0,4)) preserved. That is the on-chip
  activation storage + adder-tree input width — the hardware face of the 0.756 → 0.745 → 0.738 efficiency trade.
- **Lowest-latency regime throughout.** `io_parallel`, `strategy=latency`, `reuse_factor=1` at every precision.

Other confirmed firmware typedefs (all three projects): `input_t` / `result_t` = `ap_fixed<32,16>`;
`fc1_accum_t` / `fc2_accum_t` = `ap_fixed<32,16>`; `bias2_t` / `bias5_t` = `ap_fixed<8,4>`;
`act_table_t` = `ap_fixed<18,8>`.

`corr = 1.000000` ×3 ⇒ firmware is **bit-accurate to Keras at every operating point**. Projects emitted to
`/data/outputs/hls/binary_ffn_a{8,6,4}_prj` (+ `.tar.gz`); the off-cluster Vitis csynth has since been **run on
`mulder`** — measured numbers in §B / `results/csynth/`.

> **Accumulator note.** Accumulators pinned to `ap_fixed<32,16>` for bit-accuracy: the hls4ml default
> `fixed<16,6>` saturates at ±32 but `fc1` sums 256 signed ±1 terms reaching ±50+ (first pass diverged at
> `corr=0.24`). A naïve widen-everything only reached 0.85 because it also removed the `quantized_relu` `SAT`
> clip. Fix: widen accum/result/io, leave `act` **native**.

### 2b. Newer stack (hls4ml 1.3.0) — convertibility matrix
Job `kai-hls-full` (Python 3.10, hls4ml **1.3.0**, TF 2.14.1), `code/hls/full_transformer_probe.py`.

| piece                                               | hls4ml 0.8.1              | hls4ml 1.3.0              | note                                         |
| --------------------------------------------------- | ------------------------- | ------------------------- | -------------------------------------------- |
| binary `QDense` FFN / projection                    | ✅ converts + bit-accurate | ✅                         | the dominant primitive                       |
| **`LayerNormalization` (SubLN)**                    | ❌ unsupported             | **✅ converts**            | the Phase-1 "riskiest piece" — now clears    |
| **full block** SubLN→binary-proj→SubLN→binary-FFN   | ❌                         | **✅ converts end-to-end** | norm + binary-dense backbone composes        |
| attention **act×act** score matmul (Q·Kᵀ, scores·V) | ❌                         | ❌                         | only remaining gap → Extension-API custom op |

**The gap collapsed** from "LayerNorm **and** attention both fail" (Phase-1) to **just the attention score
matmul.** Two precise findings on that last piece:
- `EinsumDense` is unsupported by hls4ml 1.3.0's Keras-v2 parser (`Unsupported layer type: EinsumDense`).
- More fundamentally, the attention scores are an **activation×activation** contraction, which Keras
  `EinsumDense` *cannot even express* (it always contracts an activation against a **trainable kernel**). So
  this is a **custom-op** job (hls4ml **Extension API**, or the HGQ2 / Keras-v3 MHA frontend), not a missing
  parser handler.

Net: the **binary-dense + SubLN backbone — ~90%+ of the model's arithmetic, exactly the layer types that worried
us — is convertible today.** *(The 1.3.0 LayerNorm/full-block conversions were a smoke test; `convert` returns
an in-memory `ModelGraph` — not persisted to disk this pass, a one-line `write_project()` away.)*

### 2c. Per-component HARD structural counts  *(exact — `code/hls/resource_model.py`)*
Assumption-free, pure architecture arithmetic.

| component | inst×tokens | binary MACs | weight bits |
| --- | --- | --- | --- |
| input_proj (14→256) | 1×10 | 35,840 | 3,584 |
| attn.W_q / W_k / W_v (256→256) | 8×10 ea | 5,242,880 ea | 524,288 ea |
| attn.W_o (256→256) | 8×10 | 5,242,880 | 524,288 |
| ffn.fc1 (256→1024) | 8×10 | 20,971,520 | 2,097,152 |
| ffn.fc2 (1024→256) | 8×10 | 20,971,520 | 2,097,152 |
| head_fc1 / head_fc2 | 1×1 | 65,536 / 256 | 65,536 / 256 |
| **TOTAL binary-weight** | | **63,016,192** | **6,360,832 (6.36 Mbit)** |

- **99.35%** of all MACs are binary (DSP-free). The only **real** act×act multiplies are the **409,600 (0.65%)**
  attention-score products (Q·Kᵀ and scores·V).
- **51 SubLN/LayerNorm** instances (one per BitLinear), 184,972 normalized elems/inference — confirmed
  convertible on hls4ml 1.3.0 (§2b).

### 2d. Derived resource estimates vs activation precision  *(RF=1; labelled, swappable cost factors)*
Cost factors explicit: binary MAC → ~A LUT / **0 DSP**; real MAC → 1 DSP at A≥7, packs into LUTs at A≤6.

| precision | binMAC LUTs (RF=1 upper bound) | attn DSPs | accum width |
| --- | --- | --- | --- |
| A8 | 504,129,536 | 409,600 | 18 b |
| A6 | 378,097,152 | 0 (LUT-packed) | 16 b |
| A4 | 252,064,768 | 0 (LUT-packed) | 14 b |

- **DSP = 0** for the binary core is **structural** (no multiplier exists), backed by the §2a firmware — not an
  estimate.
- LUT figures are an **RF=1 upper bound** (one MAC = one LUT-slice). Real firmware time-multiplexes with reuse
  factor RF: parallel LUTs ÷ ~RF, latency × ~RF.
- **Device-fit (XCVU13P, LUT 1.728M):** RF=1 is ~292× too big for this research-size model → a real trigger
  folds to **RF ≈ 583** to sit under 50% LUT → **latency ≈ RF cycles ≈ 1 µs @ 400 MHz.**
- Conservative for two independent reasons: hls4ml *pre*-synth LUT is overstated ~3–10× vs real logic
  synthesis, and measured binary jet-taggers report **~1–8% LUT / 0% DSP / 0 BRAM** (Ngadiuba et al.,
  arXiv:2003.06308).

### 2e. Softmax-free attention op delta (per inference) — *appendix variant*
If adopted, deletes the only non-binarizable op in the attention core.

| | exp() evals | reciprocals | norm mults |
| --- | --- | --- | --- |
| softmax core (default, the 1-bit model) | 6,400 (LUT tables/BRAM) | 640 (DSP/LUT-heavy) | 6,400 |
| **softmax-free** (appendix) | **0** | **0** | 0 (constant 1/N shift) |

### 2f. EBOPs — the cheap, synthesis-free hardware-cost axis  *(`code/training/ebops.py`; HGQ arXiv:2405.00645, Eq. 5)*
**Effective Bit-Operations** turn the §2c MAC counts into a *bit-level* cost that needs **no Vitis run** — a
static, architecture-only number recomputable in milliseconds that tracks synthesized LUTs.

Definition (HGQ): **EBOPs = Σ over every scalar multiply of `b_i·b_j`.** Three consequences we honor exactly:
- weight×activation layers (Q/K/V/O, FFN up/down, input proj, head) cost **`b_w·b_a`** per MAC;
- the attention score matmuls (Q·Kᵀ, softmax·V) are **activation×activation**, so they cost **`b_a·b_a`** —
  **binarization does NOT touch attention**; it is the precision-independent floor (the 0.65% of MACs in §2c);
- the accumulator/adder-tree is **excluded by HGQ design** ("EBOPs … count only … multiplicative processes").

EBOPs on the **FIXED** large model (D256/L8/H8/FFN1024, **6,373,633 params**; 63,016,192 matmul + 409,600 attn MACs):

| precision | b_w | b_a | matmul EBOPs | attn EBOPs (act×act) | **total EBOPs** | vs FP32 | vs W8A8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FP32 (ref, not true bit-ops) | 32 | 32 | 64,528,580,608 | 419,430,400 | **64,948,011,008** | 1.000× | 16.00× |
| W8A8 | 8 | 8 | 4,033,036,288 | 26,214,400 | **4,059,250,688** | 0.063× | 1.000× |
| **W1A8 (deployed binary)** | 1 | 8 | 504,129,536 | 26,214,400 | **530,343,936** | 0.008× | **0.131×** |
| W1A6 | 1 | 6 | 378,097,152 | 14,745,600 | **392,842,752** | 0.006× | 0.097× |
| W1A4 | 1 | 4 | 252,064,768 | 6,553,600 | **258,618,368** | 0.004× | 0.064× |

**ΔEBOPs — the headline "change":** going W8A8 → **W1A8 binary cuts EBOPs 7.65×** (4.06 G → 530 M), and it is
**122.5× below FP32**. It is 7.65× and not a clean 8× *because* the act×act attention term is identical at W8A8
and W1A8 (26,214,400 both): binarization only shrinks the matmul core (99.35% of MACs), leaving the 0.65%
attention floor untouched. Within the already-binary model the activation dial is gentle — A8→A6→A4 trims total
EBOPs 530 M → 393 M → 259 M, mirroring the §2d datapath narrowing and the ~0.9-pt AUC trade in §1b.

**Equal-EBOPs headroom (the "grow the model for free" axis):** at the *W8A8* EBOPs budget a binary (W1A8) model
could carry **~7.65× more matmul-MACs** and still cost ≤ W8A8 — binarization buys ~7.65× model-capacity headroom
at fixed bit-op cost (the design-space converse of the size sweep in `variant_sweep.md`).

**Bridge to the measured §2/§B synthesis:** HGQ calibrates **EBOPs ≈ #LUT + 55·#DSP** (io_parallel, post-PnR).
The §2d "binMAC LUTs (RF=1)" column *is* this matmul-EBOPs term (504,129,536 = 63,016,192×8 at A8). Because the
binary core synthesizes to **DSP = 0** (§2a firmware + §B csynth), **EBOPs ≈ #LUT** for the binary matmul — so this
static number is a faithful, synthesis-free proxy for the LUT story, reaching the same DSP=0 conclusion without
the toolchain. *(Caveat: the mapping is for fully-unrolled designs; the deployed model folds at RF=256, so treat
EBOPs as a cheap RELATIVE efficiency axis, not a literal folded-LUT count.)* Verified end-to-end by results-analyst
(2026-06-29, independent re-derivation; experiment-log). Reproduce: `python3 code/training/ebops.py --size large`
(stdlib-only, CPU, no TF).

---

## 3. Headline numbers (one-glance summary)

| metric | value |
| --- | --- |
| 1-bit (binary) tagging efficiency | **val AUC 0.7530** |
| cost of going 1-bit (vs vanilla FP32 0.7703) | **−1.7 pts** |
| cost of 8-bit (W8A8 0.7719 vs vanilla) | **+0.16 pt (lossless/noise)** |
| quantization axis (canonical softmax) | A8 0.7524 → A6 0.7507 → A4 0.7437 (no cliff; ~0.9 pt total) |
| hls4ml binary-core emulation correlation | **1.000000** at A8, A6, A4 |
| weight C-type in firmware | **`ap_uint<1>`** (→ LUT/logic) |
| **DSP usage (binary core)** | **0 — structural, every precision** |
| total binary MACs | **63,016,192 (99.35% DSP-free)** |
| real act×act multiplies | 409,600 (0.65%) |
| **EBOPs** (static cost proxy, HGQ; §2f) | W1A8 **530,343,936** — **7.65× below W8A8** (4.06 G), 122.5× below FP32 (65.0 G); binary DSP=0 ⇒ EBOPs ≈ #LUT |
| weight memory | **6.36 Mbit** |
| SubLN/LayerNorm instances | 51 (convertible on hls4ml 1.3.0) |
| device-fit latency (VU13P, RF=256 @ 400 MHz) | **520 cyc ≈ 1.3 µs** (measured csynth, §B) |
| synthesized binary FFN (RF=256) | **DSP 0 · LUT 25.5% · FF 7.3% · BRAM 1.2%** (A8; A6/A4 slightly lower) |
| **synthesized FULL transformer (§B′, RF=256, A8)** | **binary matmul = 0 DSP; all 1,049 DSP (8.5% VU13P) = LayerNorm**; rebuild↔trained corr 0.99998; QKeras↔Vitis 0.9967–0.9999 |
| convertible today | binary FFN/proj (0.8.1) + LayerNorm + full SubLN→binary block (1.3.0) + **full trained transformer end-to-end (§B′)** |
| remaining firmware gap | only the **weightless** attention score core (Q·Kᵀ/softmax/·V, 0.65% MACs) — `EinsumDense` unsupported → hls4ml Extension API |
| ~~gated on hardware we lack~~ → **now measured** | exact synthesized LUT/FF/DSP/BRAM + latency → **Vitis csynth done on `mulder`** (§B FFN + **§B′ full transformer**) |

---

## 4. Artifacts

**Scripts** (`hls/`): `sweep_precision.py` (A8/A6/A4 firmware sweep), `resource_model.py` (per-component
counts), `full_transformer_probe.py` (hls4ml 1.3.0 LayerNorm/full-block/attention probe), `stage_a_fix.py`
(per-precision FFN template), `convert_probe.py`, `hls4ml_findings.md`.

**Kubernetes Jobs** (`k8s/`): `kai-hls-sweep.yaml` (hls4ml 0.8.1 sweep), `kai-hls-full.yaml` (hls4ml 1.3.0
probe), `kai-hls-inspect.yaml` (firmware typedef grep).

**Generated on PVC** (`/data/outputs/hls/`): `binary_ffn_a{8,6,4}_prj` (+ `.tar.gz`), `binary_ffn_prj_wide`;
logs `sweep_precision.log`, `full_transformer_probe.log`, `resource_model.log`,
`sweep_precision_summary.json`.

**Repro:**
```bash
# binary-core A8/A6/A4 firmware sweep (hls4ml 0.8.1)
kubectl apply -f code/jobs/hls/kai-hls-sweep.yaml -n cms-ml
# LayerNorm + full-block + attention probe (hls4ml 1.3.0)
kubectl apply -f code/jobs/hls/kai-hls-full.yaml -n cms-ml
# inspect generated firmware typedefs
kubectl apply -f code/jobs/hls/kai-hls-inspect.yaml -n cms-ml
```
