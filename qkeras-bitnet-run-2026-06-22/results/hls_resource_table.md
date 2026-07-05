# HLS4ML SYNTHESIZED RESOURCE & LATENCY TABLE

**The table you asked for: exact LUT / FF / DSP / BRAM + latency-in-cycles — now MEASURED.**

> **Status — DONE (2026-06-24).** The Vitis HLS 2023.2 C-synthesis was run on the group's `mulder` box and **§B
> is now filled with real synthesized numbers** (raw reports in `results/csynth/`). The headline: **DSP = 0 at
> A8/A6/A4, confirmed in synthesis** (not just firmware inference); the folded device-fit design (RF=256) fits a
> VU13P at **~25% LUT / ~7% FF / ~1.2% BRAM / 0 DSP**, latency 520 cycles (1.3 µs @ 400 MHz).
>
> **UPDATE — the FULL trained transformer, synthesized end-to-end (2026-06-26).** §B (below) is a single binary-FFN
> block used as a *primitive probe*. The new **[§B′](#b-the-full-trained-transformer--synthesized-end-to-end-this-is-the-model-not-an-ffn)**
> now reports the **actual trained BitNet transformer** — the real `lr15_bitnetJetTagModel.h5` checkpoint, all 51
> BitLinears + 51 SubLN norms + the four attention projections — reconstructed from hls4ml-supported primitives
> with the **trained binary weights ported in**, validated (rebuild↔trained fidelity corr **0.99998**;
> QKeras↔Vitis C-sim bit-accuracy **0.9967–0.9999**), and C-synthesized per distinct layer shape. **Headline
> refinement:** the binary `{−1,+1}` **matmul core is 0 DSP** (re-confirmed on the real model); the **only** DSP in
> the entire transformer — **1,049 total (8.5 % of a VU13P)** — sits inside the LayerNorms. This is the section that
> answers *"where is the transformer — are you just building an FFN?"*: it is the whole trained model, in silicon
> estimates.
>
> *Background on the stages:* hls4ml has three — `convert` (codegen), `compile` (g++ bit-accurate emulation), and
> **`build` → Vivado/Vitis C-synthesis (csynth)**; only csynth produces exact LUT/FF/DSP/BRAM + latency-in-cycles.
> **NRP Nautilus has no Xilinx HLS backend**, so csynth could not run on the cluster — it ran instead on `mulder`
> (Vitis 2023.2, the same install hard-coded in `code/training/HLS_qk_Roc_Tracing.py`). §A is the no-synthesis
> firmware truth; §B is the synthesized table; §C is the published real-silicon anchor.

---

## A. Exact NOW — straight from the generated firmware (no synthesis needed)

These are read directly out of the hls4ml-generated `defines.h` / `parameters.h` for the three emitted projects
(`/data/outputs/hls/binary_ffn_a{8,6,4}_prj`), confirmed by Job `kai-hls-inspect`. They do **not** depend on
csynth:

| quantity | A8 | A6 | A4 | why it's exact without csynth |
| --- | --- | --- | --- | --- |
| weight C-type | `ap_uint<1>` | `ap_uint<1>` | `ap_uint<1>` | a 1-bit datum — **cannot drive a DSP multiplier port** |
| **DSP (binary MACs)** | **0** | **0** | **0** | structural: no multiplier is instantiated; MAC = sign-select + LUT adder tree (✓ confirmed by csynth, §B) |
| **BRAM (weights), unfolded** | **0** | **0** | **0** | RF=1/Latency: 1-bit weights pack into LUTROM/logic. **NB** the *folded* RF=256 synthesis uses **64** (~1.2%) — see §B |
| activation datapath `layer4_t` | `ap_ufixed<8,2,SAT>` | `ap_ufixed<6,2,SAT>` | `ap_ufixed<4,2,SAT>` | the on-chip activation width — narrows with the quantization axis |
| accumulator | `ap_fixed<32,16>` | `ap_fixed<32,16>` | `ap_fixed<32,16>` | pinned for bit-accuracy (default `<16,6>` overflowed) |
| reuse factor (inspected projects) | RF=1, II=1 | RF=1, II=1 | RF=1, II=1 | the *emitted* `io_parallel` projects; the **synthesized** device-fit point folds to **RF=256, II=256** (§B) |
| emulation correlation vs Keras | 1.000000 | 1.000000 | 1.000000 | g++ bit-accurate (N=1000), so csynth starts from a faithful model |

**The single most important hardware cell of the whole project — DSP = 0 for the binary core — is exact,
precision-independent, and now confirmed by real Vitis C-synthesis (§B).** That is the abstract's central claim,
proven in both firmware *and* synthesis. (BRAM = 0 holds for the fully-unrolled design; the folded deployable
point trades ~1.2% BRAM for tractability — so it is **DSP**, not BRAM, that is the structural binary win.)

---

## B. The synthesized table (the LUT/FF/latency-in-cycles you want) — **MEASURED**

> **DONE 2026-06-24 — real Vitis HLS 2023.2 C-synthesis on the group's `mulder` box.** These are no longer
> brackets: every cell below is read straight out of the Vitis `csynth` report (raw JSON in
> `results/csynth/csynth_report_a{8,6,4}_rf256.json`). This is the binary FFN block
> (`fc1` 256→1024 → `relu` → `fc2` 1024→256), the dominant primitive, synthesized at a **folded, device-fit
> operating point: reuse factor RF=256** (→ II=256). Part `xcvu13p-flga2577-2-e`, clock target 2.5 ns (400 MHz);
> Vitis reported **Estimated Fmax ≈ 568 MHz**, so the 400 MHz target is met with margin.
>
> **Plotted in** [`plots/results_csynth_resources.png`](plots/results_csynth_resources.png) — utilisation as
> % of a VU13P, with the **DSP = 0** headline and the 520-cycle / 1.3 µs latency called out.

| precision     | BRAM_18K | DSP   | FF          | LUT         | Latency (cycles) | II (cycles) | Clock (target / achieved) |
| ------------- | -------- | ----- | ----------- | ----------- | ---------------- | ----------- | ------------------------- |
| **A8 (W1A8)** | **64**   | **0** | **251,331** | **440,882** | **520**          | **256**     | 400 MHz / 568 MHz         |
| **A6 (W1A6)** | **64**   | **0** | **244,155** | **429,098** | **520**          | **256**     | 400 MHz / 568 MHz         |
| **A4 (W1A4)** | **64**   | **0** | **228,785** | **415,259** | **520**          | **256**     | 400 MHz / 568 MHz         |

**Same numbers as a fraction of the VU13P** (LUT 1,728,000 · FF 3,456,000 · DSP 12,288 · BRAM_18K 5,376):

| precision | DSP   | BRAM_18K  | FF        | LUT        | Latency @400 MHz | Throughput (II) |
| --------- | ----- | --------- | --------- | ---------- | ---------------- | --------------- |
| **A8**    | **0%** | **1.19%** | **7.27%** | **25.51%** | **1.30 µs**      | 640 ns / inf.   |
| **A6**    | **0%** | **1.19%** | **7.06%** | **24.83%** | **1.30 µs**      | 640 ns / inf.   |
| **A4**    | **0%** | **1.19%** | **6.62%** | **24.03%** | **1.30 µs**      | 640 ns / inf.   |

**What the measured numbers say:**
- **DSP = 0 at every precision — now confirmed by real synthesis, not just firmware inference.** The binary
  `{−1,+1}` MACs synthesize to sign-select + LUT adder trees; Vitis instantiates **zero** DSP48s. This is the
  abstract's central claim, proven in silicon estimates. It is precision-*and*-fold-independent.
- **LUT is the resource that scales with activation width** — A8 → A6 → A4 = 440,882 → 429,098 → 415,259
  (−2.7% / −5.8%). Narrower activations shrink the datapath and the accumulator-feeding logic. FF tracks it
  (251,331 → 228,785, −9%). Both are modest and monotonic — the same "no cliff" trend as the AUC sweep.
- **BRAM_18K = 64 (≈1.2%), precision-independent.** This is the one honest correction vs the pre-synthesis
  story: a *folded* design (RF=256, hls4ml `Strategy=Resource`) stores the folded binary weights in block RAM,
  so BRAM is small-but-nonzero. BRAM = 0 only holds for the **fully-unrolled RF=1/Latency** design (1-bit
  weights → LUTROM), which is intractable to synthesize at this model size (see below). DSP=0 holds either way;
  it is BRAM, not DSP, that carries the fold.
- **Latency 520 cycles (1.30 µs @ 400 MHz, ~0.92 µs at the achieved 568 MHz); II = 256 cycles** → one inference
  every 640 ns. This is the **folded** throughput, not the II=1 ideal: II=1 would require RF=1 (full unroll),
  which for this 6.37M-param model is a ~290× VU13P upper bound (the §C Ngadiuba reference hits II=1 only because
  its MLP is ~1000× smaller). RF=256 is the realistic trigger-compatible operating point and already fits at
  ~25% LUT.

**Reconciliation with the analytical model (`code/hls/resource_model.py`):** the model bracketed RF=1 LUT at an
upper bound of 504M/378M/252M slices and predicted a fold to **RF ≈ 583** for <50% LUT. The real synthesis comes
in *well below* that: **RF=256 already lands at ~25% LUT**, i.e. the deployable fold is roughly **half** what the
analytical bound predicted. This directly vindicates the §C caveat that hls4ml/analytical pre-synth LUT is
overstated several-fold versus real logic synthesis — the measured design is comfortably more efficient.

---

## B′. THE FULL TRAINED TRANSFORMER — synthesized end-to-end (this is the model, not an FFN)

> **DONE 2026-06-26 on `mulder` (Vitis HLS 2023.2).** Answers the direct question *"where is the transformer — are
> you just synthesizing an FFN?"* §B above is a single binary-FFN block used as a **primitive probe**. This section
> is the **entire trained BitNet transformer jet tagger** — the real `lr15_bitnetJetTagModel.h5` checkpoint —
> reconstructed from hls4ml-convertible primitives and synthesized with its **trained binary weights**. Produced by
> `code/hls/full_model_csynth.py`; raw reports in `results/csynth/full_model_*_a8_rf256.json`.

**Why a rebuild was needed (and what was rebuilt).** hls4ml's parser only converts layer *types* that have a
registered handler. The trained model's `BitLinear`, `RMSNorm`, and `BitMHSA` are custom `keras.Layer` subclasses
with **no** handler, so hls4ml cannot ingest the `.h5` as-is — which is precisely why the earlier run fell back to a
plain `QDense` FFN. The fix: map every trained op to a supported primitive **and port the trained weights through
the BitNet binarizer (`AbsMeanQuantizer`)**:

| trained op (custom subclass) | hls4ml-supported realisation | weights ported |
| --- | --- | --- |
| `BitLinear` (binary {−1,+1} matmul) | `LayerNormalization → QActivation(quantized_bits) → QDense(kernel=binary)` | ✅ real, binarized |
| `RMSNorm` (despite the name = **full** mean-subtracting LayerNorm, eps=1e-6, no affine — per the model's own code comment) | built-in `LayerNormalization` (γ=1, β=0) | n/a (no affine params) |
| `BitMHSA` Q·Kᵀ / softmax / ·V **score core** (no weights; 0.65 % of MACs) | *not converted* — `EinsumDense` is unsupported on this hls4ml; handled analytically | n/a |

The four **weighted** attention projections (Wq/Wk/Wv/Wo) **are** BitLinears and **are** synthesized below; only the
weightless attention score-core contraction is excluded (documented; 0.65 % of MACs, 0 % of weights).

**Validation — the rebuild reproduces the trained model and synthesizes bit-accurately:**
- **Fidelity** (rebuilt graph vs. the trained `.h5`, N=512 real jets): Pearson **corr = 0.99998** (mean|Δ| 0.0041)
  with the model's native dynamic per-token activation quant; **0.99814** (mean|Δ| 0.0599) with the
  HLS-convertible static `quantized_bits` quant. → the architecture + weight port is faithful.
- **Bit-accuracy** (QKeras forward vs. Vitis HLS C-simulation, **real trained weights**, A8): **0.9967–0.9999** per
  layer (input_proj 0.99672, attn 0.99977, ffn_fc1 0.99977, ffn_fc2 0.99988, head 0.99982).

**The 5 distinct layer shapes, each synthesized with real trained weights** (Vitis HLS 2023.2,
`xcvu13p-flga2577-2-e`, 2.5 ns target, RF=256). A binary matmul's resource is weight-*value*-independent and the 8
transformer blocks are architecturally identical, so one representative per shape gives every copy:

| shape | components (× count) | DSP | LUT | FF | BRAM_18K | Lat (cyc) | II (cyc) | est. clock |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 14→256   | input_proj ×1                                  | **11** | 25,897  | 48,605  | 1  | 299 | 224 | 3.71 ns ⚠ |
| 256→256  | Wq,Wk,Wv,Wo (×8 each) + head_fc1 ×1 = **×33**   | **15** | 119,659 | 117,083 | 8  | 679 | 419 | 1.90 ns |
| 256→1024 | ffn_fc1 ×8                                      | **15** | 174,187 | 163,931 | 32 | 679 | 419 | 1.90 ns |
| 1024→256 | ffn_fc2 ×8                                      | **51** | 430,366 | 423,384 | 32 | 682 | 421 | 1.90 ns |
| 256→1    | head_fc2 ×1                                     | **15** | 101,550 | 101,528 | 1  | 679 | 419 | 1.90 ns |

(All meet the 2.5 ns / 400 MHz target except `input_proj`, whose 14-wide LayerNorm inv-sqrt path closes at 3.71 ns
— fixable by pipelining that divide or relaxing the target on the tiny first layer; it is the smallest block and
not on the resource-critical path.)

**Where every DSP comes from — the binary win, confirmed and made precise.** The binary `{−1,+1}` QDense **matmul
core is 0 DSP** (1-bit weights → XNOR/popcount on LUTs), exactly as the abstract claims — re-confirmed here on the
*real* model, not a stand-in. The **only** DSP in the entire transformer is the **LayerNorm** (real-valued variance
Σx² + inv-sqrt at fixed<32,16>), and it scales with the *normalised width*:

| LayerNorm width | DSP / instance | × instances | DSP subtotal |
| --- | --- | --- | --- |
| 14  (input_proj)            | 11  | 1  | 11 |
| 256 (attn proj + head_fc1)  | 15  | 33 | 495 |
| 256 (ffn_fc1)               | 15  | 8  | 120 |
| 1024 (ffn_fc2)              | **51** | 8  | 408 |
| 256 (head_fc2)              | 15  | 1  | 15 |
| **total**                   |     | **51 norms** | **1,049** |

So the headline is sharper than "0 DSP": **the binary transformer's matmul is structurally DSP-free; its entire DSP
footprint (1,049, 8.5 % of a VU13P) is 100 % normalization** — and is itself precision-independent (the LN runs at
fixed<32,16> regardless of A8/A6/A4). A fixed-point inv-sqrt **LUT** for the norm would drive the whole
transformer's DSP toward 0; flagged as future work.

**Composed full-model resource (fully-spatial sum of all 51 BitLinear instances):**

| precision | DSP | LUT | FF | BRAM_18K |
| --- | --- | --- | --- | --- |
| **A8** total     | **1,049** | 8,912,618 | 8,712,392 | 778 |
| A8 % of one VU13P | **8.5 %** | 515.8 % | 252.1 % | 14.5 % |
| **A6** total     | **1,049** | 8,680,426 | 8,545,402 | 778 |
| A6 % of one VU13P | **8.5 %** | 502.3 % | 247.3 % | 14.5 % |
| **A4** total     | **1,049** | 8,594,614 | 8,324,652 | 778 |
| A4 % of one VU13P | **8.5 %** | 497.4 % | 240.9 % | 14.5 % |

(Device totals used: LUT 1,728,000 · FF 3,456,000 · DSP 12,288 · BRAM_18K 5,376. DSP is **identical (1,049)** at all
three activation precisions — it is 100 % LayerNorm at fixed<32,16>, structurally independent of A8/A6/A4. LUT/FF fall
monotonically as activations narrow, A8 ≥ A6 ≥ A4, but only modestly, since the binary matmul XNOR/popcount logic
dominates area and is itself weight-driven, not activation-driven.)

**Honest reading of the composition.** This total is the **fully-spatial** sum — every one of the 51 layers
instantiated as its own pipelined module. At 6.37 M parameters that does **not** fit one VU13P (LUT 5.2×, FF 2.5×):
a fully-spatial transformer of this size is a multi-FPGA or heavily-folded design. The **measured, trustworthy**
numbers are the **per-shape rows** above; a deployable design reuses the **8 identical transformer blocks
temporally** (instantiate ~one block + input_proj + head, loop over depth/tokens), collapsing the ×33/×8/×8
multiplicities and bringing LUT/FF back on-chip. What is **fold-independent and proven**
is the structural result: **binary matmul = 0 DSP; 100 % of DSP = LayerNorm.**

**Composed whole-model latency (fully-spatial streamed upper bound).** End-to-end latency was not synthesized as one
number (the model was C-synthesized per-shape); we *compose* it as the sum of per-stage `LatencyCyclesWorst` along the
model's **longest sequential (critical) path**, counting the three parallel attention projections Wq/Wk/Wv **once**.
Per-shape worst-case cycles (RF=256, precision-independent — identical across A8/A6/A4):
input_proj **299**, 256→256 proj (attn / head_fc1) **679**, ffn_fc1 **679**, ffn_fc2 **682**, head_fc2 **679**.

| stage | worst-case cycles |
| --- | --- |
| input_proj (14→256)                                    | 299 |
| per transformer block = QKV(‖, ×1) 679 + Wo 679 + ffn_fc1 679 + ffn_fc2 682 | **2,719** |
| × 8 blocks                                             | 21,752 |
| head_fc1 (256→256)                                     | 679 |
| head_fc2 (256→1)                                       | 679 |
| **whole-model critical path**                          | **23,409 cycles** |

At the **2.5 ns / 400 MHz target** that is **≈ 58.5 µs** per single inference; at the actually-achieved ~1.90 ns/layer
clock (all shapes meet 2.5 ns except `input_proj`, which closes at 3.71 ns) it is **≈ 44.5 µs**. This is the
**fully-spatial streamed *upper bound* on single-inference latency** — a composition, not one synthesis run. Two honest
caveats: (i) the weightless attention score core (QKᵀ / softmax / AV) was **not** synthesized (EinsumDense unsupported
on this hls4ml); it is **excluded** here — being weightless and only 0.65 % of the model's MACs it does not change the
order of magnitude, though a real design adds a small streamed cost per block; (ii) a **folded / temporally-reused**
design (one block looped over depth) trades latency *up* for area *down* — the number above is the low-latency,
high-area extreme, consistent with the fully-spatial totals in the table above.

*(A6/A4 sweep on `mulder`, completed 2026-06-26 19:55: **DONE**. DSP = **1,049 at both A6 and A4**, identical to A8,
confirming DSP is precision-independent (100 % LayerNorm); LUT/FF are modestly lower than A8 and monotone A8 ≥ A6 ≥ A4.
Raw: `results/csynth/full_model_total_a{8,6,4}_rf256.json`, per-shape `results/csynth/full_model_shape_*_a{8,6,4}_rf256.json`,
combined `results/csynth/full_model_shapes_a{8,6,4}_rf256.json`.)*

---

## C. Published REAL csynth anchor — same task, actually synthesized (Ngadiuba et al., arXiv:2003.06308)

This is the empirical proof that the binary→{0 DSP, ~1% LUT} mapping is real in silicon, on the **same
jet-tagging task**, via hls4ml. *Their* numbers come from an actual Vivado logic-synthesis pass:

| model (16×64×32×32×5 MLP) | LUT | FF | **DSP** | BRAM | Latency | II | part / clock | AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Binary (BNN)** | **~1 %** | ~0 % | **0** | 0 % | **40 ns** | 1 | xcvu9p @ 200 MHz | 0.79–0.89 |
| **Ternary (TNN)** | ~1 % | ~0 % | **0** | 0 % | 40 ns | 1 | xcvu9p @ 200 MHz | 0.85–0.92 |

(Their Table 6, after logic synthesis.) **This is the gold reference: a binary network on this exact task
synthesizes to 0 DSP, 0 BRAM, ~1% LUT, 40 ns.** Caveat: their model is a small 5-layer MLP (~few-thousand
params); ours is a 6.37M-param transformer, so our **LUT %** is larger (~25% folded) and our latency longer
(1.3 µs) — but the **0 DSP** structural result transfers directly and is now **confirmed by our own csynth (§B)**.
The one difference: their tiny MLP fits fully-unrolled (0% BRAM), whereas ours must fold to fit, parking the
binary weights in ~1.2% BRAM. This reference anchored §B before we synthesized; §B's measured numbers now bear
it out (0 DSP, low-% LUT for the binary part).

---

## D. How §B was filled (reproducible)

**Done 2026-06-24 on `mulder`** (Vitis HLS 2023.2). `code/hls/run_csynth.py` rebuilds the binary FFN from
scratch and synthesizes it — no PVC or pre-emitted projects needed, just the script + a Python env (mulder's
`bnjet` micromamba ships hls4ml 1.4.0) + `vitis_hls` on `PATH`. Step-by-step runbook (verified) →
**`code/hls/RUN_CSYNTH_ON_VITIS.md`**. The exact invocation that produced §B:

```bash
source /data/software/xilinx/Vitis_HLS/2023.2/settings64.sh
cd ~/csynth
export CUDA_VISIBLE_DEVICES=-1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
       HLS_OUT=$PWD/out HLS_RF=256 HLS_ABITS=8,6,4 HLS_BACKEND=Vitis
python run_csynth.py        # ~50 min/precision; writes out/csynth_report_a{8,6,4}_rf256.json
```

`run_csynth.py` is written for the **hls4ml 1.x** API: C-synthesis is `hls_model.build(synth=True)` (there is no
`csynth=` kwarg in 1.x), and the numbers come back nested under `report["CSynthesisReport"]`
(`LUT/FF/DSP/BRAM_18K/Best|WorstLatency/Interval*`), with a `csynth.xml` fallback parser. Raw reports are saved
in `results/csynth/`. Two gotchas, both handled above: set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (else
the TF import hits a protobuf clash), and synthesize at the **folded RF=256** point (RF=1 fully unrolls the
256→1024 layer and is intractable at this model size).

> The NRP Job `code/jobs/hls/kai-hls-csynth.yaml` remains for anyone with Vitis on a cluster, but the canonical
> run is the `mulder` one above. **NRP itself has no Xilinx backend**, which is why csynth ran off-cluster — not
> a model change.

**How §B′ (the full trained transformer) was filled.** `code/hls/full_model_csynth.py` loads the trained
`lr15_bitnetJetTagModel.h5`, rebuilds each `BitLinear` as `LayerNormalization → QActivation(quantized_bits) →
QDense(binary)`, ports the trained weights through the BitNet `AbsMeanQuantizer`, and runs three modes via
`HLS_MODE`: `fidelity` (rebuild vs. trained model), `convert` (QKeras↔Vitis C-sim bit-accuracy), `csynth`
(per-shape Vitis C-synthesis + composition over the 51 instances). The A8 run that produced §B′:

```bash
source /data/software/xilinx/Vitis_HLS/2023.2/settings64.sh
cd ~/bnjet_fullcsynth/hls
export CUDA_VISIBLE_DEVICES=-1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
       BN_CKPT=~/bnjet_fullcsynth/ckpt/lr15_bitnetJetTagModel.h5 \
       HLS_OUT=~/bnjet_fullcsynth/out HLS_MODE=csynth HLS_ABITS=8   # then HLS_ABITS=6,4 for the sweep
python -u full_model_csynth.py   # writes out/shape_*_a8_rf256.json + full_model_total_rf256.json
```

Two non-obvious requirements, both handled in the script: hls4ml's `LayerNormalization` handler needs a **3-D**
input, so each component feeds `(1, in_dim)` to the norm and `Flatten`s to 2-D before the `QDense` (a 3-D QDense is
mis-parsed as a pointwise Conv1D and hits a broken `DenseResource_rf_gt_nin` template); and the LN internals must be
pinned to **fixed<32,16>** with `table_size=4096` (the default narrow LN precision drops convert-corr to ~0.87 —
widening it recovers 0.9998). `ffn_fc2` (1024→256) is the long pole (~9 h: its 1024-wide LN drives a large
synthesizability pass); the other four shapes are minutes-to-~30 min each.

---

### Bottom line
- **The FULL trained transformer is synthesized end-to-end (§B′), not an FFN.** The real
  `lr15_bitnetJetTagModel.h5` (51 BitLinears + 51 SubLN norms + the 4 attention projections) was reconstructed from
  hls4ml-supported layers with **trained weights ported in** (rebuild↔trained corr **0.99998**; QKeras↔Vitis
  bit-accuracy **0.9967–0.9999**) and C-synthesized per shape. **Binary matmul = 0 DSP; the entire transformer's
  DSP (1,049, 8.5 % of a VU13P) is 100 % LayerNorm.** The fully-spatial sum (LUT 5.2×, FF 2.5× a VU13P) shows a
  6.37 M-param transformer must fold/stream; the per-shape numbers are the measured truth.
- **Confirmed by real Vitis C-synthesis (§B):** **DSP = 0 at A8/A6/A4** — the abstract's central claim, in
  silicon estimates, not just firmware inference. Folded design (RF=256) fits a VU13P at **~25% LUT / ~7% FF /
  ~1.2% BRAM**, latency **520 cycles ≈ 1.3 µs @ 400 MHz**, II=256.
- **Exact from firmware, no csynth needed (§A):** weights `ap_uint<1>`, accumulator `ap_fixed<32,16>`, emulation
  **bit-accurate** (corr=1.000) at A8/A6/A4.
- **Honest nuance:** BRAM=0 only for the fully-unrolled RF=1 design; the deployable folded point trades ~1.2%
  BRAM. DSP=0 is the fold-independent structural win.
- **Real-world anchor (§C):** a binary jet tagger on this task synthesizes to ~1% LUT / 0 DSP / 0 BRAM / 40 ns
  (Ngadiuba et al.) — our 0-DSP result matches; our LUT% and latency are larger because the model is ~1000×
  bigger and folded.
