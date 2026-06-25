# HLS4ML SYNTHESIZED RESOURCE & LATENCY TABLE

**The table you asked for: exact LUT / FF / DSP / BRAM + latency-in-cycles — now MEASURED.**

> **Status — DONE (2026-06-24).** The Vitis HLS 2023.2 C-synthesis was run on the group's `mulder` box and **§B
> is now filled with real synthesized numbers** (raw reports in `results/csynth/`). The headline: **DSP = 0 at
> A8/A6/A4, confirmed in synthesis** (not just firmware inference); the folded device-fit design (RF=256) fits a
> VU13P at **~25% LUT / ~7% FF / ~1.2% BRAM / 0 DSP**, latency 520 cycles (1.3 µs @ 400 MHz).
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

---

### Bottom line
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
