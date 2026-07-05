# SESSION REPORT — what happened, and what we achieved

**Project:** BitNet-style **binary-weight `{−1,+1}` 1-bit transformer** jet tagger for the CMS Level-1 trigger,
converted to FPGA firmware via **hls4ml**.
**Run folder:** `qkeras-bitnet-run-2026-06-22`
**Date:** 2026-06-23

This is the plain-English story of the whole effort — from "the model won't learn" to "the binary core
converts to bit-accurate FPGA firmware with zero DSPs." The hard numbers are in `RESULTS.md`; the fully
annotated technical writeup is in `REPORT.md`; the note to the model's author is in `message-to-russell.md`.

---

## The goal (the abstract we're serving)

> The LHC collides protons 40 million times a second; the trigger must decide what to keep in ~microseconds on
> an FPGA. We build a jet tagger from a **binary-weight, BitNet-style 1-bit transformer** (attention + FFN
> weights constrained to **{−1,+1}**). Because binary weights turn multiply–accumulate into XNOR+popcount, the
> matmuls map onto the FPGA's **lookup-table / logic fabric instead of its scarce DSP multipliers** — the path
> to a maximally compact, low-latency trigger. We evaluate the **tagging efficiency** of the 1-bit model
> (vs the vanilla full-precision version) and explore **how aggressively it can be quantized before efficiency,
> resource consumption, and latency degrade** — with hls4ml providing the resource/latency story.

So the work had to deliver three things: (1) a **working, strong 1-bit model**, (2) **baselines + a
quantization sweep** to quantify efficiency vs aggressiveness, and (3) the **hls4ml resource/latency** evidence
for the DSP→LUT claim. All three are now done. Here's how each happened.

---

## Act 1 — Getting the model to learn at all

**Starting point.** We re-cloned the upstream `Brainz22/BNJetTag`. Its structure is completely different from
the older `BNJetTagKai`: training is `qkerasModel.py` (four positional `.h5` args), model size is set by
hardcoded constants, and there's no W&B integration.

**Blocker #1 — data keys.** `qkerasModel.py` reads HDF5 keys `"Training Data"` / `"Sample Data"`, but the data
staged on the cluster uses keys `jet_constituents` / `train_jet_data`, and the column slicing `[:,146:]` didn't
match the real data (it produced **empty arrays**). Fixed with a surgical, data-driven patch: read whichever
key exists, derive the widths from the data.

**Blocker #2 — the model wouldn't learn (the big one).** At every size the model froze at `val_auc = 0.5`
(constant output). We first suspected the weight quantizer and **ruled that out**. The real cause:
`activation_quant` had **no straight-through estimator (STE)** — `tf.round` has zero gradient everywhere, so
gradient flow was severed through every activation, and only the final classifier head could train. **Adding
the activation STE** (plus moving the weight quantizer in-tape with a latent kernel) **unfroze it: `val_auc`
jumped 0.50 → 0.67 at epoch 1.** This diagnosis + the one-line fix is written up for the model's author in
`message-to-russell.md`.

---

## Act 2 — Making the 1-bit model strong (the paper recipe)

After the STE fix, the model *learned* but plateaued around `val_auc ≈ 0.66` — even a tiny 2-layer net hit the
same wall, which told us it was a **recipe** limit, not a precision limit.

We did a **component-by-component paper-alignment audit** against arXiv:2310.11453 (the original *binary* W1A8
BitNet). The forward path largely aligned (binary `Sign`, 8-bit absmax activations, LayerNorm-before-quant, STE
on `Sign` & `Clip`, latent FP weights), but found **one genuine correctness gap**: the paper **centralizes
weights to zero-mean before the sign** — `Sign(W − mean W)` (Eq. 1, 3) — which the code wasn't doing. **Added.**
The rest was training recipe (large LR ≈ 1.5e-4, no grad clipping, Adam β=(0.9,0.98), weight decay 0.01,
warmup + polynomial decay), added as knobs.

**Result: the recipe broke the wall — binary now hits `val_auc` 0.7530** (up from 0.664). The "0.66 binary
wall" was the old recipe, not the 1-bit precision.

---

## Act 3 — The abstract's measurements (baselines + quantization sweep)

**Central comparison (1-bit vs vanilla & 8-bit).** We added a `BN_VARIANT` knob (`bitnet` | `vanilla` | `w8a8`)
so the *identical* architecture/data/recipe runs at three precisions, isolating the cost of going 1-bit. The
W&B runs were given distinguishable names (`VANILLA-FP32-…`, `BASELINE-W8A8-…`).

- vanilla FP32 → **0.7703**
- W8A8 → **0.7719** (8-bit is essentially **lossless** — +0.16 pt vs vanilla = noise)
- 1-bit binary → **0.7530** (going all the way to 1-bit costs **~1.7 pts** — the price of the DSP→LUT win)

**Quantization-aggressiveness sweep (the "how far?" question).** Weights are already 1-bit, so the dial is
**activation precision**: **A8 0.7562 → A6 0.7450 → A4 0.7381** — ~1 point per 2-bit step, **no precision
cliff**, all healthy. Activation width trades cleanly against efficiency, ready to pair with firmware savings.

**Appendix (kept, off the current abstract).** Two axes from an earlier, longer abstract were preserved but
moved out of the main story: a **ternary `{−1,0,+1}`** variant (0.7685 — an accuracy ceiling; binary reaches
~98% of it, but ternary's `0` state breaks the LUT-pure XNOR+popcount mapping, so it's off-thesis), and
**softmax-free attention** (0.7562 — deleting softmax is statistically free, and hls-friendly).

---

## Act 4 — hls4ml: the FPGA resource & latency story (this session's main work)

This is the part the most recent work delivered. The abstract's core hardware claim is *binary weights → LUT/
logic, not DSPs → compact & low-latency.* We proved it with **real hls4ml**, not just on paper.

**The constraint we worked around.** The gold-standard route is hls4ml → Vitis/Vivado **C-synthesis**, which
emits exact LUT/FF/DSP/BRAM + latency. **NRP Nautilus has no Xilinx HLS backend**, so we delivered the story two
reinforcing ways: a real-hls4ml **convertibility + bit-accuracy proof** (with the generated firmware inspected),
and an **analytical per-component** resource model validated against that firmware.

**What we ran and found:**

1. **A8/A6/A4 firmware sweep (`kai-hls-sweep`, hls4ml 0.8.1).** Pushed the dominant primitive — a binary FFN
   block — through the *real* toolchain at all three activation precisions. It **converted and emulated
   bit-accurately at every precision** (`corr = 1.000000` ×3, N=1000). Then we grepped the generated firmware
   (`kai-hls-inspect`) and confirmed, straight from `defines.h`:
   - both dense kernels type as **`ap_uint<1>`** → LUT logic, **structurally 0 DSP, at every precision** (a
     1-bit weight literally cannot drive a DSP48 multiplier input);
   - the activation datapath `layer4_t` **narrows `ap_ufixed<8,2>` → `<6,2>` → `<4,2>`** exactly along the
     quantization axis — the hardware face of the 0.756 → 0.745 → 0.738 efficiency trade;
   - `io_parallel`, `reuse_factor=1` (lowest-latency) throughout.

2. **Newer stack (`kai-hls-full`, hls4ml 1.3.0).** Phase-1's blocker was that the old TF-2.11 image pinned
   Python 3.8 → hls4ml 0.8.1, which rejected `LayerNormalization`. On 1.3.0, **LayerNorm (SubLN) now converts,
   and a full SubLN→binary-proj→SubLN→binary-FFN block converts end-to-end.** The gap **collapsed** from
   "LayerNorm *and* attention both fail" to **just the attention score matmul** — and we pinned down exactly why
   that one's hard: the attention scores are an **activation×activation** contraction, which Keras `EinsumDense`
   *cannot even express* (it only contracts an activation against a trainable kernel). So it's an **Extension-API
   custom op**, not a missing parser handler.

3. **Per-component resource model (`resource_model.py`).** Pure architecture arithmetic: **63.0M binary MACs
   (99.35% DSP-free)**, **6.36 Mbit** weights, **51 SubLNs**, only **0.65%** real multiplies (the 409,600
   attention-score products). Device-fit on a VU13P folds to **RF ≈ 583 → latency ~1 µs @ 400 MHz.**

**Bottom line of the hls4ml work:** the **binary-dense + SubLN backbone — ~90%+ of the model's arithmetic, and
exactly the layer types that worried us — is convertible today and provably DSP-free**, with the activation
datapath shrinking along the quantization axis. Three synthesizable projects (`binary_ffn_a{8,6,4}_prj`) are
emitted and ready for an off-cluster Vitis csynth.

---

## Errors hit along the way (and how they were fixed)

| symptom | root cause | fix |
| --- | --- | --- |
| model frozen at `val_auc=0.5` | `activation_quant` had no STE — `tf.round`'s zero gradient severed gradient flow | added activation STE (+ in-tape weight quantizer); 0.50 → 0.67 at epoch 1 |
| binary stuck at ~0.66 | old training recipe + missing zero-mean weight centralization | added `Sign(W−mean W)` + paper recipe (LR 1.5e-4, β₂=0.98, wd, warmup/decay); → 0.7530 |
| hls emulation diverged (`corr=0.24`) | default accumulator `fixed<16,6>` saturates at ±32; fc1 sums reach ±50+ | widen accum/result/io to `ap_fixed<32,16>`, leave `act` native |
| naïve widen reached only 0.85 | widening had also removed the `quantized_relu` `SAT` clip | restore the SAT clip; widen accum only |
| sweep Job crashed: `No module named 'pyparsing'` | qkeras 0.9.0's setup.py requires a typo'd package "pyparser", so pyparsing never installs | added `pyparsing<3` + deps to the Job's pip line |
| full Job crashed: `keras has no attribute '__version__'` | a diagnostic `print(tf.keras.__version__)` tripped TF 2.14's lazy keras loader (the real stack had imported fine) | removed `tf.keras.__version__` from the print |
| Phase-1 dev pod SIGKILLed at ~6h (exit 137) | it was a bare Pod; `cms-ml` kills bare Pods at 6h | run all hls4ml work as batch **Jobs**, never bare Pods |

---

## What's done vs. the one honest remaining gap

**Done:**
- ✅ A working, strong **1-bit binary model** (`val_auc 0.7530`), paper-aligned to arXiv:2310.11453.
- ✅ **Baselines** (vanilla 0.7703, W8A8 0.7719) and the **quantization sweep** (A8/A6/A4) — both abstract
  questions answered.
- ✅ **hls4ml proof**: binary core converts + is **bit-accurate** (corr=1.0) at A8/A6/A4; **DSP=0 is in the
  generated firmware**; LayerNorm + full binary block convert on hls4ml 1.3.0.
- ✅ Per-component resource/latency model (63.0M binary MACs / 99.35% DSP-free / 6.36 Mbit / ~1 µs device-fit).

**The one gap (clearly flagged):** exact synthesized **LUT/FF/DSP/BRAM and latency-in-cycles** need a **Vitis
C-synthesis** pass, which NRP doesn't have. This is a **tooling** gap, *not* an architecture change — the
emitted `binary_ffn_a{8,6,4}_prj` projects are bit-accurate and ready for that off-cluster handoff. The
remaining *model* firmware work is the attention act×act matmul via hls4ml's Extension API.

---

## Deliverables in this folder

- **`RESULTS.md`** — every headline number (training AUCs + full hls4ml result tables) in one place.
- **`SESSION_REPORT.md`** — this narrative.
- **`REPORT.md`** — the full annotated technical report (~990 lines), including the appendix explorations.
- **`message-to-russell.md`** — the STE bug diagnosis + one-line fix for the model's author.
- **`hls/`** — the hls4ml scripts (`sweep_precision.py`, `resource_model.py`, `full_transformer_probe.py`, …).
- **`k8s/`** — the Kubernetes Jobs that ran everything (`kai-hls-sweep`, `kai-hls-full`, `kai-hls-inspect`, …).
- W&B project **`bnjettag-bitnet`** (entity `kayamaguchi-uc-san-diego`) — live training metrics.
