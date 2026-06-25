# Message to Russell — BitNet (`qkerasModel.py`): found why it wasn't learning, and a fix that works

> Draft for Kai to send (Slack / email / GitHub). Short version.

---

Hey Russell,

I ran your BitNet (`qkerasModel.py`) on the NRP GPUs and hit a wall: it trains end-to-end but
**didn't learn at any size** — `val_auc` pinned at exactly 0.5 and the net emitted a constant logit
(val accuracy = the exact background fraction, 0.3638). I tracked down the cause and have a one-line
fix that unfreezes it — wanted to run it by you before opening a PR.

**Root cause: `activation_quant` has no straight-through estimator.**

```python
return tf.clip_by_value(tf.round(x * scale), -128, 127) / scale
```

`tf.round` has zero gradient, so with no STE this severs the gradient through *every* BitLinear input.
Only the final linear head can train; every layer upstream stays at random init → constant output →
`val_auc = 0.5`. Note your *weight* quantizer (`AbsMeanQuantizer`) already has an STE — this is the
same trick, just missing on the activation path:

```python
q = tf.clip_by_value(tf.round(x * scale), -128, 127)
return (x * scale + tf.stop_gradient(q - x * scale)) / scale
```

**Verified the fix.** Same data, same config (D32/L2/FFN64), only difference is the activation STE:

| | train_auc | val_auc | val_acc | val_loss |
|---|---|---|---|---|
| before (frozen) | 0.4996 | **0.5000** | 0.3638 | 0.890 |
| after, epoch 1  | 0.6306 | **0.6749** | 0.5654 | 0.858 |

`val_auc` moves off 0.5 immediately and the constant-output behavior is gone — gradients now flow
end-to-end. A full max-size run (D256/L8/FFN1024) is training now.

**How I ruled out the other suspect.** I first thought the problem was the *weight* quantizer being
attached as a Keras `constraint` (applied outside the gradient tape, no FP latent weights) rather than
an in-`call` STE like your docstring describes. I rewired `BitLinear` to keep a latent kernel and
quantize inside `call`, and re-ran — **still frozen**. So the constraint wasn't the cause; that test is
what pointed me upstream to `activation_quant`. (In my patched copy I kept both changes: activation STE
+ weight quantizer moved in-`call`. The activation STE is the one that actually unfreezes it.)

**Weight path — binary is correct/intended; one real alignment fix + two mislabeled docstrings.**
With the STE fix in, I did a careful pass of `qkerasModel.py` against the **original BitNet paper
(arXiv:2310.11453)** — since that's what we're targeting — and the weight quantizer is **right to be
binary**. Original BitNet *is* W1A8 (binary `{−1,+1}` weights, 8-bit activations), and binary is exactly
what we want for the trigger: `{−1,+1}` turns every MAC into **XNOR + popcount**, which maps onto **FPGA
LUT/logic instead of scarce DSPs**. So `tf.sign(...)` is the intended op — it's the `AbsMeanQuantizer`
**docstring** (which describes ternary `clip(round(W/scale),−1,1)`) that's mislabeled, not the code.

The one genuine gap I found on the weight path: the paper **centralizes weights to zero-mean before the
sign** — W̃ = `Sign(W − mean(W))` (Eq 1,3) — but the code signs `W` directly. I added the centralization
(STE unchanged); small, but it's the one real deviation from the paper's `BitLinear`.

Crucially, **the centralization + recipe fix makes the 1-bit (binary) model strong.** Once I added
`Sign(W − mean W)` and used the LR the model actually wants (**1.5e-4** — the paper's 1e-3 is *too hot*
here, it peaks at epoch 1 then decays), **binary jumped from ~0.664 to `val_auc` 0.753**. So the old "binary
wall" was a *recipe* limit, not a precision limit. Binary `{−1,+1}` is the right call for our trigger: it
keeps every MAC a pure **XNOR+popcount → LUT (no DSP, no zero-mask)**, which is exactly the hardware win we're
after. (A matched-recipe ternary run lands at 0.769 — ~1.5 pts higher — but ternary's `0` state breaks the
LUT-purity, so I keep it only as an accuracy-ceiling reference, not the deployed model. Note ternary `{−1,0,+1}`
is the *separate* "b1.58" paper, arXiv:2402.17764, not this one.)

**Our research framing changed (FYI, doesn't affect the bug).** Our abstract now centers on **1-bit vs the
vanilla (full-precision) model**, with conventional **W8A8** as a mid-point — not binary-vs-ternary. To make
that a clean apples-to-apples comparison I added a single `BN_VARIANT` knob (`bitnet` | `vanilla` | `w8a8`) so
the *identical* architecture/data/recipe runs at all three precisions. Result: **vanilla 0.770, W8A8 0.772,
1-bit 0.753** — 8-bit is essentially lossless (≈ vanilla), and going 1-bit costs only **~1.7 pts** vs full
precision, which is the price of the DSP→LUT win (only binary maps to XNOR+popcount → DSP=0). The second
question is **how aggressively we can quantize** the 1-bit model before efficiency degrades:

- **Activation precision is a smooth resource dial.** A8 → A6 → A4 = **0.756 → 0.745 → 0.738** — ~1 pt per
  step, no cliff, all healthy (val ≈ train). So we can trade activation bit-width (≈ activation BRAM + popcount
  accumulator width) against efficiency cleanly: A6 costs ~1.1 pts, A4 ~1.8 pts.
- **(Side exploration, off our current abstract.)** I also tried **softmax-free attention** —
  `ReLU(QKᵀ/√d)` with a constant `1/N` scale instead of softmax, which deletes the only non-binarizable op in
  the attention core — and it matched softmax (**0.753 → 0.756**, within noise). Promising for hls4ml but no
  longer part of our headline; I'm parking it as an appendix result.

Net: the deployed model is **binary @ 1.5e-4 (A8, softmax)**, benchmarked against the vanilla/W8A8 baselines,
with A6/A4 as the quantization-aggressiveness knob.

**hls4ml — the FPGA claim, demonstrated not just asserted.** I pushed the dominant primitive (a binary FFN
block) through *real* hls4ml across the whole **A8/A6/A4** activation axis: it converts and emulates
**bit-accurately at every precision** (corr `1.000000` ×3, N=1000), the generated firmware types **both** dense
kernels as `ap_uint<1>` at every A (binary → 1-bit LUT logic, **zero DSPs** — structural, precision-independent),
and the activation datapath narrows `ap_ufixed<8/6/4,2,SAT>` exactly along that axis — the resource/latency dial
made concrete in firmware. I also stood up the newer stack (**hls4ml 1.3.0**, Python 3.10) and closed the
Phase-1 gap: **LayerNorm (SubLN) now converts, and a full SubLN→binary-proj→SubLN→binary-FFN block converts
end-to-end**. The one piece left is the attention **act×act** score matmul (Q·Kᵀ) — and interestingly it's *not*
a Keras `EinsumDense` (that contracts against a trainable kernel, not a second activation), so it's an
Extension-API custom op rather than a version gate. Per-component model: **63.0M binary MACs (99.35% DSP-free)**,
0.65% real attention multiplies, 6.36 Mbit weights, 51 SubLNs → device-fit ~1 µs @ 400 MHz on a VU13P. Only the
final Vitis csynth (exact LUT/FF/DSP/BRAM) is off-cluster; the binary core — ~90% of the arithmetic — is proven
convertible today.

**Three questions for you:**
1. Was `activation_quant` meant to carry an STE? (I'm assuming yes — your weight path has one.) This is
   the actual bug — the rest below is alignment/labeling.
2. The `AbsMeanQuantizer` docstring describes ternary, but the code does binary `sign()` — which is
   correct for original BitNet and for our XNOR/popcount target. Mind if I **fix the docstring** to say
   binary? And OK to keep the **zero-mean centralization** `Sign(W − mean W)` (Eq 1) I added?
3. `RMSNorm.call` subtracts the mean — so it's really LayerNorm/SubLN. That actually **matches** BitNet
   (Eq 12 uses SubLN), so I think the *code* is right and only the "no centering" docstring is
   misleading. Confirm it's intended SubLN?

FWIW a separate ternary-QAT pipeline trains fine on the same staged data (`val_loss` ~0.13), so the
data was never the issue — the freeze was specific to `qkerasModel.py`'s activation path.

Happy to open a PR with just the activation STE if you'd like.

Thanks!
Kai

---

### Pointers (for Kai, not part of the message)
- W&B: https://wandb.ai/kayamaguchi-uc-san-diego/bnjettag-bitnet
  - `baseline`/`large`/`max` = original (constraint), frozen at val_auc 0.5
  - `ste-test-D32-L2-FFN64` = weight-quantizer-only fix, **also frozen** (ruled out suspect #1)
  - `fixed-baseline-D32-L2-FFN64` = activation-STE fix, **learns** (val_auc 0.50→0.67 epoch 1)
  - `fixed-max-*-lr2e4-wu3` / `…-lr5e4-bs256-…` = max binary, learns but **capped ~0.66** (underfits)
  - `fixed-max-ternary-…-lr5e4…` = max ternary, **0.711** at same recipe (binary ceiling broken)
  - `fixed-max-ternary-…-lr1p5e4-…-nowu` = ternary retuned at low LR → **0.717** (accuracy ref)
  - `paper-binary-…-lr1e3-…clipON/NOclip` = paper recipe at LR 1e-3 → **too hot** (peaks ep1, ~0.70)
  - `paper-binary-…-lr1p5e4-…` = **the 1-bit model** at corrected LR (val_auc **0.753**, deployed)
  - `paper-binary-…-lr3e4-…` = binary LR-window bracket
  - `VANILLA-FP32-D256-L8-FFN1024-lr1p5e4-baseline` = **vanilla full-precision baseline** (BN_VARIANT=vanilla) → **0.7703** (current-abstract comparison)
  - `BASELINE-W8A8-D256-L8-FFN1024-lr1p5e4` = **conventional 8-bit baseline** (BN_VARIANT=w8a8) → **0.7719** (current-abstract comparison)
  - `paper-ternary-…-lr1p5e4-…` = ternary comparison at the matched LR → **0.769** (*now appendix*)
  - `paper-binary-…-SOFTMAXFREE` = binary + softmax-free attention → **0.756** (*now appendix*; on par with softmax)
  - `paper-binary-…-SOFTMAXFREE-A6` / `…-A4` = activation-precision (quantization-aggressiveness) sweep → **0.745 / 0.738**
- On-disk logs survive on the PVC under `/data/outputs/qk-*/train_*.log`
  (`qk-fixed-baseline`, `qk-fixed-max-fast`, `qk-fixed-max-fast2`, `qk-fixed-max-ternary`,
  `qk-fixed-max-tern-lr`, `qk-paper-binary*`, `qk-paper-binary-lr15`, `qk-paper-binary-lr3`,
  `qk-paper-ternary`, `qk-paper-binary-sffree`, `qk-paper-binary-sffree-a6`, `qk-paper-binary-sffree-a4`).
- Patched code (`code/qkerasModel.py`): (1) `activation_quant` STE — the unfreeze fix; (2) weight
  quantizer STE in-`call` against a latent kernel (no constraint); (3) **zero-mean centralization
  `Sign(W − mean W)`** added to the binary path (Eq 1,3); (4) paper-recipe knobs env-gated
  (`BN_LR/WARMUP_EPOCHS/DECAY_EPOCHS/BETA2/WEIGHT_DECAY/CLIPNORM`, defaults preserve prior behavior);
  plus research-axis knobs `BN_TERNARY` (weights), `BN_SOFTMAX_FREE` (binarizable attention), and
  `BN_ACT_BITS` (activation precision, default 8 = W1A8) — all default to the original binary/softmax/A8.
  `RMSNorm.call` left as-is (it's SubLN, which matches the paper).
- The minimal upstream diff = the activation STE alone (3 lines). The weight-quantizer relocation is
  optional cleanup (matches the docstring's QAT description but isn't needed to unfreeze training).
- hls4ml: `code/hls/sweep_precision.py` (A8/A6/A4 convert+emulate+firmware-inspect, corr 1.000000 ×3),
  `code/hls/full_transformer_probe.py` (LayerNorm + EinsumDense + full block on 1.3.0), `code/hls/stage_a_fix.py`
  (the original bit-accurate binary FFN), `code/hls/convert_probe.py` (3-stage probe), `code/hls/resource_model.py`
  (per-component breakdown), `methods/hls4ml_findings.md` (citations). Jobs `kai-hls-sweep` (hls4ml 0.8.1, the
  sweep) + `kai-hls-full` (Python 3.10 / **hls4ml 1.3.0**, the newer stack). Generated projects at
  `/data/outputs/hls/binary_ffn_a{8,6,4}_prj` (+ `.tar.gz`; `defines.h` → `weight2_t/weight5_t = ap_uint<1>`,
  `layer4_t = ap_ufixed<8/6/4,2,SAT>`), logs `sweep_precision.log` / `full_transformer_probe.log`. On 1.3.0:
  LayerNorm ✅, full SubLN+binary-dense block ✅; attention act×act matmul = Extension-API work; exact
  LUT/FF/DSP/BRAM = a Vitis csynth pass off-cluster.
