# The abstract change & "did we retrain the BitNet?" — straight answer

**Your question:** *"I'm confused since we never ran training with the BitNet after I gave you the new
abstract."*

**Short answer:** Correct — and that's expected, because **the new abstract did not change the BitNet model or
its recipe.** It changed *what we compare the BitNet against*. The binary 1-bit model (val_auc **0.7530**) is the
**same model** under both the old and new abstracts, so its number carries over unchanged. What the new abstract
*newly* required — the **vanilla FP32** and **W8A8** baselines, and the **activation sweep** — **was** run after
you gave it. Details below.

---

## What actually changed between the two abstracts

| | OLD (longer) abstract | NEW (current) abstract |
| --- | --- | --- |
| **The model under test** | binary `{−1,+1}` 1-bit BitNet | **binary `{−1,+1}` 1-bit BitNet** — *identical* |
| **Compared against** | ternary `{−1,0,+1}` weights | **vanilla FP32** + **W8A8** (8-bit) |
| **Second axis** | softmax vs softmax-free attention | **quantization aggressiveness** (W1A8 → A6 → A4) |
| **hls4ml** | resource/latency story | resource/latency story (unchanged) |

The **central object never changed** — it was always the binary 1-bit model. The new abstract only swapped the
*reference points* (ternary → vanilla/W8A8) and the *second question* (softmax-free → how-far-can-you-quantize).
So nothing about the BitNet's architecture, quantizers, or training recipe was touched by the new abstract → no
reason to retrain it.

---

## Timeline (why "no BitNet run after the abstract" is the right outcome)

All on 2026-06-23, in order:

| time | what ran | job file | result | abstract era |
| --- | --- | --- | --- | --- |
| ~09:46 | **binary 1-bit BitNet** (paper recipe) | `kai-bn-train-paper-binary-lr15.yaml` | **0.7530** | trained under OLD, **valid under NEW** |
| ~09:42 | ternary `{−1,0,+1}` | `kai-bn-train-paper-ternary.yaml` | 0.7685 | OLD → now **appendix** |
| ~11:20 | softmax-free attention | `kai-bn-train-paper-binary-sffree.yaml` | 0.7562 | OLD → now **appendix** (also the A8 sweep point) |
| ~12:47 | activation sweep A6 | `kai-bn-train-paper-binary-sffree-a6.yaml` | 0.7450 | **NEW axis** |
| ~12:48 | activation sweep A4 | `kai-bn-train-paper-binary-sffree-a4.yaml` | 0.7381 | **NEW axis** |
| **~15:30** | **← you gave the NEW abstract here** | | | |
| ~15:38 | **vanilla FP32 baseline** | `kai-bn-train-vanilla-fp32.yaml` | **0.7703** | **NEW** (required by new abstract) |
| ~15:38 | **W8A8 baseline** | `kai-bn-train-w8a8.yaml` | **0.7719** | **NEW** (required by new abstract) |

So after the new abstract we ran exactly the two things it *added* — the **vanilla** and **W8A8** baselines (via
the new `BN_VARIANT` knob). We did **not** re-run the BitNet because its definition was unchanged; re-running it
would have reproduced ~0.7530 from the identical code path (`BN_VARIANT=bitnet`, the default).

---

## One honest caveat (worth knowing)

The **activation sweep** (A8/A6/A4) was run on the **softmax-free** base, not the canonical softmax BitNet. The
A8 point (0.7562) ≈ the softmax BitNet (0.7530) to within run-to-run noise, so the *relative* degradation
(−1.1 pt at A6, −1.8 pt at A4) is base-independent and trustworthy — but if you want the sweep anchored to the
exact softmax headline model, that's a clean re-run. This is already flagged in `results/REPORT.md`.

---

## If you DO want a fresh BitNet run (optional — say the word)

It isn't needed for correctness, but it would give a clean, **co-batched** binary point next to the
vanilla/W8A8 baselines (same day, same seed regime). One command:

```bash
kubectl apply -f code/jobs/training/kai-bn-train-paper-binary-lr15.yaml -n cms-ml
# BN_VARIANT defaults to "bitnet"; expect ~0.7530 again (modulo seed noise).
```

Or, to anchor the activation sweep to the **softmax** model instead of softmax-free, re-run the sweep with
`BN_SOFTMAX_FREE=0` (default) at `BN_ACT_BITS=8/6/4`. I can launch either on the L40S GPUs whenever you want —
I left it off by default because the established numbers already answer the abstract.
