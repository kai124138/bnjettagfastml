# EBOPs — the synthesis-free hardware-cost axis (era 2)

*Written 2026-07-02. Every number in §1–§2 is the direct output of
`code/training/ebops.py` (closed-form, stdlib-only; the script asserts its own
param accounting against the verified round-5 preflight count). Literature
numbers in §3 are quoted from `.claude/memory/research-log.md` (2026-07-02
entry) with their original sources; see the confidence flags there.*

## 0. Definition

**EBOPs** (Effective Bit-Operations; HGQ, arXiv:2405.00645) = Σ over every
scalar multiply of `b_i · b_j`. No accumulator term (that is the HGQ definition,
not an omission — the older UNIQ-style BOPs, arXiv:1804.10969, adds
`b_w + b_a + log2(fan_in)` per MAC and is NOT what we report). Weight×activation
matmuls cost `b_w · b_a` per MAC; the attention score matmuls (Q·Kᵀ, A·V) put
activations on both operands and cost `b_a · b_a` — binarization does not help
attention, only the linear layers.

## 1. The model's MAC profile (era 2: 16 feat × 10 particles, 5-class head)

`python code/training/ebops.py` (defaults: `--size large --era 2`):

- Params **6,375,173** — matches the round-5 CPU-preflight count exactly
  (in-code assertion; the era-1 profile still reproduces 6,373,633).
- MACs/jet **63,431,936**, split: weight×act matmul **63,022,336 (99.35 %)**,
  attention act×act **409,600 (0.65 %)**.

## 2. EBOPs per quantization (large, era 2)

| precision | b_w | b_a | total EBOPs | vs FP32(ref) | vs W8A8 |
| --- | --- | --- | --- | --- | --- |
| FP32 (reference scale only — FP multiply is not a true bit-op) | 32 | 32 | 64,954,302,464 | 1.000× | 16.000× |
| W8A8 | 8 | 8 | 4,059,643,904 | 0.062× | 1.000× |
| **W1A8 (the model)** | 1 | 8 | **530,393,088** | 0.008× | **0.131×** |
| W1A6 | 1 | 6 | 392,879,616 | 0.006× | 0.097× |
| W1A4 | 1 | 4 | 258,642,944 | 0.004× | 0.064× |

**Headline: going binary buys 7.65× fewer EBOPs than W8A8** (not a clean 8×
because the 0.65 % of MACs in attention scale as b_a², untouched by weight
binarization). Equal-EBOPs framing: at the W8A8 budget, a binary model could
carry ~7.65× more MACs. Dropping activations A8→A4 buys a further 2.05×.

HGQ's empirical bridge (EBOPs ≈ LUT + 55·DSP) holds for **fully-unrolled**
designs only; our deployed point is folded (RF=256), so EBOPs is used here as a
relative, architecture-level cost metric — the measured csynth numbers in
`hls_resource_table.md` remain the deployable-resource truth.

## 3. Where we sit vs published taggers on the same dataset

Published points (research log 2026-07-02; metrics/inputs differ — noted):

| system | input | precision | cost axis (as published) | accuracy axis (as published) |
| --- | --- | --- | --- | --- |
| hls4ml/HGQ FCNN line (HGQ-1…HGQ-6, arXiv:2405.00645, XCVU9P) | 16 jet-level feats | mixed (HGQ-learned) | LUT 0.02–0.53 %, DSP 0–0.5 %, 10–30 ns | acc 71.0–76.4 % |
| Sub-µs transformers (arXiv:2510.24784, XCU250) | ≤64 particles × 3 feats | HGQ-trained, **EBOPs target 350,000** | LUT 47k–202k, DSP 0, 44–78 ns | acc 77.9–79.8 % |
| JEDI-net (arXiv:1908.05318) | constituents, interaction net | FP32 (no FPGA impl) | ~34k params | per-class AUC 0.93–0.97 |
| **BNJetTag W1A8 large (ours)** | 10 particles × 16 feats | binary weights, A8 | **EBOPs 530.4M**, folded FFN ≈25 % VU13P LUT, DSP=0 core | **macro-OvR AUC 0.8501** (seed-avg, ROC-test, verified 2026-07-03) |

*(Era-2 accuracy landed 2026-07-03 — the full quantization × accuracy picture, all ROC-test
macro-OvR, seed-averaged for binary rows: FP32 0.8765 · W8A8 0.8642 · W1A8 0.8501 · W1A6
0.8307 · W1A4 0.7329. Read together with §2: the binary trade on this dataset is
**−1.41 macro-AUC pts vs W8A8 for 7.65× fewer EBOPs**; A6 pays another −1.94 pts for 1.35×;
A4's further 1.52× EBOPs saving costs a catastrophic −9.78 pts — the cliff. Source:
`roc-results/r5/roc_auc.md`, verified per experiment-log 2026-07-03.)*

Honest positioning (framing, not spin):

1. **Scale gap of ~3 orders of magnitude.** Our W1A8 EBOPs (530.4M) is ~1,500×
   the 350k EBOPs budget the sub-µs transformer paper trains to, and our params
   (6.375M) dwarf every published tagger on this dataset (JEDI-net ~34k). A
   literal overlay on their Pareto fronts puts us far right by construction.
   The defensible claim is the *ratio* story of §2 (binary buys 7.65× at equal
   architecture), not a Pareto-dominance story.
2. **No published point is apples-to-apples.** HGQ/AutoQKeras use 16 *jet-level*
   features (different input in kind); the sub-µs transformers use 3 features
   per particle vs our 16; metrics differ (accuracy % vs per-class AUC vs our
   macro-OvR AUC). Any era-2 comparison must state both mismatches.
3. **DSP=0 is not by itself novel** — every HGQ-trained model above reaches
   DSP=0 via learned mixed precision. Our differentiator is the *hard* {−1,+1}
   weight constraint at transformer scale (structural DSP-freedom at any
   activation width, §hls_resource_table.md B′), plus the A8→A6→A4 activation
   axis on a 6.4M-param transformer.
4. **Open lever:** the attention act×act term (26.2M EBOPs at A8, 5 % of the
   W1A8 total) is the piece binarization can't touch; the BinaryAttention line
   (arXiv:2603.09582) is the literature direction if we ever binarize scores.

## 4. Reproduce

```bash
python code/training/ebops.py                # era-2 large (the model)
python code/training/ebops.py --era 1        # frozen era-1 accounting
python code/training/ebops.py --size all     # all sizes, era 2
```
