# BNJetTag — The Research (living document)

**The single source of truth for what this project claims, what it has measured, and what
is still open.** Updated only from verified numbers (recomputed from `.npz` / csynth
reports / logs — sources in [§8](#8-where-every-number-comes-from)). Layout and workflow:
[00-START-HERE.md](00-START-HERE.md).

*Last verified update: 2026-07-03.*

**Two reading rules, always in force**
1. **Validation AUC ≠ ROC-test AUC.** Val AUC is the training-time monitor; ROC-test AUC is
   measured on a held-out set. Every number below is labeled.
2. **Two dataset eras, not comparable.** Era 1 (before 2026-07-01): private 2-class dataset.
   Era 2 (round-5 onward): public HLS4ML LHC Jet dataset, 5-class. Era-1 numbers are frozen
   history; era-2 starts its own table.

---

## 1. Thesis and abstract

**Thesis.** A transformer jet tagger whose attention and feed-forward weights are constrained
to binary `{−1,+1}` (BitNet-style, 1-bit) can run in the CMS Level-1 trigger's FPGAs using
essentially **no DSP multipliers** — the binary multiply–accumulate reduces to sign-flips and
LUT adder trees — at a modest, measurable cost in tagging efficiency. Two axes are measured:
**(a)** tagging efficiency vs full-precision and conventional 8-bit baselines, and **(b)** how
aggressively activations can be quantized (A8 → A6 → A4) before efficiency, resources, or
latency degrade.

**Abstract (as submitted).**
> The LHC collides protons at a rate of 40 million collisions per second. To filter the
> massive amount of data for interesting physics, the real-time trigger systems inside
> detectors at the LHC necessitate smart and sophisticated triggers that are 1) efficient
> enough to simultaneously reject large backgrounds and keep enough signal, 2) compact enough
> to meet hardware constraints, and 3) fast enough to meet the latency requirements. We
> present a particle jet tagger based on a binary-weight, BitNet-style 1-bit transformer, in
> which the attention and feed-forward weights are constrained to {−1, +1}. Because binary
> weights reduce multiply–accumulate operations, the model's matrix multiplications can be
> mapped onto the FPGA's lookup-table and logic fabric rather than its scarce DSP resources,
> targeting a maximally compact and low-latency trigger. The model is trained and validated
> on data simulating the conditions of the LHC and converted into high-level synthesis using
> the hls4ml hardware-software codesign tool. This work evaluates the tagging efficiency of a
> 1-bit transformer model — comparing it against the vanilla version, and explores how
> aggressively it can be quantized before its tagging efficiency, resource consumption, and
> latency degrades.

---

## 2. Where things stand right now (2026-07-04)

- **The model is rebuilt on the hls4ml-NATIVE quantization stack (HGQ2) and verified
  (2026-07-04, results-analyst ✓).** The round-5 large checkpoints port into an HGQ2
  (Keras-3) model with weights **pinned to binary {−1,+1}** (all 51 BitLinears, zero
  sign-zeros) and *static* hardware-implementable activation quantizers replacing the
  training-time dynamic per-token scales. Recovered era-2 ROC-test macro-OvR AUC on the
  full 260k val split: **W1A8 0.8493 (trained: 0.8551, Δ −0.0058) · W1A6 0.8222
  (Δ −0.0173) · W1A4 0.7115 (Δ −0.0231)** — the honest cost of hardware-static
  quantization, per §6′. Native HGQ2 EBOPs: 650.4M / 501.2M / 352.2M (different
  accumulator convention than §3's 530.4M — never mix). The previously-unsynthesizable
  **attention score core now converts** through hls4ml's native QEinsum/QSoftmax path
  (C-sim corr 1.0); first HGQ2-path csynth results in §6′. Pipeline, ledger, constraints
  map: `code/hgq2/`, `results/hgq2/`.
- **Round-5 is DONE end-to-end: trained, ROC-tested, verified (2026-07-03).** All 8 jobs
  finished on the new dataset; the era-2 accuracy table now exists (§5′): **W1A8 seed-avg
  macro-OvR 0.8501 vs FP32 0.8765 / W8A8 0.8642; activation cliff at A4 (0.7329)**.
  Ops note: the kai-data PVC became unmountable cluster-wide on 2026-07-02 (rook-ceph
  incident, unresolved as of this update) — the ROC ran **PVC-free** (`kai-roc-r5-pvcfree`,
  checkpoints from W&B, data from Zenodo); `.npz` durable copies live in W&B run
  `r5-roc-pvcfree-artifacts` and locally in `roc-results/r5/`.
- **The EBOPs cost axis is computed and verified (2026-07-02).** Era-2 large: **W1A8
  530.4M EBOPs vs W8A8 4.06G — 7.65× lower** (see §3 and `results/ebops.md`). Literature
  comparison against published taggers on the same dataset: same file, §3, with caveats.
- **Reporting rule for round-5:** a number becomes a headline only when it is
  **seed-averaged** and has a **ROC-test AUC**; each baseline is the *stronger* of its
  original-recipe and 5e-5 runs (baselines may only get stronger, so any surviving
  binary-vs-FP32 gap is real).
- The architecture is **fixed at `large`** (D256 / 8 layers / 8 heads / FFN1024); round-5
  varies *quantization only*.
- The hardware side is ahead of the accuracy side: DSP=0 for the binary core and the full-model
  synthesis are **done and measured** (§6) and are structural — they do not depend on the
  dataset change.
- W&B: project `bnjettag-bitnet` (runs `r5-*`); API key rotated 2026-07-01.
- **What happens next, with checklists and success criteria: [`ROADMAP.md`](ROADMAP.md).**

---

## 3. The model

BitNet-style transformer classifier (QKeras/TensorFlow implementation:
`qkeras-bitnet-run-2026-06-22/code/training/qkerasModel.py`):

- **BitLinear layers** — weights binarized to `{−1,+1}` by an absmean quantizer
  (Sign(W−α)·β), trained with the straight-through estimator; **SubLN** normalization;
  full-precision shadow weights during training. 51 BitLinears + 51 SubLN norms + 4
  attention projections in the `large` config.
- **Activations** quantized to `BN_ACT_BITS` ∈ {8, 6, 4} (absmax).
- **Fixed architecture `large`:** D=256, L=8, H=8, FFN=1024. Params: 6,373,633 (era 1) /
  **6,375,173** (era 2: 16-feature input, 5-class head). Mean-pool over particles
  (permutation-invariant), small head → logits.
- **Variants via env knobs, not code forks:** `BN_VARIANT` (bitnet / vanilla / w8a8),
  `BN_ACT_BITS` (8/6/4), `BN_N_PART` (input length), `BN_TERNARY`, `BN_SOFTMAX_FREE`.
- **Recipe:** Adam (β₂=0.98), warmup + poly-decay, grad-clip 1.0, batch 256, early-stop on
  val AUC. Round-2→4 finding: the original peak LR 1.5e-4 ("lr15") was **too hot**; peak LR
  **5e-5** is the tuned recipe (see §5, "LR finding").
- **Hardware-cost metric:** EBOPs (Effective Bit-Operations, HGQ / arXiv:2405.00645) as the
  synthesis-free cost axis (`code/training/ebops.py`, era-aware since 2026-07-02), alongside
  measured csynth numbers. **Verified era-2 values (large):** 63.43M MACs/jet (99.35 %
  weight×act, 0.65 % attention act×act); EBOPs = 4.06G (W8A8) / **530.4M (W1A8, 7.65× below
  W8A8)** / 392.9M (W1A6) / 258.6M (W1A4). The attention act×act term is the piece weight
  binarization cannot touch (4.94 % of the W1A8 total). Full table, published-tagger
  comparison, and caveats: `qkeras-bitnet-run-2026-06-22/results/ebops.md`.

---

## 4. The data — two eras

| | **Era 1 (frozen)** — runs ≤ round-4 | **Era 2 (current)** — round-5 onward |
| --- | --- | --- |
| Dataset | Private 2-file set (merged trainPart/Jet + QCD background) | **Public HLS4ML LHC Jet (150 particles)** — Zenodo; arXiv:1804.06913 + 1908.05318; on PVC at `/data/hls4ml_lhc_jet/` |
| Task | Binary signal-vs-background | **5-class**: g / q / W / Z / t |
| Input | 10×14 constituents | **Top-10 constituents by pT × all 16 features = 160 inputs** (input-size study deferred; `BN_N_PART` knob) |
| Weighting | Flat-pT reweighted | None (balanced classes; matches published baselines) |
| Val AUC meaning | Binary AUC on a validation split | **Macro one-vs-rest AUC** (5 classes), `validation_split=0.20` of train/ |
| ROC-test set | Fixed-seed tail workaround, n = 222,912 | The dataset's own `val/` split — a true never-seen held-out set |

Migration rationale, and the full list of consequences (including that **no era-1 number may
be compared to an era-2 number**): `.claude/memory/decisions.md`, entries of 2026-07-01.
The tuned LR 5e-5 is *carried over as an assumption* to era 2, not re-derived — flagged for a
spot-check if round-5 training looks unstable.

---

## 5. Tagging efficiency — Era-1 results (frozen, private 2-class dataset)

**Validation AUC** — the reproducible lr15 anchor table (all single-run, recipe lr15):

| model | val AUC | note |
| --- | --- | --- |
| vanilla FP32 | 0.7703 | full-precision baseline |
| W8A8 | 0.7719 | conventional 8-bit baseline |
| **W1A8 BitNet (the model)** | **0.7530** | binary weights, 8-bit activations |
| A8 / A6 / A4 sweep | 0.7524 / 0.7507 / 0.7437 | activation-quantization axis |
| ternary {−1,0,+1} | 0.7685 | appendix result |
| softmax-free attention | 0.7562 | appendix result |

**ROC-test AUC** (held-out, n = 222,912):

| model | ROC-test AUC |
| --- | --- |
| FP32 vanilla | 0.8207 |
| W8A8 | 0.8283 |
| A8 binary | 0.7986 |
| A6 binary | 0.7984 |
| A4 binary | 0.7886 |

**What era 1 established:**

- **Cost of going 1-bit ≈ −1.7 val-AUC points** vs FP32 (0.7530 vs 0.7703) under the anchor
  recipe — the trade bought a structurally DSP-free core (§6).
- **The LR finding (round-4):** lr15 was tuned for the large model and still too hot — every
  smaller variant peaked at epoch 2–6 then collapsed. At peak LR 5e-5 the same W1A8 model
  reached **val AUC 0.7672** (+0.0142, single run, unverified by seed-average). This is *why*
  round-5 re-trains everything, including baselines, at the tuned LR: quoting a tuned model
  against untuned baselines is the trap the project caught itself in — twice (it also made
  "medium beats large" a false conclusion in the size sweep).
- **Size sweep (single-run, best-transient, lr15 — lower bounds):** tiny 26.5K params →
  0.7112 · small 154K → 0.7335 · medium 808K → **0.7538** · large 6.37M → 0.7530. Medium
  matched large at ~8× fewer parameters, but under the collapsing recipe; the sweep is
  **closed** and the architecture fixed at `large` — quantization, not shape, is the thesis axis.
- A4 degrades ~1 AUC point below A8/A6 on ROC-test (0.7886 vs ~0.7985) — the first sign of
  the "how far can you push it" cliff. Where it lands on era-2 data is a round-5 question.

---

## 5′. Tagging efficiency — Era-2 results (current, public HLS4ML LHC Jet 5-class)

**ROC-test macro one-vs-rest AUC** on the dataset's own held-out `val/` split
(**n = 260,000**; era-2 canon — not comparable to era-1's n = 222,912). All models trained
at the tuned peak LR 5e-5, architecture `large`, input 10×16. Verified 2026-07-03 by
recomputation from `roc-results/r5/*.npz` (exact match with the job table, 48/48 numbers).

| model | macro-OvR AUC | status | per-class range |
| --- | --- | --- | --- |
| FP32 vanilla | **0.8765** | single-run baseline | 0.8449 (g) – 0.9154 (t) |
| W8A8 | **0.8642** | single-run baseline | 0.8310 (g) – 0.9052 (t) |
| **W1A8 BitNet (the model)** | **0.8501** | **seed-averaged** (0.8551 / 0.8452) | 0.8106 (g) – 0.8934 (t)* |
| W1A6 | 0.8307 | seed-averaged (0.8394 / 0.8220) | 0.7919 (g) – 0.8880 (t)* |
| W1A4 | 0.7329 | seed-averaged (0.7346 / 0.7312) | 0.6926 (g) – 0.8393 (t)* |

*seed-averaged per-class extremes; full per-seed per-class table: `roc-results/r5/roc_auc.md`.

**What era 2 establishes (all ROC-test, this table only):**

- **Cost of going 1-bit ≈ −2.64 macro-AUC points vs FP32** (0.8501 vs 0.8765), or −1.41 vs
  the W8A8 baseline — bought at 7.65× fewer EBOPs than W8A8 (§3) and the structurally
  DSP-free binary core (§6).
- **The activation cliff is at A4:** A8→A6 costs −1.94 points; A6→A4 costs **−9.78 points**.
  Era-1's "A4 holds up surprisingly well" did **not** survive the dataset migration — on the
  richer 5-class task, 4-bit activations break the model.
- Top (t) is the most robust class at every precision; gluon (g) is always the weakest.
- Baseline caveat: era-2 baselines are single-run lr05 (no era-2 "original recipe" runs
  exist, so the round-5 "stronger baseline" rule is trivially satisfied); binary rows are
  proper seed-pairs.
- Provenance note: evaluated by `kai-roc-r5-pvcfree` (PVC-free during the 2026-07-02 ceph
  incident) — same embedded `make_roc.py` ConfigMap and same trained checkpoints
  (best-epoch, `restore_best_weights=True`), pulled from W&B; eval data pulled from the
  public Zenodo record 3602260.

---

## 6. Hardware — measured synthesis results (structural; era-independent)

Real Vitis HLS 2023.2 C-synthesis on `mulder`, target device **VU13P**
(`xcvu13p-flga2577-2-e`), clock target 2.5 ns (400 MHz). Raw reports:
`qkeras-bitnet-run-2026-06-22/results/csynth/`.

- **The central claim, confirmed: the binary matmul core uses 0 DSPs.** Structural, not
  empirical: 1-bit weights emit as `ap_uint<1>`, which cannot drive a DSP multiplier port —
  the MAC becomes sign-select + LUT adder trees. Confirmed in generated firmware *and* in
  C-synthesis (2026-06-24), at every activation width (A8/A6/A4).
- **Folded, device-fit binary FFN block** (256→1024→256, reuse factor RF=256, II=256):
  ≈ **25% LUT / 7% FF / 1.2% BRAM / 0 DSP** of a VU13P; **latency 520 cycles ≈ 1.3 µs @
  400 MHz** (Vitis estimated Fmax ≈ 568 MHz, so timing met with margin).
- **The full trained transformer, end-to-end (2026-06-26):** the actual trained checkpoint —
  all 51 BitLinears, 51 SubLN norms, 4 attention projections — reconstructed from
  hls4ml-supported primitives with trained binary weights ported in, validated
  (rebuild↔trained fidelity corr 0.99998; QKeras↔Vitis C-sim bit-accuracy 0.9967–0.9999),
  then C-synthesized per distinct layer shape. **The only DSPs in the whole model — 1,049
  (8.5% of a VU13P) — sit in the LayerNorms**, not in any binary matmul.
- **A6/A4 full-model backfill (synthesized 2026-06-26, integrated 2026-07-02): the DSP
  count is precision-independent.** DSP = **1,049 at A8, A6, and A4 alike** — 100%
  LayerNorm at every activation width, because the binary matmuls never touch a DSP at any
  precision. LUT/FF shrink monotonically (A8 ≥ A6 ≥ A4), so lower activation width buys
  fabric, not DSPs. Eliminating LayerNorm DSPs (§7 Q5) is therefore *the* remaining step to
  a fully 0-DSP model at any precision.
- **Composed whole-model latency, folded RF=256 operating point (upper bound):** summing
  per-shape worst-case csynth latencies along the critical path (input_proj 299 + 8 ×
  block(2,719) + head 679 …) gives **23,409 cycles ≈ 58.5 µs at the 2.5 ns / 400 MHz
  target**, or ≈ 44.5 µs at the actually-achieved ~1.90 ns per-layer clock (all shapes meet
  2.5 ns except input_proj at 3.71 ns). Two caveats keep this an *upper bound, not a
  quotable L1 latency*: (i) the attention score core (QKᵀ/softmax/AV; EinsumDense
  unsupported in hls4ml, 0.65% of MACs) is excluded; (ii) RF=256 is the resource-frugal
  extreme — a spatially unrolled / streamed design trades fabric for far fewer cycles.
  The L1-relevant operating point remains the folded-vs-unfolded trade to be chosen (§7).
  Source: `results/hls_resource_table.md` §B′.

---

## 6′. Hardware — the HGQ2 / hls4ml-native path (2026-07-04)

The model was rebuilt on **HGQ2 0.1.9 + hls4ml 1.3.0** (the stack that mainlined
transformer/MHA support): binary weights pinned via a frozen 1-bit KBI quantizer, static
per-channel-calibrated activation grids, the parameter-free SubLN as a **custom hls4ml
extension** (new IR layer + range-reduced inverse-sqrt kernel valid for any input
variance — the stock LayerNorm path is unconvertible/limited to var ≤ 1), BitNet β scales
handled fold-aware (exact folds into the softmax exp-LUT / next-LN / bias where scale
invariance allows; CSD-2 constants in frozen affines at the 18 residual-contributor
sites). Pipeline is config-driven (`code/hgq2/`, results keyed by config hash in
`results/hgq2/`); every claim below was re-verified by the results-analyst gate
(experiment-log 2026-07-04 ✓).

**Efficiency recovered by the hardware-faithful rebuild** (era-2 ROC-test macro-OvR,
n=260,000, vs the verified trained scores; the Δ is the total cost of static
quantization + β snapping, measured, not assumed):

| model | trained (§5′ s1) | HGQ2 rebuild | Δ | EBOPs (HGQ2-native) |
|---|---|---|---|---|
| W1A8 | 0.8551 | **0.8493** | −0.0058 | 650.4M |
| W1A6 | 0.8394 | **0.8222** | −0.0173 | 501.2M |
| W1A4 | 0.7346 | **0.7115** | −0.0231 | 352.2M |

ROC overlays (log-FPR, per class, rebuild vs trained vs FP32): `results/hgq2/roc_hgq2_overlay.png`.
A4's larger loss is the static-vs-dynamic activation substitution itself (its dynamic-mode
correlation is 0.987) — a QAT run with static quantizers (native HGQ2 training) is the
identified fix, not more calibration.

**First HGQ2-path synthesis (Vitis HLS 2023.2, VU13P, real csynth on mulder):**

- **SubLN (the custom norm), dim 256, fully parallel II=1:** 165,695 LUT / 151,297 FF /
  **1,792 DSP** / 0 BRAM / 36 cycles @ est. 1.82 ns. Folded inside a dense probe it drops
  to **14 DSP** and ≈170k LUT — the norm remains the model's DSP-and-LUT consumer, now
  measured at both operating extremes (internal widths deliberately generous; narrowing
  them is the known next lever, same conclusion as §6/§7-Q5).
- **A DSP regression was caught, root-caused, and the fix synthesis-verified** (the
  reason per-instance breakdowns are mandatory): with `Strategy: Resource` (RF-folded),
  hls4ml stores weights in BRAM/ROM as *runtime operands*, so the Vitis "≤2-signed-digit
  constants are DSP-free" rule does not apply — an in-weight ±β̃ scale synthesized to
  **256 real DSPs** on one 256×256 binary matmul. Fix: keep the datapath **pure ±1** and
  carry β̃+bias in a frozen affine; with `Strategy: Latency` (weights inlined as
  constants) the fix is **confirmed in real csynth** (head_fc2 probe, RF=32): binary
  dense **0 DSP**, CSD-2 affine **0 DSP** (285 LUT), every DSP in the probe (112) inside
  the SubLN — the QKeras-path DSP-0-core result reproduced on the HGQ2-native path, with
  its enabling conditions now explicit (`results/hgq2/constraints_map.md`). The QKeras
  path never hit the trap because it emits literal 1-bit `ap_uint<1>` weights; the
  HGQ2/keras-v3 frontend has no binary special-case.
- **The attention score core (QKᵀ → softmax → attn·V) — excluded from every previous
  synthesis — is now SYNTHESIZED** (QEinsum + table-based QSoftmax with the score scale
  folded into the exp LUT for free; C-sim corr 1.0). At the fully-spatial extreme
  (large-model core, T=10/H=8/E=32, 16-bit grids, II=1): **31 cycles @ est. 1.81 ns**,
  but 4.27M LUT / **52,000 DSP** — every one of the ~51k act×act products is a real
  multiplier, because there are no weights to binarize. Per-module (store-backed): each
  act×act einsum (QKᵀ and attn·V, 10×10×8×32 = 25,600 MACs) is **25,600 DSP — exactly
  1 DSP per MAC** — while the softmax module itself is only 10 DSP; scaled-multiplies,
  not the nonlinearity, are the entire cost. This *quantifies* the §3/ebops
  statement that attention is the piece weight-binarization cannot touch: 0.65% of the
  model's MACs, disproportionately expensive per MAC, and necessarily folded in any real
  deployment. **The RF=64 folded variant is also now synthesized**: 193 cycles @ II=64
  (0.48 µs at the 2.5 ns target, est. 2.01 ns), **DSP 820** (each einsum 400 = 25,600/64,
  softmax 10) — but LUT only drops 1.7× to 2.57M (148% VU13P), ~1.2–1.35M per einsum:
  folding converts multiplier cost into operand-routing muxes, so the large-model core
  alone still exceeds the device even folded. Deploying attention at this d_model needs
  either the 8×-smaller small-model core or io_stream-style reuse, not just RF.
- **Big-shape Latency-strategy synthesis is a hard wall (negative result, 3× confirmed):**
  Vitis 2023.2 crashes (`HLS 200-1715`) after ~4 h elaborating a fully-unrolled 65k-MAC
  dense at A8, A6, and A4 alike. Deployable big shapes stay on Resource strategy — where
  HGQ2-path DSP-0 needs true 1-bit weight-type emission (bounded future work), or one
  cites the QKeras-path per-shape numbers that already prove DSP-0 there.

The layer-by-layer support matrix for HGQ2+hls4ml (what converts clean, what needs the
custom-layer recipe, where DSPs hide) lives in **`results/hgq2/constraints_map.md`** —
future architectures should be designed against it.

Provenance note (2026-07-05): every per-function split quoted above is now re-derivable
from raw store data — `results/hgq2/runs/b224a8ea/<probe>/csynth.xml` (the Vitis
per-Module report, fetched from mulder) with the parsed table in `csynth_modules.json`
beside it (`code/hgq2/parse_csynth_modules.py`). Previously these splits were quoted
only from the ledger. The era-2 large head count is checkpoint-confirmed: `"n_heads": 8`
in the `model_config` h5 attribute of all three r5 checkpoints.

---

## 7. Knowledge ledger

**Established (verified, safe to state):**
- Binary-core DSP = 0, in firmware and in real synthesis; the model's only DSPs live in
  LayerNorms (1,049 = 8.5% VU13P).
- **DSP count is precision-independent: 1,049 at A8, A6, and A4** (A6/A4 full-model
  backfill, 2026-07-02); LUT/FF monotone A8 ≥ A6 ≥ A4.
- Folded binary FFN fits comfortably on a VU13P (≈25% LUT) at 1.3 µs.
- Composed whole-model latency at the folded RF=256 point: 23,409 cycles ≈ 58.5 µs
  @ 400 MHz target (≈ 44.5 µs at achieved clock) — an *upper bound* (attention score core
  excluded; folded extreme), not an L1 latency claim.
- Era-1 accuracy tables of §5, as labeled (anchor recipe, single-run).
- The lr15 recipe over-heats smaller/binary models; 5e-5 substantially improves W1A8
  (+0.0142 val, single run).
- The QKeras→hls4ml→Vitis path is bit-faithful (C-sim corr ≥ 0.9967); re-verified end-to-end
  on mulder 2026-07-02 (smoke csynth: binary dense DSP=0; toolchain gotcha — hls4ml 1.4.0.dev
  needs full-Vitis `vitis-run` on PATH — documented in `code/hls/RUN_CSYNTH_ON_VITIS.md`).
- **Era-2 EBOPs (verified 2026-07-02):** W1A8 large = 530.4M = 7.65× below W8A8 (4.06G);
  A8→A4 buys a further 2.05×. Context vs published taggers on the same dataset: our EBOPs
  is ~1,500× the 350k budget of the sub-µs FPGA transformers (arXiv:2510.24784) — the claim
  is the equal-architecture ratio, not Pareto dominance (`results/ebops.md` §3).

- **Era-2 accuracy (2026-07-03, §5′): W1A8 costs −2.64 macro-AUC pts vs FP32 (−1.41 vs
  W8A8), seed-averaged ROC-test; the activation cliff is at A4 (−9.78 pts below A6).**

**Provisional (measured once, awaiting seed-average / ROC-test):**
- W1A8 @ 5e-5 val 0.7672 (single run, era 1) — historical; superseded in practice by §5′.
- Era-2 FP32/W8A8 baselines are single-run (see §5′ caveat).

**Open questions (the current frontier):**
1. ~~Round-5 / era-2 quantization story~~ — **answered 2026-07-03, §5′.**
2. ~~Where the activation cliff is on era-2 data~~ — **answered: A4** (A6 −1.94 pts from A8,
   A4 −9.78 pts from A6). Whether anything between 4 and 6 bits (A5?) softens it is open.
3. Input-size axis (`BN_N_PART`): accuracy vs latency/resources as constituents vary — deferred by design.
4. Full-model latency at an *L1-relevant* operating point. The folded RF=256 upper bound is
   now composed (§6: 23,409 cycles); what remains is (a) synthesizing the attention score
   core once hls4ml supports it (or hand-mapping it), and (b) picking and synthesizing a
   lower-RF / streamed point that fits the L1 budget.
5. Whether LayerNorm DSPs can be eliminated (integer/shift LayerNorm, or the softmax-free +
   norm-light direction) → would make the *entire* model 0-DSP **at every activation width**
   (§6: all 1,049 DSPs are LayerNorm at A8/A6/A4 alike). Two-axis plan logged in
   `decisions.md` 2026-07-02; gated on round-5.
6. LR 5e-5 validity on era-2 data (carried over, not re-derived).

---

## 8. Where every number comes from

| Claim | Source (relative to `qkeras-bitnet-run-2026-06-22/`) |
| --- | --- |
| Era-1 val AUC table | `README.md` headline table |
| Era-1 ROC-test AUCs | `roc-results/roc_auc.md`; recomputable from `roc-results/*.npz` |
| **Era-2 ROC-test AUCs (§5′)** | `roc-results/r5/roc_auc.md`; recomputable from `roc-results/r5/*.npz` (verified 2026-07-03, 48/48 exact); durable copy: W&B run `r5-roc-pvcfree-artifacts` |
| LR finding, 0.7672@5e-5 | round-4 PVC logs; `README.md` round-4 blockquote |
| Size/ablation sweep | `results/variant_sweep.md` |
| DSP=0 + FFN csynth (RF=256) | `results/hls_resource_table.md` §A–B; raw `results/csynth/*.json` |
| Full-model synthesis, 1,049 DSP | `results/hls_resource_table.md` §B′ |
| A6/A4 DSP=1,049; composed 23,409-cycle latency | `results/hls_resource_table.md` §B′; raw `results/csynth/full_model_*_a{4,6}_rf256.json` (commit 52cb818) |
| Param counts | round-4/round-5 preflight output (logged in experiment log) |
| Round-5 status | `.claude/memory/experiment-log.md` 2026-07-01/07-02 entries |
| Dataset migration + rules | `.claude/memory/decisions.md` 2026-07-01 entries |
| Era-2 EBOPs table + comparison | `results/ebops.md`; recomputable via `code/training/ebops.py` (verified 2026-07-02) |
| Published-tagger numbers (HGQ, sub-µs transformers, JEDI-net) | `.claude/memory/research-log.md` 2026-07-02 entry (with per-number confidence flags) |
| mulder toolchain re-verification | experiment-log 2026-07-02; smoke report `mulder:~/bnjet_smoketest/prj/.../csynth.xml` |
| **HGQ2 rebuild AUCs + EBOPs (§6′)** | `results/hgq2/tradeoff_table.md`; recomputable from `results/hgq2/runs/<hash>/scores.npz` vs `roc-results/r5/*.npz` (verified 2026-07-04, results-analyst pass in experiment-log) |
| HGQ2-path csynth probes (SubLN, binary dense, attn core) | `results/hgq2/runs/<hash>/probe_*/csynth_report.json` (raw from mulder `~/bnjet_hgq2/`) |
| HGQ2+hls4ml layer support / constraints map | `results/hgq2/constraints_map.md`; change trail in `code/hgq2/LEDGER.md` |

Verification procedure: `.claude/skills/verify-roc/SKILL.md`. Anything not traceable to this
table is not a project claim.

---

## 9. Literature anchors

Full annotated notes in [`papers/`](papers/); the four load-bearing references:

- **BitNet** (arXiv:2310.11453) — the 1-bit transformer training method we adapt
  (+ b1.58 ternary, 2402.17764; a4.8 activations, 2411.04965). PDF in
  `papers/bitnet-1bit-ternary/`.
- **Sub-microsecond Transformers for Jet Tagging on FPGAs** (arXiv:2510.24784) — the
  closest published system: full-precision-ish transformers through hls4ml into L1-trigger
  latency. Our differentiator is the 1-bit weight core and its DSP-free mapping. PDF in
  `papers/jet-tagging-transformers/`.
- **hls4ml** (arXiv:1804.06913) — the codesign tool and the trigger-ML program it anchors.
- **HLS4ML LHC Jet dataset** (arXiv:1804.06913 + 1908.05318, Zenodo) — the era-2 public
  benchmark dataset, which makes our numbers comparable to published taggers.
