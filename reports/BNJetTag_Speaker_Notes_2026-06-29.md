# BNJetTag — Speaker Notes

**Companion to:** `BNJetTag_Slides_2026-06-29.docx` (figures only).  
**Date:** 2026-06-29 · **Project:** binary {-1,+1} 1-bit BitNet transformer jet tagger for the CMS L1 trigger.

> One-line thesis to open with: *we build a 1-bit binary BitNet transformer jet tagger; the binary {-1,+1} weights map to FPGA LUT/logic instead of DSP multipliers (~0 DSP on the matmul core) while tagging efficiency stays close to full precision. Today: the silicon result is proven on the real model, and a GPU sweep improved the tagger for free.*

Every number below is from the project's authoritative sources: `results/variant_sweep.md` (sweep), `results/hls_resource_table.md` (Sections B / B' / C), `roc-results/roc_auc.md` (ROC-test), and the validation-AUC headline table. AUCs on the ROC slide were recomputed from the `.npz` and match `roc_auc.md` exactly.

---

## Slide 1 — The model — a binary BitLinear transformer jet tagger

**Figure:** `qkeras-bitnet-run-2026-06-22/results/plots/model_architecture.png`

- Input per jet = 10 constituents x 14 features. `input_proj` (a BitLinear) lifts to D=256; then 8 BitTransformerBlocks (this diagram is the 'large' config D256/H8/L8/FFN1024 = 6,373,633 params); global-average-pool over constituents; 2-layer head -> 1 logit (top vs QCD).
- Every dense layer is a BitLinear (BitNet, arXiv:2310.11453): weights binarized {-1,+1} on the fly by a straight-through estimator (alpha=mean(W); W~=Sign(W-alpha); scale beta=mean|W-alpha|). Activations are 8-bit absmax (A8). Normalization is SubLN (one LayerNorm per BitLinear).
- The whole model is env-knob driven, so this is just ONE point on a size axis tiny->small->medium->large (26.5K / 154K / 808K / 6.37M params).

---

## Slide 2 — The data — jet-level kinematics (signal vs background)

**Figure:** `qkeras-bitnet-run-2026-06-22/results/plots/data_jet_kinematics.png`

- Standard hls4ml jet-tagging data: signal (top jets) vs background (QCD).
- The pT spectra differ strongly (background reaches much higher pT), so if we don't correct for it the classifier can cheat on pT. eta/phi are ~flat/symmetric. -> motivates the pT reweighting (next-but-one slide).

---

## Slide 3 — The inputs — 14 per-constituent features

**Figure:** `qkeras-bitnet-run-2026-06-22/results/plots/data_particle_kinematics.png`

- The 14 per-constituent inputs the transformer actually sees (track displacement dz/dx/dy, relative pT, eta, phi, ...), signal vs background.
- Constituents are a permutation-invariant SET (no natural order). Remember this: later we show that adding a learned positional encoding HURTS — order is not signal here.

---

## Slide 4 — Preprocessing — pT reweighting

**Figure:** `qkeras-bitnet-run-2026-06-22/results/plots/sample_pt_reweighting.png`

- Per-jet training weights: most near 1 with a tail. These balance the signal/background pT spectra so AUC reflects jet substructure, not the pT difference. The same reweighting is used across every run in this deck.

---

## Slide 5 — System & code map — what we built and edited

**Figure:** `qkeras-bitnet-run-2026-06-22/results/plots/workflow_code.png`

- Walk the diagram left-to-right (training), then the checkpoint feeds two branches (ROC + HLS). Everything blue is code we wrote or edited. Detail below:
- **qkerasModel.py (training core).** Made fully env-driven: added ~22 knobs so a variant is a YAML, not a code change -- BN_D_MODEL/N_HEADS/N_LAYERS/FFN_DIM (size); BN_NORM_TYPE/NORM_PLACEMENT/POS_ENC/POOL/FFN_ACT (structure); BN_LR/WARMUP_EPOCHS/DECAY_EPOCHS/DECAY_POWER/BETA2/WEIGHT_DECAY/CLIPNORM/L1_REG/BATCH (recipe); BN_ES_MONITOR/MODE/PATIENCE; BN_SEED. Added layers TrueRMSNorm, IdentityNorm, CLSToken, LearnedPosEnc, _sinusoidal_pos and a make_norm() factory. Added a `--sanity` CPU preflight (build + forward + train_on_batch, no GPU).
- **Bug we fixed in qkerasModel.py:** the default 'learned' positional encoding built Embedding(N,D) applied to a constant, which Keras folds to a fixed-random, 0-parameter no-op. We added BN_POS_ENC=learned_real (a genuine trainable (N,D) table via add_weight) so we could honestly test whether position helps. (It doesn't.)
- **gen_round{2,3,4}_jobs.py (variant engine).** Each script emits the per-variant Job YAMLs + a CPU preflight from one BASE_RECIPE dict, so an entire round is reproducible from a single command. Round-4 labels app=bnjet-variant-r4 and writes logs to /data/outputs/qk-variants-r4/.
- **Orchestration scripts** (preflight_r4.sh, launch_all_r4.sh, run_round4_chain.sh): gate on the CPU preflight (grep PREFLIGHT_ALL_PASS), launch the round, poll all jobs to terminal, then extract best val_auc from the PVC logs -- one command per round.
- **NRP Job YAMLs + utility pods** (kai-bn{2,3,4}-*.yaml, kai-util3.yaml): GPU work runs as Kubernetes Jobs (backoffLimit 0); short alpine 'sleeper' pods kubectl-exec into the PVC to read durable train_*.log after the job pods are garbage-collected.
- **make_roc.py (offline ROC).** cmd_eval loads a model + a fixed-seed held-out tail -> AUC -> writes {y,score}.npz; cmd_plot aggregates -> ROC overlay + AUC table. This is the source of the ROC-test numbers.
- **HLS reconstruction** (full_model_csynth.py, run_csynth.py, resource_model.py): rebuild every BitLinear as LayerNormalization -> QActivation(quantized_bits) -> QDense(binary), port the trained weights through the BitNet AbsMeanQuantizer, and run fidelity / convert (bit-accuracy) / csynth. These produced the silicon numbers (Sections B / B').
- **Infrastructure:** training on NRP Nautilus (Kubernetes GPU, namespace cms-ml, image tensorflow 2.11-gpu, PVC kai-data); HLS C-synthesis on the group's 'mulder' box (Vitis HLS 2023.2 -- NRP has no Xilinx backend); tracking in W&B project bnjettag-bitnet. The W&B API key is kept out of the repo (never printed or committed).

---

## Slide 6 — Headline result — DSP = 0 in real Vitis C-synthesis

**Figure:** `qkeras-bitnet-run-2026-06-22/results/plots/results_csynth_resources.png`

- This is the thesis in one chart. The binary FFN block was C-synthesized for real on mulder (Vitis HLS 2023.2), folded operating point RF=256, part xcvu13p (VU13P), 400 MHz target (Vitis estimated ~568 MHz).
- DSP = 0 at A8, A6, AND A4. A 1-bit weight is the C-type ap_uint<1> -- it cannot drive a DSP multiplier port, so the MAC becomes sign-select + a LUT adder tree and Vitis instantiates ZERO DSP48s. This is structural and precision-independent -- the abstract's central claim, now in silicon estimates, not just firmware inference.
- The rest is modest: LUT ~25%, FF ~7%, BRAM ~1.2% of a VU13P; latency 520 cycles ~ 1.3 us @ 400 MHz; II=256.
- Honest nuance: BRAM=0 only for the fully-unrolled RF=1 design; the deployable folded point parks the binary weights in ~1.2% BRAM. DSP=0 holds either way -- it is the fold-independent win.

---

## Slide 7 — Two axes — efficiency vs FPGA resources (A8 -> A4)

**Figure:** `qkeras-bitnet-run-2026-06-22/results/plots/results_quant_axis.png`

- The abstract's two axes in one figure. TOP: validation AUC vs activation bits -- A8 0.7524 -> A6 0.7507 -> A4 0.7437: a gentle ~0.9-point slide over 8->4 bits, NO cliff.
- BOTTOM: measured FPGA resources vs bits -- LUT 25.5% -> 24.0%, FF 7.3% -> 6.6%, BRAM 1.19% (flat), DSP 0%. Narrower activations shrink the datapath monotonically -- the same 'no cliff' trend as the AUC.
- Takeaway: you can push activations far down before either efficiency OR resources degrade -- exactly what we wanted to show for a trigger deployment.

---

## Slide 8 — The FULL transformer in silicon — 100% of DSP is LayerNorm

**Figure:** `qkeras-bitnet-run-2026-06-22/results/plots/hls_dsp_breakdown.png`

- This answers the obvious question: 'are you just synthesizing an FFN?' No. The entire trained transformer (51 BitLinears + 51 SubLN norms + the 4 attention projections) was reconstructed from hls4ml-supported primitives with the TRAINED weights ported in, and synthesized per layer shape on mulder.
- Validation that the rebuild IS the trained model: fidelity corr = 0.99998; QKeras-vs-Vitis bit-accuracy 0.9967-0.9999 per layer.
- Result: the binary matmul core = 0 DSP. The ONLY DSP in the whole transformer -- 1,049 total (8.5% of a VU13P) -- is 100% LayerNorm (real-valued variance Sum(x^2) + inv-sqrt at fixed<32,16>). It is precision-independent (A8=A6=A4). A LUT-based inv-sqrt would drive even that toward 0 (future work).
- Caveat: the fully-spatial composition does NOT fit one chip (LUT ~5.2x) -- a 6.37M-param transformer must fold/stream the 8 identical blocks temporally; the per-shape numbers are the measured truth.

---

## Slide 9 — Tagging efficiency — cost of going 1-bit ~ -1.7 points

**Figure:** `qkeras-bitnet-run-2026-06-22/results/plots/results_auc_by_variant.png`

- Validation AUC across weight/activation precision: FP32 0.7703, W8A8 0.7719 (8-bit is ~lossless), and the deployed 1-bit BitNet W1A8 = 0.7530.
- Cost of going fully 1-bit ~ -1.7 points vs FP32 -- paid ONCE, in exchange for 0 DSP on the binary core.
- Ternary {-1,0,+1} (0.7685) and softmax-free (0.7562) are appendix/reference points. Ternary is OFF-thesis (binary is the target); it's only here as a literature comparison.

---

## Slide 10 — ROC on the held-out test set (n = 222,912)

**Figure:** `qkeras-bitnet-run-2026-06-22/results/plots/roc_overlay_clean.png`

- ROC on the held-out test set; AUCs recomputed from the .npz: FP32 0.8207, W8A8 0.8283, binary A8 0.7985, A6 0.7984, A4 0.7886.
- Note the two metrics differ on purpose: this ROC-test AUC (~0.80) is HIGHER than the validation AUC (~0.75) from the previous slides -- different data split, do not conflate. We always say which one we mean.
- The binary curves sit just under the baselines; A8 and A6 are nearly on top of each other ('no cliff' again); A4 is slightly lower.

---

## Slide 11 — GPU sweep — once the LR is tuned, bigger wins

**Figure:** `qkeras-bitnet-run-2026-06-22/results/plots/sweep_auc_vs_size.png`

- Recent work: the ask was to forget HLS for a bit and find the best BitLinear config on the GPU. The key finding is that the OPTIMIZER (peak learning rate), not the architecture, was the hidden ceiling.
- Our original recipe 'lr15' (peak LR 1.5e-4) was tuned for the big model and is too hot for the smaller ones. Lowering to ~5e-5 helps at EVERY size (blue dashed = un-tuned lr15; red = tuned).
- Once tuned, AUC is monotonic in size: small ~0.750 < medium 0.7567 < large 0.7672. The 'large' model itself improved +0.0142 (0.7530 -> 0.7672) for free, just by lowering the LR -- no architecture change.

---

## Slide 12 — The recipe was the ceiling — lower LR fixes the collapse

**Figure:** `qkeras-bitnet-run-2026-06-22/results/plots/sweep_lr_small.png`

- The mechanism, on the small model. Lower peak LR -> higher AUC and a later, more stable peak. Under lr15 the model peaked at epoch 6 then collapsed; at 5e-5 it trains to epoch 17.
- It plateaus around 0.750 between 3.5e-5 and 5e-5 (best 0.7537 @ 3.5e-5). The 5e-5 point is a 3-seed mean +/- 0.0036.
- So the round-1 'peak-then-collapse' we first saw was a learning-rate artifact, NOT an architecture problem.

---

## Slide 13 — Rigor — seed repeats show the 'architecture wins' were noise

**Figure:** `qkeras-bitnet-run-2026-06-22/results/plots/sweep_seed_noise.png`

- Why we trust the previous two slides: seed repeats. In round-1, a single CLS-pool run looked +0.014 better than the default.
- With 3 seeds each, the MEANS are equal (clspool 0.7355 ~ small 0.7353); clspool just has far higher run-to-run variance (sigma 0.023 vs 0.003). The '+0.014' was one lucky draw.
- Conclusion: the round-1 'architecture wins' (CLS-pool / GELU / softmax-free) were noise and don't stack. Keep the DEFAULT architecture; spend the effort on the recipe.

---

## Slide 14 — Efficiency frontier & recommendation

**Figure:** `qkeras-bitnet-run-2026-06-22/results/plots/sweep_efficiency_frontier.png`

- Best validation AUC achieved per model size, all at a tuned LR. The recommendation:
- Best AUC: large (6.37M) @ 5e-5 -> 0.7672 (single run; seed-confirm is the next step).
- Best efficiency / deployment pick: medium (808K) @ 5e-5 -> 0.7567 -- within ~0.010 of large at ~8x fewer params (and ~8x less FPGA logic).
- Smallest viable: small (154K) ~0.750. And keep the default architecture knobs -- none of the ablations beat default under seed-averaging.
- Bottom line for the PI: the silicon claim (0 DSP) is proven on the real model, the binary tagger costs ~1.7 AUC points vs FP32 with no cliff to A4, and a free LR fix lifted it 0.7530 -> 0.7672.

---

## Appendix A — Two AUC metrics (never conflate)

- **Validation AUC** (during-training split): the headline-table / W&B number and what the GPU sweep optimizes. FP32 0.7703 · W8A8 0.7719 · W1A8 binary 0.7530 · A8 0.7524 · A6 0.7507 · A4 0.7437.
- **ROC-test AUC** (held-out tail, n=222,912, recomputed from .npz): FP32 0.8207 · W8A8 0.8283 · A8 0.7985 · A6 0.7984 · A4 0.7886.
- They differ by ~+0.045 to +0.056 on purpose (different split). Both are internally consistent; we never reconcile them into one number.

## Appendix B — Published silicon anchor (Section C)

- Ngadiuba et al., arXiv:2003.06308 — a binary jet-tagging MLP synthesized through hls4ml with real Vivado logic synthesis: **0 DSP, 0 BRAM, ~1% LUT, 40 ns**. This independently validates our 0-DSP result on the same task. Their model is ~1000x smaller (fits fully-unrolled), so our LUT% (~25% folded) and latency (1.3 us) are larger — but the structural 0-DSP result transfers directly.

## Appendix C — Next steps

- **Seed-confirm large @ 5e-5:** 0.7672 is a single run; repeat with 2-3 seeds for an error bar.
- **Round-5 sweep:** 10 more variants, run <=3 concurrent (precision A6/A4 on medium/small, a couple of shape probes, the medium LR ceiling, one bigger model). Concurrency-cap scope to confirm before launch.
- **ROC curves in W&B:** backfill per-run ROC via make_roc.py + wandb.plot.roc_curve and a cross-run overlay; add a logging hook to qkerasModel.py (today it logs scalars only).
- **A6/A4 full-model csynth backfill** into Section B' (DSP is precision-independent -> expect the same 1,049 DSP with modestly smaller LUT/FF).

## Appendix D — Caveats to state honestly

- `large @ 5e-5 = 0.7672` is a single run (seed-confirm pending).
- The fully-spatial Section B' composition does not fit one VU13P (LUT ~5.2x); a deployable design folds the 8 identical blocks temporally. The trustworthy numbers are the per-shape rows.
- BRAM=0 only for the fully-unrolled RF=1 design; the folded deployable point uses ~1.2% BRAM. DSP=0 is the fold-independent structural result.
- Ternary {-1,0,+1} is OFF-thesis (binary is the target); shown only as a reference.
