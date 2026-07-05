# ROADMAP — BNJetTag task list, checklists, and success indicators

*The steering document. `RESEARCH.md` says what we know; this file says what we do next,
in what order, and how we'll know each step worked.*

**How to use:** work top-down within a phase; phases 2–3 can overlap phase 1 (they need no
GPUs). Check boxes as items land. An item is done only when its **success indicator** is
met — not when the work "ran". Update this file as things land; when a phase completes,
its verified numbers go to `RESEARCH.md` and the checked list stays here as history.

Status marks: `[ ]` open · `[x]` done · `[~]` in flight/partial.

---

## Where the research stands (2026-07-02)

**Banked (verified, safe to claim):**
- Binary matmul core = **0 DSP**, structural, confirmed in real Vitis csynth at A8/A6/A4.
- All 1,049 model DSPs (8.5% VU13P) are LayerNorm — **precision-independent** (= at A8/A6/A4).
- Full trained model synthesized end-to-end; QKeras↔Vitis fidelity ≥ 0.9967.
- Composed whole-model latency **upper bound**: 23,409 cycles ≈ 58.5 µs @ 400 MHz target
  (folded RF=256; attention score core excluded).
- Era-1 accuracy tables (frozen in `RESEARCH.md` §5) — historical only, not comparable to era 2.

**In flight:** round-5 — 8 jobs on NRP Nautilus (era-2 public 5-class data, launched 2026-07-01).

**Pending on Kai:** `git push origin main round-5` (2 commits + tag, sandbox has no SSH key).

**Decided next experiment:** LayerNorm-DSP elimination (round-6) — the path to a **fully
0-DSP transformer**, the claim FP-transformer prior art (arXiv:2510.24784) structurally can't match.

---

## Phase 0 — Housekeeping (this week; minutes each)

*Goal: no irreplaceable state exists in exactly one place.*

- [ ] **Push the run repo.** In `qkeras-bitnet-run-2026-06-22/`:
      `git push origin main round-5`.
      ✅ *Success:* `git status` shows "up to date with origin/main"; tag `round-5` and
      commits `adfa379` + `52cb818` visible on GitHub.
- [ ] **Third copy of irreplaceable artifacts** (`models/`, `roc-results/`,
      `results/csynth/`) to durable storage (university drive / cloud bucket / external disk).
      ✅ *Success:* a copy exists that is neither this laptop nor the `kai-data` PVC, with a
      dated manifest (file list + sizes); calendar reminder set to refresh quarterly.
- [ ] **Delete stale git lock debris**: `rm qkeras-bitnet-run-2026-06-22/.git/index.lock.stale
      qkeras-bitnet-run-2026-06-22/.git/lock-graveyard-*`.
      ✅ *Success:* no `*.lock*`/`lock-graveyard` files under `.git/`.
- [ ] **Tag discipline standing rule:** tag every round at launch (`r6-...` when round-6 goes up).
      ✅ *Success:* every future round's YAMLs are reachable from an annotated tag.

---

## Phase 1 — Round-5: the era-2 accuracy story (THE gate; everything accuracy-side hangs here)

*Goal: the full quantization matrix on the public HLS4ML LHC Jet 5-class set — the numbers
the paper will actually quote.*

**The matrix (8 jobs, in flight):** `kai-bn5-fp32-lr05`, `kai-bn5-w8a8-lr05`,
W1A8 `-s1/-s2`, A6 `-s1/-s2`, A4 `-s1/-s2`. Playbook: `.claude/skills/nrp-training-run/`.

- [~] **1.1 All 8 jobs complete.** Monitor (`monitor.sh` / `kubectl get pods -n cms-ml`);
      watch W&B (`bnjettag-bitnet`) loss curves — LR 5e-5 was tuned on era-1 and carried over,
      so instability here is a *finding*, not just a failure.
      ✅ *Success:* 8/8 pods `Completed`; no NaN/divergence in curves; param count printed =
      6,375,173; checkpoints on the PVC.
      ⚠️ *If a run diverges:* log it, halve LR for that variant only, relaunch that job —
      do not touch the others (keeps the sweep interpretable).
- [ ] **1.2 ROC pass.** Submit `kai-roc-r5.yaml` against all 8 checkpoints on the held-out
      ROC-test split.
      ✅ *Success:* one `.npz` per run (keys `y`/`score`/`meta`), n = held-out test size,
      brought home to `roc-results/`.
- [ ] **1.3 Verification gate** (results-analyst, `.claude/skills/verify-roc/`): recompute
      macro-OvR AUC from every `.npz` — era-2 = 5-class,
      `roc_auc_score(y_onehot, scores, multi_class="ovr", average="macro")`.
      ✅ *Success:* every number carries the three labels (metric / era / status) and a ✓ in
      the experiment log; **no unverified number leaves this step.**
- [ ] **1.4 Seed-average.** Report mean ± spread for W1A8/A6/A4 (2 seeds each).
      ✅ *Success:* seed spread quantified; if |s1−s2| macro-OvR AUC > ~0.005 for the W1A8
      flagship, queue a third seed for W1A8 only before headlining it.
- [ ] **1.5 Fill era-2 tables in `RESEARCH.md` §5** (skeleton first — see 2.1): the
      quantization ladder FP32 → W8A8 → W1A8 → A6 → A4, val + ROC-test, vs published
      baselines on this dataset (framing = "binary-vs-FP gap", never an accuracy-SOTA chase).
      ✅ *Success:* no empty cells in the minimum publishable matrix (W1A8+A6+A4 vs tuned
      FP32/W8A8, ≥2 seeds); every cell traceable via §8 source map.
- [ ] **1.6 Regenerate downstream prose** — slides/report text now quoting era-2 only;
      era-1 numbers survive only in the frozen §5 tables and `reports/`.
      ✅ *Success:* grep for era-1 headline AUCs (0.7530, 0.8207…) finds them only in
      frozen/labelled contexts.

**Phase-1 exit criterion (the "accuracy story exists" bar):** seed-averaged, ROC-tested,
era-2 quantization ladder in `RESEARCH.md` with the binary-vs-FP32 gap stated ± spread.

---

## Phase 2 — GPU-free parallel work (start now; needs no cluster)

*Goal: when round-5 lands, landing the numbers is mechanical, and the paper's framing
already exists.*

- [ ] **2.1 Era-2 results skeleton** (paper-writer): §5 tables with empty cells + captions +
      methods paragraph (dataset, split, metric, seeds) written *before* numbers exist.
      ✅ *Success:* filling = paste numbers; zero structural writing left for phase 1.5.
- [ ] **2.2 Positioning paragraph vs arXiv:2510.24784** (physics-researcher close-read →
      paper-writer): what they do / what we do differently / why it matters.
      ✅ *Success:* ≤1 page in `papers/jet-tagging-transformers/`, answering: their
      latency & DSP spend; our DSP=0 core + quantization-frontier story; one sentence a
      reviewer could quote. Logged in research log with date+URL.
- [ ] **2.3 hls4ml LayerNorm feasibility question** (blocks Axis B of phase 3): does hls4ml's
      `LayerNormalization` support a table-based inv-sqrt, or does `full_model_csynth.py`
      need a custom norm primitive?
      ✅ *Success:* written answer in the research log with the hls4ml source/docs cited;
      Axis-B implementation path chosen.
- [ ] **2.4 PI update** (`/pi-update`): where things stand + the 0-DSP differentiator plan.
      ✅ *Success:* one honest page in `reports/`, era-labels intact, no unverified numbers.

---

## Phase 3 — Round-6: LayerNorm-DSP elimination (the differentiator; gated on round-5)

*Goal: a fully 0-DSP transformer — the structurally unanswerable claim. Decision log:
`decisions.md` 2026-07-02 (top entry).*

**Axis A — cheapest norm that holds AUC (NRP, era-2 data):**
- [ ] **3.1** ml-engineer scaffolds `kai-bn6-norm-*` YAMLs: `BN_NORM_PLACEMENT`
      {per_linear, shared_prenorm} × `BN_NORM_TYPE` {layernorm, rmsnorm, none} — knobs exist;
      mirror the `gen_round5_jobs.py` pattern + preflight.
      ✅ *Success:* generator + YAMLs committed; `PREFLIGHT_ALL_PASS` printed before launch.
- [ ] **3.2** Train sweep vs the round-5 layernorm baseline.
      ✅ *Success:* macro-OvR val AUC per variant, verified per phase-1 gate; a
      DSP-proxy-vs-AUC table (51 norms / ~17 norms / 0 norms).
- [ ] **3.3** *If* no existing knob both holds AUC and reaches ~0 DSP: add a DSP-free norm
      (power-of-2/shift scale, or frozen precomputed scale → constant multiply).
      ✅ *Success:* new knob value documented in README knob table; same preflight bar.

**Axis B — firmware (mulder, `.claude/skills/hls-mulder/`):**
- [ ] **3.4** Re-synthesize the winning norm with fixed-point inv-sqrt LUT / narrowed LN
      precision (reuse `full_model_csynth.py`; answer from 2.3 decides the path).
      ✅ *Success:* csynth JSON shows whole-model **DSP → 0** (or a clean DSP-vs-AUC
      frontier), **and** QKeras↔Vitis C-sim corr ≥ 0.997 (the 0.87-corr trap from narrow LN
      is the known failure mode).

**Phase-3 exit criterion / headline:** *a fully 0-DSP BitNet transformer jet tagger* with
macro-OvR AUC within seed-noise of the layernorm baseline — or, failing that, a quantified
DSP-vs-AUC frontier. Either is publishable; the first is the headline.

---

## Phase 4 — Complete the hardware story (mulder; interleave as available)

*Goal: turn "upper bound, with exclusions" into a quotable whole-model hardware claim.*

- [ ] **4.1 Attention score core** (QKᵀ/softmax/AV — currently excluded, 0.65% of MACs):
      synthesize via hls4ml when EinsumDense support lands, or hand-map the three small ops.
      ✅ *Success:* composed latency loses its exclusion caveat; DSP census re-checked
      (does softmax/score add DSPs? if yes, feeds phase 3's 0-DSP accounting).
- [ ] **4.2 L1-relevant operating point:** pick and synthesize a lower-RF / streamed
      configuration sized against the CMS L1 budget (folded RF=256 ≈ 58.5 µs is the frugal
      extreme, not the L1 point).
      ✅ *Success:* one stated (RF, clock) point with measured latency + resources on VU13P,
      quotable as "the model runs in X µs using Y% LUT / 0 DSP"; input_proj timing fixed
      (currently 3.71 ns > 2.5 ns target).
- [ ] **4.3 (Stretch) place-and-route sanity** on the chosen point — csynth estimates vs
      post-implementation.
      ✅ *Success:* Fmax and resources within ~20% of csynth estimates, or the delta documented.

---

## Phase 5 — Assembly (after phases 1 + 3 land)

- [ ] **5.1** `RESEARCH.md` final pass: knowledge ledger has no "provisional" entries the
      paper relies on; every §8 source row resolves.
- [ ] **5.2** Paper draft (paper-writer): abstract (exists, §1) → methods (decision log makes
      this mechanical) → era-2 results → hardware → positioning (2.2).
      ✅ *Success:* every figure/number carries metric+era+source; a "numbers audit" block
      passes results-analyst review.
- [ ] **5.3** PI/venue decision with advisor: workshop paper vs internal note vs poster.
      ✅ *Success:* target + deadline recorded in `decisions.md`.
- [ ] **5.4** Deferred axis on record: `BN_N_PART` input-size study stays deferred by design
      (advisor call, 2026-07-01) — stated as future work, not silently dropped.

---

## Project-level definition of done

The thesis is defended when all four hold:

1. **Accuracy:** era-2 seed-averaged ROC-tested ladder (FP32 → W8A8 → W1A8 → A6 → A4)
   quantifying the binary-vs-FP gap on a public benchmark. *(Phase 1)*
2. **Hardware:** whole-model synthesis with **DSP = 0 core** (ideally fully 0-DSP) and one
   quotable latency/resource operating point. *(Phases 3–4)*
3. **Differentiation:** written positioning vs 2510.24784 — the claim they structurally
   cannot match. *(Phase 2.2 + 3)*
4. **Reproducibility:** every number recomputable from committed artifacts; every round
   reachable from a git tag; three copies of irreplaceable data. *(Phase 0, continuous)*

## Standing rules (apply to every phase)

- Never compare across eras; every number carries metric (val | ROC-test) + era + status.
- No number reaches `RESEARCH.md`/reports without the results-analyst gate (verify-roc).
- Training = NRP, synthesis = mulder, never locally; preflight must print
  `PREFLIGHT_ALL_PASS` before GPUs are spent.
- `reports/` is frozen history; corrections are dated notes.
- Log as you go: experiments → experiment-log, choices → decisions, sources → research-log.
