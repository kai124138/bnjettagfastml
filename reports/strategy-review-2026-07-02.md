# Strategy review — how BNJetTag is being run (2026-07-02)

An honest outside-in assessment of the project as a whole: what the current way of working
gets right, where the real risks are, and what to do next. Written against the state of
2026-07-02 (round-5 training in flight on the new dataset).

## What's working — keep doing it

**The verification culture is the project's best asset.** The two-AUC discipline, the
"recompute from `.npz` or don't quote it" rule, and the results-analyst gate have already
paid for themselves twice: the lr15-too-hot discovery and the tuned-vs-untuned trap (which
would have produced the false headline "medium beats large", and later a false
binary-vs-FP32 gap). Most student projects discover these traps in peer review. This one
catches them in-house.

**The experiment design scales.** Variants live in env knobs (`BN_VARIANT`, `BN_ACT_BITS`,
…), sweeps are *generated* (`gen_round5_jobs.py`), and every round ships with a preflight
that must print `PREFLIGHT_ALL_PASS` before GPUs are spent. That is exactly how you run
dozens of runs without losing track of what differs between them.

**The hardware half of the thesis is already banked.** DSP = 0 for the binary core is
structural, confirmed in real Vitis C-synthesis, and independent of the dataset change —
so the riskiest, most original claim is de-risked regardless of how round-5's accuracy
numbers land. The full trained model has been synthesized end-to-end with fidelity checks.
Few "we'll put it on an FPGA eventually" projects ever get here.

**The dataset migration was the right call at the right time.** Moving to the public HLS4ML
LHC Jet set (5-class, proper held-out ROC-test) makes every future number comparable to the
published literature and citable in a paper — and the decision log records exactly what it
invalidated, which is what will make the paper's methods section easy to write.

## The risks, ranked

**1. Everything accuracy-side now hangs on round-5.** All eight jobs must finish, pass
ROC (`kai-roc-r5.yaml`), and be seed-averaged before a single era-2 accuracy claim exists.
Until then the slides and reports quote era-1 numbers that are officially non-comparable to
what comes next. Mitigations: (a) pre-write the era-2 results section and tables with
empty cells now, so landing numbers is mechanical; (b) decide *now* what the minimum
publishable matrix is (W1A8 + A6 + A4 vs tuned FP32/W8A8, 2 seeds) so "done" is defined;
(c) two seeds bounds the noise but is thin for a headline — if GPUs allow, add a third seed
for the W1A8 flagship only.

**2. Version control is one step behind the work.** The run repo has uncommitted
round-5-critical changes (`qkerasModel.py`, `make_roc.py`, `README.md`, `kai-roc-r5.yaml`
untracked) and no remote. The training that is running *right now* corresponds to a code
state that exists only as working-tree changes in a Downloads folder. Fix this week:
commit, tag (`r4-lr15-anchor`, `r5-migration-launch`), and push to a private remote.
Tag every round the moment it launches — reproducibility for the paper depends on being
able to say "round-5 = commit X".

**3. Irreplaceable artifacts have no third copy.** Checkpoints, `.npz` ROC arrays, and
csynth JSONs exist here and on the `kai-data` PVC. Nautilus PVCs are not guaranteed
backups, and this folder is a personal machine. A quarterly-effort task with
losing-the-thesis downside: put `models/`, `roc-results/`, `results/csynth/` somewhere
durable (university storage, cloud bucket, even an external drive with a calendar
reminder).

**4. The differentiator needs sharpening against the closest prior art.** Sub-microsecond
transformer jet taggers on FPGAs are now published (arXiv:2510.24784, filed in
`papers/jet-tagging-transformers/`). That's good news — it validates the problem — but it
means the paper's claim can't be "a transformer in the trigger"; it must be the thing they
don't have: **a {−1,+1} weight core whose matmuls use zero DSPs**, plus the
how-far-can-you-quantize story. Two concrete moves: a close read of that paper with a
written positioning paragraph (what they do, what we do differently, why it matters), and
resolving the **LayerNorm DSP question** — the model's only DSPs (1,049; 8.5% of a VU13P)
sit in LayerNorms, so an integer/shift-based or norm-lite variant could make the *entire*
model 0-DSP. That single result would turn a good differentiator into an unanswerable one,
and the `rmsnorm`/`sharednorm`/`sffree` variant machinery to test it already exists.

**5. "So what's the latency of the whole model?" has no single answer yet.** Per-layer
csynth numbers exist; a composed end-to-end latency/resource figure at a stated operating
point (RF, clock) does not — and it's the first question a reviewer or the PI will ask of
the hardware story. The A6/A4 full-model backfill (`BACKFILL_A6_A4_FULL_MODEL.md`) is part
of the same gap. This work needs mulder, not GPUs — it can proceed *while round-5 trains*.

**6. Small carried assumptions, flagged so they stay small.** Peak LR 5e-5 was tuned on
era-1 data and carried over unverified (watch round-5 curves; spot-check if unstable).
Comparability on a public benchmark cuts both ways: published baselines exist for this
dataset, so the framing must stay "we measure the binary-vs-FP gap and the hardware win",
not an accuracy-SOTA chase the model isn't sized for.

## What to do next, in order

1. **Today (5 minutes):** commit + tag + push the run repo (risk #2).
2. **While round-5 trains (GPU-free work):** on mulder, backfill A6/A4 full-model csynth
   and build the composed full-model latency/resource estimate (risk #5); meanwhile
   paper-writer drafts the era-2 results skeleton with empty tables and the positioning
   paragraph vs 2510.24784 (risks #1, #4).
3. **When round-5 lands:** `verify-roc` everything, seed-average, fill the tables, then
   regenerate slides/report from era-2 numbers only.
4. **Next experiment decision:** the LayerNorm-DSP elimination variant — highest
   marginal-claim value per GPU-hour of anything on the list (risk #4).
5. **This month:** set up the third copy of irreplaceable artifacts (risk #3).

## One structural observation

The project's biggest process risk used to be that its state lived in one person's head and
a chat scrollback. The memory logs fixed the "why"; `RESEARCH.md` + `00-START-HERE.md` now
fix the "what" and "where". The remaining single point of failure is *artifacts* (risks
#2, #3) — that's deliberately where two of the five next moves point.
