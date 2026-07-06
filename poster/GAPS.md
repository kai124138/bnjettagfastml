# Poster gaps — numbers/attributions the poster wants but the store cannot give

Nothing below was invented or approximated in the figures/draft; each item is
either absent from the store or failed the raw-data gate (poster/VERIFICATION.md §4).

1. **Per-instance DSP attribution inside the HGQ2 probes.** The store has only
   whole-probe csynth totals; the per-instance Vitis report that attributes the
   head_fc2 probe's 112 DSP to SubLN (binary dense 0, CSD-2 affine 0) and the
   Resource probe's 270 to dense 256 + SubLN 14 lives on **mulder** and is only
   quoted in `code/hgq2/LEDGER.md`. The DSP figure works around it with the raw
   probe-level dichotomy. *To close (live session, not this one): fetch the
   per-instance csynth reports (`csynth.xml` module table) for
   `probe_bitlinear_head_fc2_rf32` and `probe_bitlinear_rf256` into the store —
   that un-bans the sharper per-function split for the poster.*
   > **CLOSED 2026-07-05 (live session).** Raw `csynth.xml` + parsed
   > `csynth_modules.json` now in the store for all four finished probes
   > (`results/hgq2/runs/b224a8ea/<probe>/`). Confirmed per-function split:
   > head_fc2_rf32 → SubLN 112 / binary dense 0 / CSD-2 affine 0 (top 112);
   > rf256 Resource → dense 256 / SubLN 14 (top 270); subln_rf1 → 1792.
   > Parser: `code/hgq2/parse_csynth_modules.py`; fetch script now pulls both
   > files automatically. The per-function split is un-banned.
   > *Poster side (2026-07-05): independently re-derived from the csynth.xml
   > module tables (`poster/scripts/verify_dsp_split.py`, all checks PASS —
   > VERIFICATION.md §5); fig3 and the draft now carry the measured split.*

2. **Attention-core synthesis.** `probe_attn_core_rf1` (QEinsum + QSoftmax score
   core, C-sim corr 1.0) was still synthesizing on mulder at freeze. The poster
   currently has **no measured resource/latency number for the attention core**;
   the composed latency explicitly excludes it. *To close: fetch when it lands.*
   > **CLOSED (rf1) 2026-07-05 (live session).** `probe_attn_core_rf1` landed
   > and is in the store (`runs/b224a8ea/probe_attn_core_rf1/csynth_report.json`
   > + raw xml + modules json): 31 cycles @ est. 1.812 ns, LUT 4,271,510 (247%
   > VU13P), FF 9,467,557, DSP 52,000 (423%), BRAM 720 — fully-spatial extreme.
   > Per-module: each act×act einsum (QKᵀ and ctx·V, 10×10×8×32 = 25,600 MACs)
   > = 25,600 DSP, i.e. exactly 1 DSP per MAC; softmax itself only 10 DSP.
   > This is the measured quantification of why act×act attention cannot be
   > binarized (no weights to binarize → every product a real multiplier).
   > A deployable folded variant `probe_attn_core_rf64` is synthesizing on
   > mulder; fetch when it lands (already in the fetch script's PAIRS).
   > **CLOSED (rf64) 2026-07-05 (live session).** `probe_attn_core_rf64` landed
   > (`runs/b224a8ea/probe_attn_core_rf64/` — report + raw xml + modules json):
   > 193 cycles @ II=64, est. 2.009 ns (0.48 µs at 2.5 ns target), **DSP 820**
   > (einsums 400 each = 25,600/64, softmax 10), LUT 2,565,431 (148% VU13P),
   > FF 1,997,539, BRAM 18. Folding cuts DSP 63× but LUT only 1.7× — the
   > operand-routing muxes (~1.2–1.35M LUT per einsum) dominate, so the
   > large-model core still exceeds the device even folded; the small-model
   > (D32/H4/E8) core is 8× smaller. Both attention-core rows are real.

3. **The deployable single-model table (r6-small).** No era-2 D32/L2 checkpoint
   exists; training YAMLs are validated but unlaunched (needs Kai's approval).
   Blocked poster items: single-FPGA utilization %, single-model latency in µs at
   an L1-defensible operating point, monolith DSP count per precision.

4. **"Russell-confirmed" truncation attribution.** Not documented anywhere in the
   repo: the top-10-by-pT decision is recorded as **advisor guidance relayed by
   Kai (decisions.md, 2026-07-01)**; the only Russell artifact concerns the STE
   bug. The draft says "advisor guidance". *To close: Kai confirms Russell = the
   advisor (or captures the confirmation), then the attribution can be named.*

5. **Whole-model HGQ2 hls4ml conversion / C-sim bit-exactness.** Marked
   `[blocked: convert not run]` in the store for all three rebuilds. The draft
   states rebuild fidelity at score level only.

6. **A4 substitution-cost quantifiers.** The statements "dynamic-mode correlation
   0.987 at A4" and the E2/E4 per-calibration ΔAUC ladder come from gold-model
   experiments on a 20k subset, recorded in LEDGER/RESEARCH but not re-derivable
   from raw arrays in the store. The draft uses the qualitative claim only. *To
   close: re-run the gold-model A4 dynamic-vs-static scoring on the full 260k and
   store the scores npz.*

7. **Large-model head count.** "H8" appears only in an era-1 speaker-notes label;
   the era-2 code default is H4 with env override, and the store does not record
   the era-2 large run's head count explicitly. The draft says "multi-head"
   without a number. *To close: read it off the r5 checkpoint config and cite.*
   > **CLOSED 2026-07-05 (live session).** `"n_heads": 8` read directly from
   > the `model_config` h5 attribute of all three era-2 large checkpoints
   > (`models/r5/large-lr05-{s1,a6-s1,a4-s1}/bitnet/noNorm_train_bitnetJetTagModel.h5`).
   > The era-2 large model is H8 (E=32); the draft may say "8-head".
   > *Poster side (2026-07-05): re-verified from all three checkpoints' h5
   > `model_config` (n_heads=8); the draft now says "8-head".*

8. **FastML26 poster logistics.** No poster size/orientation/template requirement
   is documented in the repo; figures were built vector (PDF/SVG) so they rescale
   to any format. *To close: Kai supplies the venue spec.*

9. **Seed spread.** The tradeoff table quotes single-run (s1) trained references,
   matching the store's own convention; second seeds exist and reconcile (A8
   0.8452, A6 0.8220, A4 0.7312) and ebops.md separately quotes a seed-avg 0.8501
   for W1A8. If the poster wants error bars, decide the convention (s1 vs
   seed-avg) once, everywhere.
   > **Updated 2026-07-05 (post adversarial review):** the draft and figures now
   > explicitly disclose the s1 convention, quote the s2 values, and note that
   > the A6 seed spread (1.7 pts) is as large as the quoted A8→A6 step.

10. **Era-2-shape synthesis for the QKeras-path composition.** The adversarial
    review established (from the raw census files) that the per-shape probes, the
    binary FFN block, the 1,049-DSP census, the 515.8%-LUT composition, and the
    23,409-cycle bound are **era-1-shape builds** (input_proj 14→256, head
    256→1); 49/51 instances are shape-identical in era 2. All are now labeled
    era-1-shape in the draft/figures (VERIFICATION.md §6.1). *To close: r6-small
    monolith synthesis (preferred), or re-run the two differing shape probes
    (16→256, 256→5) at era-2 shapes on mulder.*

11. **`probe_bitlinear_rf256` firmware/report build mismatch.** The stored
    firmware for that probe is a later Latency (v4) build; the stored csynth.xml
    is the earlier v3 Resource build (β̃-in-weights, no affine module) — they are
    not the same build (VERIFICATION.md §6.2). The 270-DSP composition is taken
    from the self-describing csynth.xml module table. *To close (live session):
    archive the v3 project (or re-tar the matching firmware) on mulder so the
    store holds a consistent artifact pair, and/or re-synthesize the v4 chain at
    RF=256 Resource for a true controlled strategy comparison.*
    > **CLOSED (consistency) 2026-07-05 (live session).** Mulder's
    > `probe_bitlinear_rf256` project (the Resource build that produced the
    > stored csynth.xml) was tarred as-built and pulled home: the store dir +
    > tarball now hold the matching Resource firmware (defines.h md5 34bdc118…)
    > beside the Resource reports — a consistent pair. The displaced Latency
    > firmware was verified byte-identical (md5) to mulder's
    > `probe_bitlinear_v3lat_rf256` ship tarball and now lives in the store
    > under that probe's own dir (no csynth report there — that build crashed,
    > HLS 200-1715; a stale contaminated report inside the old ship tarball was
    > deleted again on extraction). The optional controlled v4-chain RF=256
    > Resource re-synthesis remains open; note `probe_bitlinear_v2_rf256`
    > (pure-±1 + affine at Resource: dense 256 / SubLN 14 / affine 0) already
    > brackets the strategy effect on a nearly-identical chain.
