# Experiment Log — BNJetTag

Append-only. **Newest entries on top.** One entry per experiment or verification pass.
Written by `lead-pm`, `ml-engineer`, and `results-analyst`.

Format:

```
## YYYY-MM-DD — <experiment / check>
- Goal:
- Change / job file:
- Result (recomputed):
- Verified by results-analyst: ✓ / ✗
- Next:
```

## 2026-07-04 — VERIFICATION PASS ✓ — HGQ2 rebuild numbers (W1A8/W1A6/W1A4) gated for reporting
- Goal: independent results-analyst gate on `results/hgq2/tradeoff_table.md` + the three
  run payloads before anything reaches RESEARCH.md. Recomputed from raw arrays, not JSONs.
- Method: for each run (b224a8ea=W1A8, a428e6e2=W1A6, 53b202bc=W1A4) loaded
  `results/hgq2/runs/<hash>/scores.npz` + reference `roc-results/r5/{A8,A6,A4}-lr05-s1.npz`;
  checked `y` arrays exactly equal (row alignment ✓ all three, n=260,000); recomputed
  macro-OvR AUC (sklearn per-class OvR, float64, mean of 5) for both score sets + Pearson
  corr of flattened scores; summed ebops.json per_layer; read raw csynth_report.json files.
- Result: **all recomputed numbers match — verify.json gate1_vs_trained payloads match to
  full float precision; tradeoff_table.md rows match at quoted rounding.**
  * W1A8: ref 0.855104 / rebuild 0.849328 / Δ −0.005776 / corr 0.958927 ✓
  * W1A6: ref 0.839434 / rebuild 0.822153 / Δ −0.017281 / corr 0.894436 ✓
  * W1A4: ref 0.734561 / rebuild 0.711487 / Δ −0.023074 / corr 0.678891 ✓
  * EBOPs totals 650,360,941 / 501,183,824 / 352,214,162 ✓ (= per_layer sums exactly)
  * Reference AUCs match era-2 canon `roc-results/r5/roc_auc.md` (s1 rows + per-class) ✓;
    baselines line FP32 0.8765 / W8A8 0.8642 ✓
  * csynth quotes match raw JSONs ✓: probe_bitlinear_rf256 LUT 196,871 / FF 118,597 /
    DSP 270 / BRAM 32 / 832–833 cyc / II 573 / 3.035 ns; probe_subln_rf1 LUT 165,695 /
    FF 151,297 / DSP 1,792 / BRAM 0 / 36 cyc / II 1 / 1.818 ns (VU13P, target 2.5 ns)
  * W1A8 per-class Δ (rebuild − trained), g/q/W/Z/t: −0.0031 / −0.0023 / −0.0129 /
    −0.0088 / −0.0017 → worst class (W) −0.0129 < 0.02 flag threshold — no per-class
    pathology hidden under the small macro Δ. (Informational: at W1A6/W1A4 the W and Z
    classes degrade > 0.02 — −0.034/−0.030 and −0.038/−0.028 — consistent with the larger
    macro Δ already quoted there.)
- Notes: (1) the preliminary numbers in the earlier 07-04 entry below (0.84868, Δ −0.0064)
  were a mid-session calibration state; the final artifacts say 0.84933 / Δ −0.0058 — the
  tradeoff table uses the final values, which are what is verified here. This entry closes
  that entry's "pending" verification. (2) verify.json gate2_vs_gold pass:false is
  informational-tolerance only (LN eps/float order); table correctly marks hls4ml C-sim
  bit-exactness as blocked/not-run, so no bit-exact claim is being made. (3) DSP 270 (A8
  bitlinear probe) and DSP 1,792 (SubLN II=1) are correctly quoted but are NOT the 0-DSP
  binary-core result — keep the §Real synthesis caveat attached wherever these are cited.
- Verified by results-analyst: ✓
- Next: safe to cite tradeoff_table.md numbers in RESEARCH.md; C-sim bit-exact column
  stays [blocked] until hls4ml convert runs.

## 2026-07-04 — HGQ2 rebuild of era-2 large W1A8: keras-side verification + first csynth
- Goal: rebuild the binary {−1,+1} transformer in HGQ2 (hls4ml-native path), port the
  round-5 trained weights, verify fidelity BEFORE synthesis (directive of 2026-07-04).
- Change / job file: new pipeline `code/hgq2/` (configs → extract → calibrate → build →
  verify → ebops → convert), custom SubLN hls4ml extension, fold-aware CSD-2 β handling.
- Result (recomputed, full 260k era-2 val, vs verified roc-results/r5/A8-lr05-s1.npz):
  HGQ2 rebuild macro-OvR AUC **0.84868** vs trained 0.85510 (Δ −0.0064), score corr
  0.958. Numpy gold model in exact-QKeras mode reproduces the stored scores at corr
  0.99995 (architecture proof). All 51 BitLinears binarize to exactly {−1,+1}
  (0 sign-zeros, all 3 checkpoints). First mulder csynth of the HGQ2 path:
  SubLN dim-256 II=1 → LUT 165,695 / FF 151,297 / DSP 1,792 / BRAM 0 / 36 cyc @ 1.82 ns.
- Verified by results-analyst: pending (final pass at session end)
- Next: A6/A4 sweep, EBOPs, attention-core + bitlinear csynth on mulder, tradeoff table.

---
<!-- new entries go below this line, newest first -->

## 2026-07-03 — PVC INCIDENT ROOT-CAUSED: NRP-wide rook-cephfs backend outage, NOT our config; still ongoing
- Goal: Kai believed the kai-data PVC recovered — verify with a live mount test and find the
  actual root cause of the 2026-07-02 wedge.
- Method: fresh minimal test pod `kai-pvc-test` (ubuntu + kai-data mount, 1 CPU/2Gi);
  landed on a 5th node / 4th site never tried before (yge-nrp-01.kreonet.net).
- Result: **still wedged, identical signature** — AttachVolume succeeds, then
  MountVolume.MountDevice `DeadlineExceeded` (~2 min hang), then every retry
  `Aborted: an operation with the given Volume ID already exists`.
- **Root cause found (smoking gun):** two OTHER users' pods in cms-ml are failing right now
  with the exact same error on DIFFERENT volumes — `tim-moe-inter` → `moe-interpretability-pv`
  (vol `…rook-0000000000000001-2cd8ea0b…`, ContainerCreating 5 h) and
  `tn-part-serverdep-…` → `tn-pvc-base-jetclass2` (vol `…rook-0000000000000001-0f1726c8…`).
  All three (incl. kai-data `…rook-0000000000000001-9f276e98…`) are on ceph cluster
  `rook-0000000000000001` = the `rook-cephfs` storage class. So: **backend-wide CephFS
  NodeStageVolume outage since 2026-07-02**, not volume-specific, not our stale pods, not
  our YAMLs. Existing mounts unaffected (multi-day zh-dino trainings still running) —
  only NEW mounts fail. `rook-cephfs-central` is a different backend and is healthy
  (ajd-inspect-pvc mounted j-jepa-vol fine).
- Per NRP admin docs (nrp.ai/documentation/admindocs/storage/volume-mounting/), this error
  class is known and the fix (restart csi-cephfsplugin / clear stale CSI state / reboot
  nodes) is **admin-only** → must be reported to NRP support. Deleting our completed pods
  (yesterday's hypothesis (a)) would NOT have helped — different users/volumes are hit.
- State: `kai-pvc-test` left retrying as a recovery sentinel (monitor watching; goes
  Running the moment the backend heals). Support-report text drafted for Kai to post.
- Verified by results-analyst: n/a (infra, no numbers).
- Next: Kai posts the report to NRP Matrix/support; on recovery → delete kai-pvc-test,
  resume normal PVC-based job flow (the PVC-free W&B/Zenodo path from 07-02 remains the
  proven fallback).

## 2026-07-03 — ERA-2 ROC-TEST VERIFIED ✓ — round-5 complete; the era-2 table exists
- Goal: verify-roc gate on the 8 era-2 `.npz` from `kai-roc-r5-pvcfree` (Succeeded 03:08 UTC;
  ~14 min/model on CPU). Files: `roc-results/r5/*.npz` (+ roc_auc.md, roc_overlay.png),
  fetched from W&B run `r5-roc-pvcfree-artifacts`.
- Method (per `.claude/skills/verify-roc/`): recomputed macro one-vs-rest AUC + 5 per-class
  AUCs from every npz (`roc_auc_score(y, s, multi_class="ovr", average="macro")`, keys
  y/score, one-hot checked), n = 260,000 each (the dataset's own val split — first era-2
  ROC-test, so this n is the era-2 canon; do not compare with era-1's 222,912).
- Result: **all 48 recomputed numbers match roc_auc.md exactly** (4 decimals). ✓
  ERA-2 ROC-test macro-OvR (labels: era 2 · ROC-test · seed-avg where two seeds exist):
  * FP32-lr05 (single-run baseline)  **0.8765**
  * W8A8-lr05 (single-run baseline)  **0.8642**
  * W1A8 seed-avg **0.8501** (s1 0.8551 / s2 0.8452)
  * W1A6 seed-avg **0.8307** (s1 0.8394 / s2 0.8220)
  * W1A4 seed-avg **0.7329** (s1 0.7346 / s2 0.7312)
- Headlines this establishes (era-2 only): 1-bit costs **−2.64 macro-AUC pts vs FP32**
  (−1.41 vs W8A8); activation axis A8→A6 −1.94 pts, A6→**A4 −9.78 pts (the cliff)** — era-1's
  "A4 holds up" did NOT survive the dataset migration. Per-class: top-tagging is the most
  robust class everywhere (W1A8 avg ~0.893); gluon the weakest.
- Note on baselines: era-2 has only the lr05 baselines (no "original-recipe" era-2 run), so
  "baseline = stronger of the two" is trivially the lr05 run; both singles-run — flag if a
  seed-pair for baselines is ever wanted.
- Verified by results-analyst: ✓ (recomputation performed in-session per the skill; exact
  match, zero discrepancies).
- Next: era-2 table → RESEARCH.md §5; EBOPs×AUC positioning → results/ebops.md; commit.

## 2026-07-02 — PIVOT: PVC-free ROC job launched (kai-roc-r5-pvcfree) — routes around the ceph incident
- Goal: get the era-2 ROC-test done without the wedged kai-data PVC (per Kai: "the job is
  never going to run — try another way").
- Key discovery enabling this: `qkerasModel.py` line ~980 does `wandb.save(<checkpoint>)` —
  **all 8 round-5 best-epoch checkpoints (76.9 MB each, `restore_best_weights=True`) are in
  W&B**, verified via the API (all runs `finished`, one `*_bitnetJetTagModel.h5` per run).
  The eval data is public (Zenodo 3602260, `hls4ml_LHCjet_150p_val.tar.gz`, 1.14 GB) — so
  nothing needed from the PVC.
- New job `code/jobs/training/kai-roc-r5-pvcfree.yaml`: same embedded `make_roc.py`
  (reuses ConfigMap `kai-roc-script-r5` — byte-identical), same BN_* env, same eval labels;
  qkerasModel.py shipped via new ConfigMap `kai-qkerasmodel-r5` (created from local copy,
  **md5 4d0a3fed… = the exact code that trained**); checkpoints pulled from W&B; val split
  pulled from Zenodo; outputs (.npz, roc_auc.md, overlay png) uploaded to W&B run
  `r5-roc-pvcfree-artifacts` as the durable copy (PVC replaced by emptyDir).
- Also recovered from W&B (last-epoch summary val_auc — still UNVERIFIED, and the saved
  checkpoints are best-epoch, so ROC-test may exceed these): fp32 0.8486 · w1a8-s1 0.8465 /
  s2 0.8298 · a6-s1 0.8322 / s2 0.7995 · a4-s1 0.7126 / s2 0.7156 · w8a8 0.8378.
- Stuck `kai-roc-r5` job deleted (never mounted; 39+ aborted mount ops over ~70 min).
  Admission-policy note: cms-ml enforces limit/request ratio ≤1.2 — resources set 8/9 CPU,
  24/28Gi mem.
- Verified by results-analyst: ⧗ pending (verify-roc gate runs once the .npz come home
  from W&B).
- Next: monitor job → download npz from W&B → `roc-results/` → verify-roc → era-2 table.

## 2026-07-02 — INFRA INCIDENT: kai-data PVC unmountable cluster-wide; kai-roc-r5 blocked
- Goal: run `kai-roc-r5` (era-2 ROC-test). Blocked by infrastructure, not by our configs.
- Symptom: every new pod mounting PVC `kai-data` (volume
  `pvc-c5be55ec-3362-40a9-b283-f8d0750dcf95`, CSI vol ID `0001-0004-rook-0000000000000001-
  9f276e98-3be5-415d-b665-c05c6036f24c`) hangs in ContainerCreating: first
  MountVolume.MountDevice attempt → `DeadlineExceeded`, all retries → `Aborted: an operation
  with the given Volume ID already exists`. Reproduced on **4 nodes / 3 sites** over ~2.5 h:
  emporia.gp-argo, sdsmt.gp-argo, gp-argo.usd.edu, gpengine-uams.areon.net. The last of
  these mounted this exact volume successfully ~6.5 h earlier (kai-readlogs), so the
  degradation began 2026-07-02 within that window. Initial "gp-argo site is wedged"
  hypothesis (job recreated with the 6 gp-argo nodes excluded via nodeAffinity NotIn,
  scratchpad-only YAML patch) was falsified when the off-site node wedged identically →
  volume/ceph-backend level, not site level.
- Tried: node exclusion (above); deleting our stale Completed pods to force clean unstages —
  denied by the session permission layer (pre-session pods; Kai's call). VolumeAttachment
  objects not visible at our RBAC level.
- State: `kai-roc-r5` (pod `kai-roc-r5-2xpxr`) left in place, retrying — if/when the
  backend recovers, the mount completes on its own and the job runs with no further action.
- Next: (a) Kai may try deleting our completed pods (`kai-bn5-*`, `kai-preflight-r5`,
  `kai-readlogs`, `kai-util`) to release possibly-stale ceph client sessions; (b) if still
  wedged, report to NRP support with the volume ID + symptom above; (c) on mount success →
  fetch `.npz` → verify-roc → era-2 table.

## 2026-07-02 — Verification pass: ebops.py era-2 + results/ebops.md
- Goal: gate the new EBOPs work (marks the "ebops.py fixed for era 2" entry below as verified).
- Method: ran all four modes of `code/training/ebops.py` (default era-2 large, `--era 1`,
  `--size all`, `--size all --era 1`); independently recomputed params, MACs, and every
  EBOPs total/ratio from first principles in a throwaway python (not reusing ebops.py); and
  cross-checked `results/ebops.md` §1–§3 line-by-line against script output + research-log.
- Result (recomputed):
  - Params: era-2 large **6,375,173** [OK] in-code assertion; era-1 large **6,373,633** [OK];
    era-1 all four sizes match known-good (tiny 26,529 / small 153,793 / medium 808,065 /
    large 6,373,633, all OK). Hand closed-form (input_proj 4,352 + 8×787,968 + head_fc1
    65,792 + head_fc2 1,285) = 6,375,173 ✓.
  - MACs/jet 63,431,936 = matmul 63,022,336 (99.35 %) + attn 409,600 (0.65 %) ✓; fc1 spot-check
    10·256·1024 = 2,621,440 ✓.
  - EBOPs totals (era-2 large) all reproduced exactly: FP32 64,954,302,464 · W8A8 4,059,643,904 ·
    W1A8 530,393,088 · W1A6 392,879,616 · W1A4 258,642,944. Ratios: 7.65× (W8A8/W1A8), further
    2.05× (W1A8/W1A4), 0.131×/0.097×/0.064× vs W8A8, 0.062× (W8A8/FP32), 16.0× (FP32/W8A8) — all ✓.
  - §3 literature numbers all traceable to research-log 2026-07-02: HGQ 71.0–76.4 % acc,
    LUT 0.02–0.53 %, DSP 0–0.5 %, 10–30 ns; sub-µs transformers EBOPs-target 350k, LUT 47k–202k,
    DSP 0, acc 77.9–79.8 %, 44–78 ns (the 64-particle MHA/Linformer configs); JEDI-net ~34k
    params, per-class AUC 0.93–0.97. Scale-gap ~1,500× = 530,393,088/350,000 = 1,515.4 ✓;
    attn term 26.2M EBOPs at A8 = 4.94 % of W1A8 total ✓. No untraceable number found.
- Verified by results-analyst: ✓ — ebops.py + results/ebops.md §1–§3 PASS; clears the
  "⧗ pending" on the "ebops.py fixed for era 2" entry below. (No era-2 *accuracy* verified —
  none exist yet; round-5 ROC still in flight.)
- Next: put era-2 AUC next to the EBOPs axis after round-5 verify-roc lands.

## 2026-07-02 — Round-5 training COMPLETE (all 8 jobs); kai-roc-r5 launched
- Goal: confirm round-5 finished on the HLS4ML 5-class data and start the era-2 evaluation.
- Result: all 8 `kai-bn5-*` jobs **Complete 1/1** on Nautilus (runtimes 58 m–3 h 43 m,
  finished ~2026-07-02 early UTC). Partial last-epoch val AUCs recovered from surviving pod
  logs (W&B summary lines — **last-epoch, not best-checkpoint; single-seed; UNVERIFIED**):
  w8a8 0.83782 (ep 32) · w1a8-s2 0.82982 (ep 29) · a6-s2 0.79955 (ep 20) · a4-s1 0.71256 ·
  a4-s2 0.71562. fp32/s1/a6-s1 pod logs already GC'd — their numbers come via W&B/PVC.
  Early signal (needs ROC-test confirmation): the A4 cliff looks much steeper on era-2 data
  than it did in era 1.
- Launched `kai-roc-r5.yaml` (server dry-run validated first). Infra friction: the kai-data
  PVC hit a rook-ceph mount wedge ("operation with the given Volume ID already exists") on
  two different nodes (~10+ min in ContainerCreating) — attach succeeds, MountDevice is
  slow/queued; pods left to retry. Same symptom on the checkpoint-verification util pod
  (`kai-util3`, repinned off emporia — no cure, so it is volume-level, not node-level).
- Verified by results-analyst: ⧗ pending — no era-2 number is headline-eligible until the
  ROC `.npz` come home, AUCs are recomputed (verify-roc), and seeds are averaged.
- Next: monitor kai-roc-r5 → fetch `/data/outputs/roc-r5/*.npz` → verify-roc → start the
  era-2 table in RESEARCH.md §5.

## 2026-07-02 — hls4ml-on-mulder verification: chain WORKS; 2 env gotchas (1 new, documented)
- Goal: prove the hls4ml→Vitis path on mulder is alive end-to-end (user request), fix breakage.
- Method: fresh smoke test in `~/bnjet_smoketest` — QDense binary(16→32, RF=8) → hls4ml
  convert → compile (C-sim) → real Vitis csynth, env `bnjet` (tf 2.11.1, hls4ml
  1.4.0.dev7+ga3064d50), Vitis 2023.2.
- Result: **PASS.** convert OK · compile OK with C-sim fidelity corr = 1.0 · csynth OK:
  binary dense at **DSP=0**, LUT 7,253, FF 2,398, latency 5 cycles (VU13P). Thesis mechanics
  (binary → LUT, no DSP) reconfirmed on today's toolchain.
- Gotcha 1 (known, documented): TF import dies on protobuf 4.25.3 unless
  `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (already in RUN_CSYNTH_ON_VITIS.md +
  `full_model_csynth.py`; NOT a new breakage — protobuf installed 05-31, June runs fine).
- Gotcha 2 (NEW): hls4ml 1.4.0.dev's Vitis backend launches csynth via **`vitis-run`**, which
  only exists in the FULL Vitis install — `source /data/software/xilinx/Vitis/2023.2/settings64.sh`,
  not the Vitis_HLS-only path the runbook gave (June runs worked because full Vitis was on
  PATH; see `out/csynth_a8.log` banner). **Fixed:** dated correction added to
  `code/hls/RUN_CSYNTH_ON_VITIS.md` §3.
- Verified by results-analyst: ✓ (numbers read directly from
  `~/bnjet_smoketest/prj/myproject_prj/solution1/syn/report/csynth.xml` on mulder).
- Next: nothing blocking synthesis; next mulder run is the era-2 checkpoint after round-5 verify.

## 2026-07-02 — ebops.py fixed for era 2; EBOPs table computed; literature comparison written
- Goal: get EBOPs working for the model we actually train now (user request) and place it
  against published taggers on the same dataset.
- Change: `code/training/ebops.py` was still era-1-only (N_FEAT=14, 1-logit head). Added era
  support (`--era`, default 2 = 16 feat × 10 particles, 5-class head), era-keyed known-param
  assertions. Era-2 large reproduces the round-5 preflight count **6,375,173 exactly**; era-1
  still reproduces 6,373,633 (all four sizes OK).
- Result (era-2 large, per jet): MACs 63,431,936 (99.35 % weight×act matmul / 0.65 % attn
  act×act). EBOPs: W8A8 4.060G · **W1A8 530.4M (7.65× below W8A8)** · W1A6 392.9M · W1A4
  258.6M. Written up with the published-tagger comparison (HGQ Table I, sub-µs transformers
  EBOPs-target 350k, JEDI-net) + honest caveats (≈1,500× scale gap, input/metric mismatches,
  DSP=0 not novel per se) in **`results/ebops.md`** (new). Literature table + sources:
  research-log 2026-07-02 entry (physics-researcher).
- Verified by results-analyst: ⧗ pending (this entry written by main session; verification
  pass queued).
- Next: after round-5 verify-roc, put the era-2 accuracy numbers next to the EBOPs axis
  (AUC-vs-EBOPs is the plot the PI will want).

## 2026-07-02 — Backfill committed (52cb818) + propagated to RESEARCH.md §6/§7/§8
- Goal: land the approved follow-ups on the A6/A4 backfill (entry below).
- Change: run-repo commit **52cb818** "Backfill A6/A4 full-model csynth; compose whole-model
  latency upper bound" — 17 files (16 `results/csynth/full_model_*_a{4,6}_rf256*.json` +
  `results/hls_resource_table.md`). Verified no secrets staged. RESEARCH.md updated: §6 adds
  DSP=1,049 precision-independence + composed 23,409-cycle ≈58.5 µs @400 MHz target
  (≈44.5 µs achieved clock) upper bound with both caveats (attention score core excluded;
  folded RF=256 extreme); §7 promotes both to Established, narrows open Q4, sharpens Q5;
  §8 adds the source row. All numbers quoted from `hls_resource_table.md` §B′ only.
- Verified by results-analyst: ✓ (underlying numbers verified in the entry below; this entry
  is propagation only — no new numbers introduced).
- Next: **push is pending** — sandbox has no SSH key; Kai runs
  `git push origin main round-5` in `qkeras-bitnet-run-2026-06-22/`. Round-5 still in flight.

## 2026-07-02 — A6/A4 full-model csynth BACKFILL fetched + integrated; whole-model latency composed
- Goal: close the pending A6/A4 full-model csynth backfill (`code/hls/BACKFILL_A6_A4_FULL_MODEL.md`)
  and answer the "what's the whole-model latency" question §B′ had been punting on.
- Finding: the A6/A4 sweep had in fact COMPLETED on `mulder` 2026-06-26 19:55 (rc=0) — it was
  never fetched. Data lived at `~/bnjet_fullcsynth/out/` (combined `full_model_total_rf256.json`
  + `shapes_all_rf256.json`), NOT the path the runbook guessed. No re-synthesis needed.
- Result (recomputed from the JSONs): A6 total LUT 8,680,426 / FF 8,545,402 / **DSP 1,049** /
  BRAM 778; A4 LUT 8,594,614 / FF 8,324,652 / **DSP 1,049** / BRAM 778. Sanity anchors all PASS —
  DSP=1,049 at both (precision-independent, = A8), per-shape DSP identical across A8/A6/A4
  (11/15/15/51/15), LUT & FF monotone A8 ≥ A6 ≥ A4.
- Composed whole-model latency (new): critical-path sum of per-stage worst-case cycles (parallel
  Q/K/V counted once), 8 blocks → **23,409 cycles ≈ 58.5 µs @ 400 MHz** (≈44.5 µs at achieved
  ~1.90 ns/layer). Fully-spatial streamed *upper bound*; weightless attn score core excluded
  (0.65% MACs, EinsumDense unsupported); folded design trades latency up for area down.
- Files: `results/csynth/full_model_{total,shapes,shape_*}_a{6,4}_rf256.json` created (mirror A8
  naming). `results/hls_resource_table.md` §B′ updated (A6/A4 rows + latency subsection + placeholder
  note replaced). NOT yet committed to the run repo; NOT yet propagated to RESEARCH.md §6.
- Verified by results-analyst: ✓
- Next: propagate the composed latency + A6/A4 totals to RESEARCH.md §6; prioritize the
  LayerNorm-DSP-elimination experiment (100% of the 1,049 DSP is LayerNorm — see decisions.md).

## 2026-07-02 — Verification pass for the workspace restructure (RESEARCH.md numbers)
- Goal: confirm every number written into the new `RESEARCH.md` against its source before
  the doc goes live (see decisions.md 2026-07-02 for the restructure itself).
- Result (recomputed, era-1 ROC-test from `roc-results/*.npz`, sklearn `roc_auc_score`):
  A8 claimed 0.7986 → 0.7986 ✓ · FP32 claimed 0.8207 → 0.8207 ✓ · A4 claimed 0.7886 →
  0.7886 ✓ (n=222,912 each, keys `y`/`score`). Val-AUC and HLS numbers quoted with sources
  (run README table; `results/hls_resource_table.md` §A/§B/§B′). All paths referenced by
  the new docs/skills verified to exist; no dangling references to pre-restructure
  locations remain.
- Verified by results-analyst: ✓ (procedure per `.claude/skills/verify-roc/SKILL.md`)
- Next: round-5 completes → `kai-roc-r5.yaml` → era-2 table starts in `RESEARCH.md` §5.

## 2026-07-01 — LAUNCHED round-5 sweep on NRP (HLS4ML LHC Jet dataset, all 8 jobs)
- Goal: run + finish the round-5 quantization sweep on the migrated dataset, best GPUs.
- Pre-launch fixes (both were pending per prior log entries):
  * Synced Jul-1 `qkerasModel.py` (md5 4d0a3fed…) + `make_roc.py` to `/data/BNJetTag`
    on the kai-data PVC — the PVC copy was the Jun-28 pre-migration version and
    `make_roc.py` was absent. (Flaky link → gzip+md5-verified transfer.)
  * Updated cms-ml secret `kai-wandb` with the rotated key (verified by sha256; jobs'
    `wandb.init()` is un-try/excepted so a stale key would crash every run).
- Preflight: ran `preflight_r5.sh` in a CPU TF pod → **PREFLIGHT_ALL_PASS**; dataset
  `jetConstituentList (10000,150,16)`, all 8 configs build, **6,375,173 params** (new
  16-feat / 5-class count — NOT the old 6,373,633).
- Launch: applied all 8 `kai-bn5-*` jobs to ns `cms-ml`. **Deviation from the staged
  ≤3 launcher**: launched all-at-once (per Kai, so the laptop can close) — the local
  `launch_r5_staged.sh` was killed so it can't delete/recreate jobs mid-train. Cluster
  scheduler self-limits concurrency (only fp32 Running, 7 Pending on GPU availability).
- Infra note: ns quota **h100/h200/gh200 = 0/0 (banned)**, a100 16/22 used — so despite
  YAML affinity preferring H200/H100, jobs land on A100 / L40S / 4090 / A40, not H100/H200.
- W&B: project `bnjettag-bitnet`, runs `r5-{large-lr05-s1,s2, -a6-s1,s2, -a4-s1,s2, fp32-lr05, w8a8-lr05}`.
- Verified by results-analyst: ⧗ pending (no AUCs yet — training in flight; ROC via `kai-roc-r5.yaml` after).
- Next: let all 8 finish; run `kai-roc-r5.yaml` for ROC-test AUCs; seed-average before any headline.

## 2026-07-01 — W&B ROC curves: x-axis = tagging efficiency (HEP convention)
- Goal: per Kai — ROC x-axis must be efficiency. The matplotlib overlay already plotted
  x=TPR (efficiency) / y=mistag (log); only the W&B `wandb.plot.line` curves had x=fpr.
- Change: `make_roc.py` `_wandb_log_roc` — table columns now (efficiency, mistag_rate),
  x=efficiency; re-embedded into `kai-roc-r5.yaml` (byte-identical). py_compile OK.
- Also: W&B key rotated by Kai; new key stored in `wandb-api-key.txt` (chmod 600, not
  committed). NRP secret `kai-wandb` still needs updating from that file.
- Verified by results-analyst: ✓ (compile + embed check)

## 2026-07-01 — Input revised: top-10 constituents by pT × 16 feats (supersedes top-32)
- Goal: apply the advisor's guidance (top 10 particles by pT, high→low; input-size effects
  on latency/resources deferred to a later study) with Kai's choice to keep ALL 16 features
  (10×16 = 160 inputs, not 10×14=140). See decisions.md 2026-07-01 (top-10 entry).
- Change / files (all LOCAL prep — nothing launched):
  * `code/training/qkerasModel.py` — `N_PART_PER_JET` default 32→10 (BN_N_PART knob kept);
    docstring/diagram now (batch,10,16).
  * `code/training/make_roc.py` — `N_PART` default 32→10 (mirrors trainer).
  * `code/jobs/training/variants/gen_round5_jobs.py` — BN_N_PART=10 in JOB_TMPL + preflight
    env; docstring top-32→top-10. RERUN → 8 `kai-bn5-*.yaml` + `preflight_r5.sh` +
    `launch_r5_staged.sh` regenerated.
  * `code/jobs/training/kai-roc-r5.yaml` — BN_N_PART=10; make_roc.py re-embedded
    (`reembed_make_roc.py`, byte-identical check).
  * `README.md` migration blockquote → top-10 × 16 (160 inputs).
- Result (recomputed): verification suite re-run — py_compile + AST asserts, bash -n,
  all YAMLs parse with BN_N_PART=10 (zero `BN_N_PART=32` remaining anywhere), synthetic-h5
  loader smoke test X=(200,10,16), trainer/eval loaders byte-identical.
- Verified by results-analyst: ✓ (checks above)
- Next: Kai runs preflight_r5.sh on NRP → launch_r5_staged.sh; roc job after each wave.

## 2026-07-01 — DATASET MIGRATION executed: HLS4ML LHC Jet (150p), 5-class, top-32×16
- Goal: per Kai's directive, point ALL training/eval code + YAMLs at the public HLS4ML LHC Jet
  dataset (150 particles) already on the PVC at `/data/hls4ml_lhc_jet/` (see decisions.md
  2026-07-01 for the choices: 5-class g/q/W/Z/t, top-32 constituents by pT, 16 feats).
- Change / files (all LOCAL prep — nothing launched):
  * `code/training/qkerasModel.py` — NEW `load_hls4ml_jets(dir)` (globs `*.h5`; reads
    `jetConstituentList`/`jets`; labels + pT columns located BY NAME via
    `jetFeatureNames`/`particleFeatureNames`; re-sorts constituents by pT desc before top-N
    truncation); N_FEAT=16, N_PART_PER_JET=env `BN_N_PART` (default 32), N_CLASSES=5; head
    `BitLinear(5)`; loss CategoricalCrossentropy(from_logits); metric
    `AUC(multi_label=True, num_labels=5, from_logits=True)` = macro-OvR "val_auc" (ES knobs
    unchanged); CLI now ONE positional arg (TrainDir); dropped IP-norm + sig/bkg pT-reweighting
    + kinematics_plotter (old-format-specific); output name `bitnet/noNorm_train_*` kept so
    find_model/hls tooling still matches; --sanity updated to (8,32,16)→(8,5).
  * `code/training/make_roc.py` — rewritten for the new set: `--data` dir (val/val) replaces
    `--sig/--bkg`; softmax scores; per-class OvR AUC + macro; per-class W&B curves/summaries;
    plot = 2×3 per-class panels (log-y mistag) + macro panel; table has AUC(g/q/W/Z/t)+macro.
  * `code/jobs/training/variants/gen_round5_jobs.py` — JOB_TMPL data args → single
    `/data/hls4ml_lhc_jet/train/train`, exports `BN_N_PART=32`; docstring caveats (old-dataset
    AUCs not comparable; LR 5e-5 carried over as assumption); preflight_r5.sh gained a dataset
    check (files exist + h5 keys + label names in BOTH train/ and val/). All 8 `kai-bn5-*.yaml`
    + preflight_r5.sh + launch_r5_staged.sh REGENERATED.
  * `code/jobs/training/kai-roc-r5.yaml` — `DATA=/data/hls4ml_lhc_jet/val/val` (true held-out
    ROC-test set); old-data r4 large-lr05 entry REMOVED (10×14/1-logit checkpoint is
    format-incompatible); make_roc.py re-embedded via reembed_make_roc.py (byte-identical ✓).
- Result (verified locally): 8 YAMLs parse + contain new path/`BN_N_PART=32`/no `/data/bnjet`;
  kai-roc-r5.yaml = 2 docs, embed byte-identical, r4 dropped; py_compile + bash -n pending in
  final verification pass (same date, below in this entry's checklist).
- Verified by results-analyst: ✓ static checks + synthetic-h5 loader smoke test (see below).
- Next (Kai, on NRP): run `preflight_r5.sh` (now includes the data check) → if
  `PREFLIGHT_ALL_PASS`, `launch_r5_staged.sh`; apply `kai-roc-r5.yaml` after each wave.
  NOTE: round-5 numbers start a FRESH 5-class table — do not compare to any pre-migration AUC.

## 2026-07-01 — ROUND-5 prepared (quantization round) + ROC-r5 re-eval job + W&B ROC logging
- Goal: execute resolutions for every open item in the logs: (1) large@5e-5=0.7672 is single/unseeded;
  (2) A6/A4 thesis axis only ever measured under the too-hot lr15; (3) FP32/W8A8 baselines never
  LR-tuned (tuned-vs-untuned trap); (4) no ROC-test AUC for any tuned model; (5) ROC curves not in
  W&B; (6) A6/A4 full-model csynth backfill never collected from mulder.
- Change / job files (all LOCAL prep — nothing launched):
  * NEW `code/jobs/training/variants/gen_round5_jobs.py` → 8 `kai-bn5-*.yaml` (all FIXED large
    D256/L8/H8/F1024, peak LR 5e-5): large-s1/s2 (seed-confirm), a6-s1/s2 + a4-s1/s2 (tuned quant
    axis), fp32 + w8a8 (fair baselines; report max(old recipe, 5e-5) per variant) + `preflight_r5.sh`
    + `launch_r5_staged.sh` (staged ≤3 concurrent, per round-4 constraint).
  * NEW `code/jobs/training/kai-roc-r5.yaml` → ROC-test re-eval of r4 large-lr05 + all 8 r5 runs into
    `/data/outputs/roc-r5/` (skips missing checkpoints, so it can run after each wave); old
    `/data/outputs/roc` (lr15) kept untouched for before/after.
  * `code/training/make_roc.py`: added opt-in `--wandb` (per-run ROC curve + AUC, overlay + table);
    guarded so W&B failure never kills an eval. Re-embed helper: `code/jobs/training/reembed_make_roc.py`
    (fixed a re.sub backslash-expansion bug; embed now verified byte-identical to canonical source).
  * NEW `code/hls/BACKFILL_A6_A4_FULL_MODEL.md` — mulder runbook; step 0 = check if the 2026-06-26
    sweep already finished. Sanity anchor: composed DSP must be 1,049 at A6/A4 (precision-independent).
  * `README.md`: added clearly-labeled round-4 UPDATE note under the headline table (0.7672, single
    run, round-5 pending) — headline lr15 table left intact as the reproducible anchor.
- Result (verified): all 8 bn5 yamls + roc-r5 yaml parse (k8s YAML); embedded make_roc.py compiles and
  is byte-identical to canonical; bash -n passes on preflight/launcher/job script; every BN_* knob in
  the yamls confirmed present in qkerasModel.py; fixed a nullglob `ls` bug in roc_eval's checkpoint
  check (would have false-positived on missing checkpoints) → `find -print -quit`.
- Verified by results-analyst: ✓ (static checks above; no training numbers produced by this pass)
- Next: (1) preflight_r5.sh on NRP, then `launch_r5_staged.sh`; (2) apply kai-roc-r5.yaml after each
  wave; (3) run the mulder backfill runbook; (4) ROTATE the W&B key (still plaintext in the synced
  folder); (5) after round-5: update README/REPORT headlines from seed-averaged numbers only.

## 2026-06-29 — VERIFICATION: EBOPs profiler (`code/training/ebops.py`) — full PASS
- Goal: independently verify every EBOPs/param/MAC/AUC claim before any reaches REPORT.md.
- Method: ran `ebops.py --size large` and `--size all`; re-derived all numbers from scratch in
  a separate stdlib script (no import of ebops.py), per HGQ Eq.5 (arXiv:2405.00645): EBOPs =
  matmul_MACs*b_w*b_a + attn_MACs*b_a*b_a, accumulator term = 0 by definition.
- Result (all independently recomputed, both methods agree exactly):
  * Params: tiny 26,529 / small 153,793 / medium 808,065 / large 6,373,633 — match preflight. ✓
    (Confirms large = 6.37M; the old "~3M" doc label is WRONG.)
  * Attn MACs large = 2*L*N²*D = 2*8*100*256 = 409,600 = 0.65% of total. ✓
  * matmul MACs = 63,016,192 (99.35%); total = 63,425,792. ✓
  * EBOPs large: FP32 64,948,011,008 / W8A8 4,059,250,688 / W1A8 530,343,936 /
    W1A6 392,842,752 / W1A4 258,618,368 — all match by-hand. ✓
  * Ratios: W8A8/W1A8 = 7.654x (not 8x); FP32/W1A8 = 122.46x. ✓
    Attn floor identical W8A8 vs W1A8 = 409,600*64 = 26,214,400 → explains the 7.65x. ✓
  * Round-4 AUC ordering/arithmetic (validation AUC, numbers taken as given): large@5e-5=0.7672
    is sweep-best; +0.0142 over old large@1.5e-4=0.7530; size order monotonic; −0.0031 vs FP32
    val 0.7703. All arithmetic ✓ (NOTE: only checked arithmetic/ordering — did NOT re-extract
    the four AUCs from the PVC train logs, which are not in this local tree).
- Verified by results-analyst: ✓ (PASS on all 6 numbered items)
- Next: cleared for REPORT.md. Fix any "~3M" param label to 6,373,633.

## 2026-06-29 — Static MAC/param profile + EBOPs scaffold (`code/training/ebops.py`)
- Goal: accurate per-layer MAC + parameter profile of the FIXED main arch (D256/L8/H8/F1024) as
  the basis for an EBOPs hardware-cost metric. STATIC architectural count — no training run.
- Change / job file: NEW `qkeras-bitnet-run-2026-06-22/code/training/ebops.py` (stdlib only, no TF).
  Encodes closed-form per-layer MACs + params; `bops(macs,b_w,b_a)=macs*b_w*b_a`; `accumulator_bops()`
  is a 0-stub hook for the HGQ accumulator term (TODO, pinned in parallel).
- Result (recomputed, pure-python, NO heavy compute):
  - PARAM cross-check PASSES exactly for all four sizes: tiny 26,529 / small 153,793 / medium 808,065 /
    **large 6,373,633** (confirms the "~3M" doc label was WRONG; true large = 6.37M). Match requires the
    `learned` pos-Embedding to count 0 params (it folds to a fixed offset — matches `count_params()`).
  - large MACs/jet = **63,425,792**: matmul (weight*act) 63,016,192 (99.35%) vs attention (act*act)
    409,600 (0.65%). Attention MAC formula per block = N*N*D for Q.K^T + N*N*D for (softmax)A.V, N=10.
  - large BOPs vs FP32(=32x32 ref): W8A8 0.062x, W1A8 0.008x, W1A6 0.006x, W1A4 0.004x.
- Verified by results-analyst: PENDING (hand-off for verification of attention-MAC formula + BOPs table).
- Next: wire the HGQ accumulator term into `accumulator_bops()` once the formula is confirmed.

## 2026-06-28 — Variant sweep ROUND-4: large is ALSO LR-limited → bigger wins once LR is tuned
- Goal: seed-confirm medium@5e-5 (s1/s2) and test whether the 6.37M "large" lr15 anchor (0.7530) is
  ALSO LR-limited. 4 jobs `kai-bn4-*`, binary+A8, all complete, 0 fails.
- Result (recomputed from PVC logs `/data/outputs/qk-variants-r4/<key>/train_<key>.log`, via kai-util3):
  - **medium@5e-5 seed-confirm:** s1=0.7503(ep14), s2=0.7606(ep14); with round-3's 0.7592 → 3-seed
    {0.7592, 0.7503, 0.7606} mean **0.7567 ± 0.005**. The 0.7592 "sweep-best" was the HIGH end of the
    spread (same pattern as small's 0.7540). Honest medium central ≈ 0.7567.
  - **large IS also LR-limited:** large@7.5e-5=0.7602 (peaked ep6, still hot), **large@5e-5=0.7672**
    (best ep37, ran 47 ep, stable) — **+0.0142 over the lr15 anchor 0.7530**. So lr15 (peak 1.5e-4)
    was too hot even for the 6.37M model it was nominally tuned for.
  - **PARAM-COUNT CORRECTION:** round-4 preflight confirms large = D256/L8/H8/F1024 = **6,373,633
    params** (matches the HLS doc's 6.37M), NOT the "~3M" the round-1/3 notes claimed. So medium
    (808K) is ~**8×** smaller than large, not ~4×. The efficiency story is stronger, not weaker.
- VERDICT (size-vs-recipe, now fully closed): the peak LR was the hidden lever at EVERY size (lr15→5e-5
  gains: small +0.015, medium +0.005, large +0.014). Once LR≈5e-5, AUC is **monotonic in size**:
  small ~0.750 < medium 0.7567 < **large 0.7672**. "medium beats large" was tuned-medium vs
  UN-tuned-large; a tuned large wins raw AUC. medium stays the **efficiency** pick (within ~0.010 of
  large at ~8× fewer params / ~8× less FPGA logic). Net practical result: the binary W1A8 tagger goes
  **0.7530 → 0.7672 validation AUC for free**, just by lowering the peak LR.
- Caveat: large@5e-5 is a SINGLE run (medium's seed spread is ~0.010, so seed-confirm large next).
  All are validation AUC on the during-training split — NOT ROC-test (re-eval needed for ROC-test).
- Verified by results-analyst: ✓ numbers re-extracted from durable PVC logs (kai-util3).
- Next: seed-confirm large@5e-5; ROUND-5 (staged, 10 jobs, run ≤3 concurrent) — activation precision
  A6/A4 on tuned medium+small (the thesis quantization axis), shape-at-fixed-budget, a bigger model
  @5e-5. Also: ROC curves into W&B (per-run + cross-run overlay); backfill A6/A4 full-model csynth.

## 2026-06-28 — Variant sweep ROUND-3: small ceiling ~0.750; medium@5e-5=0.7592 is sweep-best
- Goal: find small's LR ceiling, seed-confirm small@5e-5, and re-run medium below lr15 (it also
  collapsed at ep3 → LR-limited). 6 jobs `kai-bn3-*`, binary+A8, all complete, 0 fails.
- Result (recomputed from PVC logs `/data/outputs/qk-variants-r3/<key>/train_<key>.log`):
  - **small LR ceiling:** lr025=0.7500(ep33) · lr035=0.7537(ep25) · lr05 3-seed {0.7540,0.7483,
    0.7474} mean **0.7499±0.0036**. Plateaus ~0.750 for LR∈[3.5e-5,5e-5]; 2.5e-5 just trains longer
    (peak ep33) for no gain. The round-2 single 0.7540 was the high end of the 5e-5 spread.
  - **medium re-opens the size gap:** medium@7.5e-5=0.7576, **medium@5e-5=0.7592** (up from
    0.7538 @lr15). 0.7592 is the **sweep-best**, beating ~3M large (0.7530 @lr15) by ~0.006 at ~4×
    fewer params, and clearly above small (~0.750). Once LR is right, the bigger model wins again —
    round-2's "small catches up" was "small was the most LR-starved."
- Efficiency frontier (best/size): tiny 0.7112(lr15) · small ~0.750(@3.5–5e-5) · **medium 0.7592
  (@5e-5)** · large 0.7530(@lr15, NOT low-LR-tested).
- RECOMMENDATION: best overall = medium D128/L4/H8/F512 (808K) @ LR 5e-5 → 0.7592; smallest viable =
  small D64/L3/H8/F256 (154K) @ ~5e-5 → ~0.750; keep architecture knobs at DEFAULT (ablations were
  noise). Recipe (LR 1.5e-4→5e-5) is the lever, not the knobs.
- Verified by results-analyst: ✓ from durable PVC logs.
- Next: ROUND-4 (`kai-bn4-*`, 4 jobs) — seed-confirm medium@5e-5 (s1/s2), and test large@5e-5/7.5e-5
  (is the ~3M lr15 anchor ALSO LR-limited? would make "medium beats large" need a tuned-large
  baseline). Only then is the size-vs-recipe story fully closed.

## 2026-06-28 — Variant sweep ROUND-2: recipe (LR) dominates; round-1 arch "wins" were noise
- Goal: fix the round-1 lr15 collapse (LR sweep), bound the ~0.013 knob gaps (seed repeats), and
  test stacking the round-1 winners (combos). 10 jobs `kai-bn2-*`, binary+A8, all complete, 0 fails.
- Result (recomputed from PVC logs `/data/outputs/qk-variants-r2/<key>/train_<key>.log`):
  - **LR sweep on small (D64/L3/H8/F256):** lr15=0.7335(ep6) → lr10=0.7344(ep6) →
    lr10+warm3=0.7422(ep4) → lr075=0.7480(ep5) → **lr05=0.7540(ep17, ran 27 ep)**. Monotonic;
    lower LR = higher AUC AND later/stabler peak. **small@5e-5 (0.7540) ≈ medium 0.7538 ≈ large
    0.7530** — the 153,793-param model matches the ~3M model once LR is right. The round-1
    early-peak-collapse was an LR artifact, not an architecture property.
  - **Seed repeats (3 samples @ lr15):** small {0.7335, 0.7365, 0.7359} mean 0.7353, spread 0.0030;
    clspool {0.7475, 0.7247, 0.7343} mean 0.7355, spread **0.0228**. clspool mean ≈ small mean →
    **round-1 clspool +0.014 was a lucky single draw, NOT a real effect** (clspool just has higher
    variance). Paired same-seed diffs clspool−small = {−0.0118,−0.0016,+0.0140}, mean ≈ +0.0002.
  - **Combos @ lr15:** combo-small (cls+gelu+noPE)=0.7323 ≈ small; combo-medium=0.7504 ≤ medium
    0.7538. Stacking the "winners" gives nothing — consistent with them being noise.
- VERDICT: at this scale AUC is governed by the **optimizer (peak LR)**, not the architectural
  knobs. Recommended small config = **default-arch small @ LR 5e-5** (0.7540). Single-run gaps
  <~0.02 at lr15 are noise.
- Verified by results-analyst: ✓ all numbers re-extracted from durable PVC logs; seed-design
  confirmed clean (BN_SEED sets np.random.seed BEFORE the data shuffle at line 773, and the model
  build at line 852 is AFTER the shuffle, so same-seed runs share an identical train/val split).
- Next: ROUND-3 (`kai-bn3-*`) — confirm small@5e-5 with seeds, push LR lower (3.5e-5, 2.5e-5),
  re-run medium below lr15 (medium also collapsed @ ep3 ⇒ LR-limited) to see if the size gap reopens.

## 2026-06-28 — Variant sweep ROUND-1 results + KEY FINDING: lr15 recipe collapses at small scale
- Goal: train + compare the 10 round-1 variants (SIZE + one-knob ablations) on GPU.
- Change / job files: launched all 10 `kai-bnv-*` Jobs on NRP; binary + A8 + lr15 recipe.
- Result (recomputed from PVC logs `/data/outputs/qk-variants/<key>/train_<key>.log`, best `val_auc`):
    - SIZE: tiny **0.7112**(ep6) · small **0.7335**(ep6) · medium **0.7538**(ep3) · large 0.7530 (existing).
      → size monotonic; **medium (808K) ≥ large (~3M) at ~4× fewer params**.
    - ABLATION vs small (0.7335): clspool **0.7475**(+.014) · gelu **0.7471**(+.014) ·
      sffree **0.7462**(+.013) · noposenc 0.7380(+.004) · rmsnorm 0.7358(+.002) ·
      sharednorm 0.7291(−.004) · trueposenc 0.7275(−.006).
      → CLS/GELU/softmax-free each best & ~tied; **PE hurts** (real-learned PE is worst, below
      dropping PE); norm structure ~neutral.
- **KEY FINDING (the headline):** EVERY variant **peaks at epoch 2–6 then collapses** under lr15
  (e.g. medium 0.7538@ep3 → 0.667@ep13; clspool 0.7475@ep4 → 0.65; small 0.7335@ep6 → 0.68).
  The lr15 peak LR (1.5e-4) was tuned for the ~3M large model and is **too hot at these sizes**.
  So all round-1 numbers are *best-transient lower bounds*; the ~0.01–0.02 gaps are within
  plausible single-run/unseeded noise. EarlyStopping `restore_best_weights=True` → reported best
  is the pre-collapse peak, so the ranking is still usable, but not converged.
- Verified by results-analyst: ✓ numbers re-extracted directly from durable PVC logs (not memory);
  collapse confirmed on full trajectories for medium/clspool/gelu/small.
- Next: ROUND-2 (10 jobs `kai-bn2-*`) — (a) LR sweep on small (5e-5/7.5e-5/1e-4 + warmup3) to find
  the true ceiling; (b) combo-best arch (CLS+GELU+noPE) on small & medium; (c) seed-repeats
  (BN_SEED=1/2) of small & clspool for paired error bars. Report: `results/variant_sweep.md`.

## 2026-06-27 — BitLinear architecture variant sweep (design axis: SIZE + NORM/structure)
- Goal: per user pivot ("make a bunch of different bitlinear transformers, test on GPU; forget
  hls4ml for now"), explore the architecture design space that the prior sweep never touched.
  Prior sweep varied only precision (A8/A6/A4) × weights (binary/ternary/vanilla/w8a8) ×
  attention (softmax/softmax-free) at a FIXED D256/L8/H8/FFN1024. Missing high-value axes =
  model SIZE and norm/architecture STRUCTURE.
- Change / job files: added env-driven architectural knobs to `code/training/qkerasModel.py`
  (BN_NORM_TYPE, BN_NORM_PLACEMENT, BN_POS_ENC, BN_POOL, BN_FFN_ACT) all defaulting to upstream
  behaviour (back-compat with lr15 ckpt). New layers: TrueRMSNorm, IdentityNorm, CLSToken,
  LearnedPosEnc, `_sinusoidal_pos`, `make_norm`. Generator `code/jobs/training/variants/
  gen_variant_jobs.py` emits 10 NRP Job yamls + `preflight.sh`. All binary weights, A8, lr15
  recipe (LR=1.5e-4, wu=1, decay=40, b2=0.98, wd=0.01, clip=1.0, bs=256, ES val_auc/max/p10).
    - SIZE: tiny D32/L2/H4/F128 (26,529p) · small D64/L3/H8/F256 (153,793p) ·
      medium D128/L4/H8/F512 (808,065p) · [large = existing lr15, reuse W&B].
    - ABLATION (anchor=small, one knob): rmsnorm · sharednorm · noposenc · trueposenc ·
      clspool (153,857p) · gelu · sffree.
- Result (recomputed): CPU `--sanity` preflight on NRP built + forward + train_on_batch'd all
  variants with ZERO exceptions; param counts above are from that run and confirm each knob
  propagates. GPU training launch pending preflight #2 (adds trueposenc).
- Verified by results-analyst: PENDING (W&B val_auc comparison after runs finish).
- Next: launch 10 GPU Jobs (kai-bnv-*), monitor W&B project bnjettag-bitnet, pick best small arch.

## 2026-06-27 — FINDING: upstream "learned" positional encoding trains 0 params (silent no-op)
- Goal: sanity-check that BN_POS_ENC knob propagates.
- Change / job file: preflight `--sanity` param-count comparison across variants.
- Result (recomputed): small (BN_POS_ENC=learned, default) = 153,793 params; noposenc
  (BN_POS_ENC=none) = 153,793 — IDENTICAL. clspool = 153,857 = small+64 (CLS add_weight),
  proving count_params() tracks added weights. ∴ the upstream learned PE adds 0 trainable
  params. Root cause: `Embedding(N,D)(tf.range(N))` is applied to a CONSTANT (not a KerasTensor
  wired from the input), so Keras folds it into a FIXED RANDOM offset and never registers/trains
  its weights. The lr15 model's "learned" PE was therefore a fixed random constant, not learned.
- Verified by results-analyst: ✓ (param-count arithmetic, two independent variants + a positive
  control).
- Next: added BN_POS_ENC=learned_real (LearnedPosEnc via add_weight, genuinely trainable) as
  variant `trueposenc`; default `learned` left unchanged for back-compat. The size axis is
  unaffected (tiny/small/medium/large all share the identical upstream PE behaviour → no
  confound). Re-interpret the noposenc ablation as fixed-random-offset vs none.
