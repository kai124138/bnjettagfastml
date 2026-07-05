---
name: nrp-training-run
description: Run a BNJetTag training experiment on NRP Nautilus end-to-end — create or regenerate the job YAML, preflight, sync code to the PVC, launch, monitor, and bring results home. Use whenever the task involves submitting/monitoring a training or ROC job, adding a new experiment variant, editing kai-bn*-*.yaml job files, or anything mentioning Nautilus, kubectl, the kai-data PVC, or "launch round-N".
---

# NRP training run — the canonical lifecycle

Training NEVER runs locally. This skill turns an experiment idea into a completed run on
NRP Nautilus (namespace **`cms-ml`**) with artifacts back in this folder. Deep background:
`nrp/bnjettag-nrp-gpu-training-runbook.md` and `nrp/nrp-nautilus-setup.md`.

## Mental model

- Pods are **disposable**; the `kai-data` PVC (mounted at `/data`) is **permanent**.
  Code lives at `/data/BNJetTag`, dataset at `/data/hls4ml_lhc_jet/` (era 2), outputs under
  `/data/outputs/`. Anything installed in a pod dies with the pod.
- One experiment = one k8s **Job** YAML in
  `qkeras-bitnet-run-2026-06-22/code/jobs/training/` (round sweeps in `variants/`).
  Experiments differ by **env knobs**, not code forks: `BN_VARIANT` (bitnet/vanilla/w8a8),
  `BN_ACT_BITS` (8/6/4), `BN_N_PART` (default 10), `BN_TERNARY`, `BN_SOFTMAX_FREE`, seed.

## Lifecycle

1. **Create the config.** For a round sweep, edit the generator
   (`variants/gen_round5_jobs.py` is the current pattern) and **regenerate** — it emits the
   `kai-bn5-*.yaml` set plus `preflight_r5.sh` and `launch_r5_staged.sh`. Never hand-edit a
   generated YAML (it will be overwritten); for a one-off, copy the newest `kai-bn*` job and
   change only the knobs + `metadata.name` + labels.
2. **Local checks (cheap, always):** YAML parses; any embedded Python compiles
   (`python -m py_compile`); if `make_roc.py` changed, re-embed it into `kai-roc-r5.yaml`
   with `code/jobs/training/reembed_make_roc.py` and confirm the byte-identical check.
3. **Sync code to the PVC.** The PVC copy of `qkerasModel.py` / `make_roc.py` is what
   actually trains — a stale PVC copy silently runs old code (this bit us on 2026-07-01).
   Transfer with gzip + **md5 verification** on both ends.
4. **Preflight before launching anything:** run the round's `preflight_*.sh` in a cheap CPU
   pod; require the literal `PREFLIGHT_ALL_PASS`. It validates dataset files, builds every
   config, and prints the exact param count (era 2: 6,375,173 — do not quote the old
   6,373,633).
5. **Launch:** `kubectl apply -f <job>.yaml -n cms-ml`. Staged launcher (≤3 concurrent) vs
   all-at-once: staged is polite but needs the laptop alive; all-at-once survives laptop
   close and the cluster scheduler self-limits. Record which was used.
6. **Monitor:** `kubectl get jobs,pods -n cms-ml`, `kubectl logs -f <pod> -n cms-ml`, and
   W&B project **`bnjettag-bitnet`**. Known quota reality: `h100/h200/gh200 = 0` — jobs land
   on A100 / L40S / 4090 / A40 regardless of YAML affinity preferences.
7. **Evaluate:** after training, apply `kai-roc-r5.yaml` (ROC on the held-out `val/` split);
   it writes `.npz` per model to the PVC.
8. **Bring artifacts home** to `qkeras-bitnet-run-2026-06-22/roc-results/` (and logs if
   needed), then **hand off to the `verify-roc` skill** — no number is real until recomputed.
9. **Log it:** append the run (goal, job files, where artifacts landed, deviations) to
   `.claude/memory/experiment-log.md`, newest on top.

## Gotchas that have actually happened

- **Stale W&B secret kills every job**: `wandb.init()` is not wrapped in try/except. After a
  key rotation, update the `kai-wandb` secret in `cms-ml` and verify (sha256) before launch.
  The key file `wandb-api-key.txt` is a secret — never print or commit it.
- Idle-GPU pods get reaped; batch **Jobs** with `backoffLimit: 0`,
  `activeDeadlineSeconds: 172800` are the pattern, not interactive pods.
- Flaky transfers: always md5-check code pushed to the PVC.
- Era discipline: round-5+ runs train on the HLS4ML LHC Jet data (5-class, macro-OvR) —
  never compare their AUCs to pre-2026-07-01 numbers.
