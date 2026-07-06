# BNJetTag — Start Here

**This file is the front door.** It explains what this project is, where everything lives,
and how work gets done — written so that someone who has never seen the project (including
a non-expert) can find their way. The science itself lives in one document: **[RESEARCH.md](RESEARCH.md)**.

---

## What this project is, in plain language

The Large Hadron Collider smashes protons together **40 million times per second**. No
computer on Earth can store all of that, so detectors like CMS use a hair-trigger filter —
the **Level-1 trigger** — that must decide in ~microseconds, on special chips called
**FPGAs**, which collisions look interesting enough to keep.

One of the trigger's jobs is recognizing **jets** (sprays of particles) and what produced
them. Modern AI models called **transformers** are great at this, but they are normally far
too big and slow for trigger hardware. Our idea: build a transformer whose learned weights
are only **−1 or +1** (a "1-bit" or **BitNet**-style network). Multiplying by ±1 isn't
really multiplication — it's just add-or-subtract — so the model no longer needs the FPGA's
scarce multiplier circuits (**DSPs**) and can be built from its abundant logic fabric
(**LUTs**) instead.

The research measures the trade: **how much tagging accuracy do we give up** by going 1-bit,
and **how much hardware do we win**? We also push further — shrinking the activations
(the numbers flowing *between* layers) from 8 bits down to 6 and 4 — to find where the
model breaks. Headline so far: the binary core uses **zero DSPs** (proven in real synthesis),
at a modest accuracy cost. Current numbers, caveats, and status: **[RESEARCH.md](RESEARCH.md)**.

---

## The three machines

Nothing heavy runs in this folder. This folder is the project's **brain and notebook**;
the muscle lives elsewhere:

```
┌─────────────────────────┐      job YAMLs (kubectl apply)      ┌──────────────────────────┐
│  THIS FOLDER (laptop)   │ ──────────────────────────────────▶ │  NRP NAUTILUS (cluster)  │
│                         │                                     │                          │
│  edit code & job YAMLs  │ ◀────────────────────────────────── │  GPU training pods       │
│  parse logs, recompute  │    trained models, .npz ROC data,   │  ns cms-ml, kai-data PVC │
│  AUCs, plots, reports,  │    training logs (via PVC)          │  W&B: bnjettag-bitnet    │
│  memory logs, planning  │                                     └──────────────────────────┘
│                         │
│                         │      model checkpoint + hls4ml      ┌──────────────────────────┐
│                         │ ──────────────────────────────────▶ │  MULDER (group server)   │
│                         │                                     │                          │
│                         │ ◀────────────────────────────────── │  Vitis HLS 2023.2        │
└─────────────────────────┘    csynth reports (LUT/FF/DSP/      │  C-synthesis → real      │
                               BRAM, latency JSON)              │  resource & latency #s   │
                                                                └──────────────────────────┘
```

- **This folder** — code editing, job preparation, analysis, verification, writing. Never
  full training, never synthesis.
- **NRP Nautilus** — Kubernetes GPU cluster where all training runs. Runbooks: [`nrp/`](nrp/).
  *(Round-5 used the `kai-data` PVC; from round-6 the jobs are PVC-free — data pulled from
  Zenodo in-pod, checkpoints and metrics to W&B.)*
- **mulder** — the group's server with Xilinx Vitis HLS installed. This is the only place
  the model-to-hardware synthesis (csynth) runs, producing the *measured* resource/latency
  numbers. Nautilus has no Xilinx backend — that's why mulder exists in the loop.

---

## Finding your way: what's CURRENT vs. what's FROZEN (as of 2026-07-05)

The folder names accumulated history (the working dir is named after the June-22 run but
now hosts everything). This table is the shortcut — **"which code do we actually run?"**:

| I want… | Go to | Status |
| --- | --- | --- |
| **The training code** (runs on NRP) | `qkeras-bitnet-run-2026-06-22/code/training/qkerasModel.py` — one trainer, all variants via `BN_*` env knobs; shipped to the cluster as ConfigMap `kai-qkerasmodel-r5` | **CURRENT** |
| **The current job YAMLs** | `…/code/jobs/training/variants/kai-bn6s-*.yaml` + `launch_r6s_staged.sh` (round-6-small, fire-ready). `kai-bn5-*` = round 5 (done). YAMLs in `…/code/jobs/training/` root = era-1, frozen | **CURRENT** |
| **The conversion / verification / synthesis pipeline** (HGQ2 path, runs locally + mulder) | `…/code/hgq2/` — `run_stage.py` CLI over `configs/*.json`; `probes.py` builds synthesis probes; `fetch_mulder_reports.sh` brings numbers home; `LEDGER.md` is its change trail | **CURRENT** |
| **The current results** | `…/results/hgq2/` — the structured store (`runs/<config-hash>/`), `tradeoff_table.md`, `constraints_map.md`, `dashboard.html` | **CURRENT** |
| **The poster** | `poster/` (top level) — FastML26 figures + draft, built against a verified store snapshot with its own gate (`VERIFICATION.md`, `GAPS.md`) | **CURRENT** |
| Era-1 QKeras-path synthesis scripts | `…/code/hls/` — produced the era-1-shape DSP-0 numbers; superseded for new work by `code/hgq2/` | frozen reference |
| Era-1 run outputs | `…/results/*.md`, `…/results/csynth/`, `…/results/plots/` (June-22 run) | frozen |
| ROC arrays | `…/roc-results/*.npz` = era-1 · `…/roc-results/r5/` = era-2 round 5 | data (both eras) |

**Naming decoder** (the vocabulary that trips people up):

- **`qkeras-bitnet-run-2026-06-22/`** — named for the June-22 era-1 run, but it grew into
  the working tree for *all* code and results. Historical name, current contents. (It was
  also the git-repo root until 2026-07-05; the repo root is now this top-level folder,
  pushed to GitHub `kai124138/bnjettagfastml`.)
- **era 1 / era 2** — before/after the 2026-07-01 dataset migration (private 2-class →
  public HLS4ML LHC Jet 5-class). Numbers are never compared across eras.
- **rounds** — r1–r4 = era-1 recipe tuning · **r5** = first era-2 round, *large* arch
  (D256/H8/L8/FFN1024) · **r6s** = round-6-*small*, the deployable arch (D32/H4/L2/FFN64),
  fire-ready but not yet launched.
- **W1A8 / W1A6 / W1A4 / W8A8** — weight bits and activation bits: W1 = binary {−1,+1}.
- **config hashes** (`b224a8ea`, `a428e6e2`, `53b202bc`) — sha256[:8] of the pipeline
  configs `code/hgq2/configs/era2-large-w1a{8,6,4}.json`; the results store is keyed by them.
- **synthesis probes** (in `results/hgq2/runs/<hash>/`, each self-describing via its
  `csynth_modules.json`): `probe_subln_rf1` = the norm alone, fully parallel ·
  `probe_bitlinear_rf256` = Resource-strategy dense (the DSP-regression exhibit) ·
  `probe_bitlinear_v2_rf256` = same at Resource with the ±1/affine split ·
  `probe_bitlinear_v3lat_rf256` = big-shape Latency attempt (crashed — the negative result) ·
  `probe_bitlinear_head_fc2_rf32` = the DSP-0 verification (Latency) ·
  `probe_attn_core_rf1`/`_rf64` = the attention core, spatial / folded.

---

## Folder map

| Path | What it is |
| --- | --- |
| **`00-START-HERE.md`** | This file — the front door. |
| **`RESEARCH.md`** | **The one living research document**: thesis, data, every verified result, current status, open questions. If you read one file, read that one. |
| **`ROADMAP.md`** | The steering doc: phased task list with checklists and per-item success indicators. `RESEARCH.md` = what we know; `ROADMAP.md` = what we do next and how we'll know it worked. |
| `CLAUDE.md` | Working agreement for the AI assistant team (conventions, guardrails). |
| **`qkeras-bitnet-run-2026-06-22/`** | The working tree (historical name — see the decoder above). `code/training/` the trainer + ROC scripts · **`code/hgq2/` the current conversion/verification/synthesis pipeline** · `code/hls/` era-1 synthesis scripts (frozen) · `code/jobs/` job YAMLs (current = `training/variants/kai-bn6s-*`) · **`results/hgq2/` the current results store** · `results/` era-1 HLS tables (frozen) · `roc-results/` the `.npz` ROC arrays · `methods/` methodology notes · `README.md` the run-level map. |
| **`poster/`** | FastML26 poster deliverables: verified figures, methods/results draft, its own verification gate (`VERIFICATION.md`) and gap list (`GAPS.md`). |
| `nrp/` | How-to runbooks for the cluster: Nautilus setup, GPU training runbook, mulder↔NRP data transfer. |
| `reports/` | Frozen deliverables: the June-22 run report, presentation outline + slides + speaker notes (.docx), the strategy review. Historical records — corrected by dated notes, never silently rewritten. |
| `papers/` | Literature notes (and key PDFs), organized: BitNet/1-bit · hls4ml/FPGA triggers · jet-tagging transformers · quantization foundations. |
| `reference-code/` | Read-only clones of upstream code (Microsoft BitNet, hls4ml, QKeras, Particle Transformer, weaver). |
| `models/` | Trained model binaries (large; see `models/MODEL.md`). |
| `LEARN/` | Kai's private study notes (Obsidian vault) — a guided tour of the physics, the math, and the code, `00` → `11`. |
| `archive/` | Old session exports, superseded duplicates, and the June-22 stray logs. Nothing here is load-bearing. |
| `.claude/` | The assistant team: `agents/` (the workers) · `skills/` (the three workflow playbooks) · `commands/` (shortcuts) · `memory/` (the logging system, below). |

---

## How an experiment happens (the loop)

Every result in this project is produced the same way:

1. **Idea** — e.g. "what happens at 4-bit activations?" Prior art gets checked first
   (`/literature-scan`, logged to the research log).
2. **Job YAML** — the idea becomes a config in `qkeras-bitnet-run-2026-06-22/code/jobs/training/`,
   set by environment knobs (`BN_VARIANT`, `BN_ACT_BITS`, `BN_N_PART`, `BN_TERNARY`,
   `BN_SOFTMAX_FREE`) — code changes are rare; knob changes are the norm.
3. **Train on Nautilus** — preflight script, then `kubectl apply`. Progress watched via
   W&B (project `bnjettag-bitnet`) and pod logs. *(Playbook: `.claude/skills/nrp-training-run/`)*
4. **Evaluate** — a ROC job (`kai-roc-r5.yaml`) runs the trained model on the held-out set
   and writes `.npz` arrays back to the PVC, which come home to `roc-results/`.
5. **Verify** — nothing is believed until the AUC is **recomputed from the `.npz`** and the
   outcome is logged. *(Playbook: `.claude/skills/verify-roc/`)*
6. **Synthesize (hardware axis)** — the checkpoint goes through hls4ml on mulder; csynth
   reports land in `results/csynth/`. *(Playbook: `.claude/skills/hls-mulder/`)*
7. **Record** — verified numbers flow into `RESEARCH.md`; the experiment log gets an entry;
   decisions get a dated rationale.

**House rules** (they exist because each one was earned):

- **Two AUCs, never conflated:** *validation AUC* (training monitor) vs *ROC-test AUC*
  (held-out set). Every quoted number says which it is.
- **Two dataset eras, never compared:** numbers before the 2026-07-01 migration (private
  dataset, 2-class) vs after (public HLS4ML LHC Jet, 5-class, macro-OvR AUC). Round-5 starts
  the new era's table from scratch.
- **No invented numbers:** every figure is recomputed from `.npz`/logs or quoted with its
  source file.
- **No heavy compute here:** training = Nautilus, synthesis = mulder.
- **Secrets:** `wandb-api-key.txt` is a real credential — never printed, pasted, or committed.

---

## The logging system (`.claude/memory/`)

Four files carry the project's memory. All logs are **append-only, dated, newest on top**.

| File | What goes in it | Written by |
| --- | --- | --- |
| `project-context.md` | Durable facts only — the thesis, infrastructure, headline results, layout. Loaded at the start of **every** assistant session. | edited when a durable fact changes |
| `experiment-log.md` | One entry per experiment or verification pass: goal, change, recomputed result, ✓/✗ verified, next step. | ml-engineer, results-analyst, lead-pm |
| `research-log.md` | Literature findings with **source URLs**. | physics-researcher |
| `decisions.md` | Every consequential choice + its rationale (e.g. the dataset migration, fixing the architecture at `large`). | whoever decides, usually via lead-pm |

Why it matters: any new session — or any new person — can reconstruct *where things stand
and why* by reading `RESEARCH.md` + the tops of these logs. Nothing important lives only in
someone's head or a chat scrollback.

---

## The assistant team (Claude Code)

Run `claude` in this folder; it auto-loads `CLAUDE.md` + `project-context.md`. Full guide:
[`.claude/HOW-TO-USE.md`](.claude/HOW-TO-USE.md).

| Piece | Members | Job |
| --- | --- | --- |
| **Agents** | `lead-pm` · `physics-researcher` · `ml-engineer` · `results-analyst` · `paper-writer` | plan → research → implement → **verify** → write. The results-analyst is the gate: no number reaches a report unverified. |
| **Skills** | `nrp-training-run` · `verify-roc` · `hls-mulder` | Step-by-step playbooks for the three workflows, so any session runs them identically. |
| **Commands** | `/status` · `/new-experiment` · `/verify-results` · `/update-report` · `/literature-scan` · `/log-decision` · `/pi-update` | Shortcuts for the recurring moves. |

---

## If you're new, read in this order

1. This file (done!)
2. **`RESEARCH.md`** — what we're claiming and what we've measured.
3. **`ROADMAP.md`** — what happens next: phases, checklists, success indicators.
4. `LEARN/00 - Start Here.md` → the guided tour, if you want the concepts from zero.
5. `qkeras-bitnet-run-2026-06-22/README.md` — the run-level map of code, jobs, and results.
6. `reports/strategy-review-2026-07-02.md` — how the project is being run, risks, next moves.
