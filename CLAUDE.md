# BNJetTag — Working Agreement (CLAUDE.md)

You are working in the **BNJetTag** research project: a 1-bit BitNet `{−1,+1}`-weight
transformer jet tagger for the CMS Level-1 trigger, pushed through hls4ml to FPGA.
Durable project facts are imported at the bottom; read the memory logs on demand.

**Orientation:** `00-START-HERE.md` is the map of everything; `RESEARCH.md` is the single
living document of what we claim, what we've measured, and what's open. Keep both true.

## How we work (conventions)

- **Accuracy over speed.** Never invent numbers. Any AUC / resource / latency figure
  must be recomputed from the `.npz` or logs, or quoted with its source file. Always
  distinguish *validation AUC* from *ROC-test AUC*, and **era 1** (pre-2026-07-01 private
  2-class data) from **era 2** (public HLS4ML 5-class data) — never compare across eras.
- **No heavy compute here.** Do not run full training or Vitis synthesis locally. Training
  runs on **NRP Nautilus**, HLS synthesis on **`mulder`**. Locally we edit code, prepare
  job YAMLs, parse logs, recompute ROC/AUC from `.npz`, make plots, and update docs.
- **Follow the playbooks.** The three recurring workflows have skills — use them instead of
  improvising: `.claude/skills/nrp-training-run/` (submit/monitor cluster jobs),
  `.claude/skills/verify-roc/` (the numbers gate), `.claude/skills/hls-mulder/` (synthesis).
- **Cite sources.** Web findings go in `.claude/memory/research-log.md` with date + URL.
- **Log as you go.** Experiments and verifications → `.claude/memory/experiment-log.md`;
  choices + rationale → `.claude/memory/decisions.md`. Append, date-stamped, newest on top.
- **Reports:** `RESEARCH.md` is living and updated from verified numbers only; everything in
  `reports/` is frozen history — correct with dated notes, never silent rewrites.
- **Secrets & noise.** Treat `qkeras-bitnet-run-2026-06-22/wandb-api-key.txt` as a secret —
  never print, paste, or commit it (rotated 2026-07-01; NRP secret `kai-wandb` must match).
  Ignore for context: `.venv-plots/`, `archive/`, `*.zip`, `__pycache__/`, `.DS_Store`.

## The agent team (delegate with the Task tool)

- **lead-pm** — turns a goal into a research → implement → verify plan and coordinates the
  others. (Or just talk to me, the main session — I act as PM by default.)
- **physics-researcher** — literature/web on BitNet, quantization, hls4ml, jet tagging;
  owns the research log. Read-only on code.
- **ml-engineer** — edits training / HLS code and NRP job YAMLs; debugs scripts.
- **results-analyst** — recomputes AUC from `.npz`, regenerates plots, and verifies any
  number before it reaches a report. **This is the verification gate.**
- **paper-writer** — drafts abstract / report / poster / PI-update text from verified
  numbers only; every figure labeled and sourced.

Default loop: **research → implement → verify.** End substantive work with a
results-analyst check before updating `RESEARCH.md` or anything in `reports/`.

@.claude/memory/project-context.md
