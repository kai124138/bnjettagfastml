# How to use the BNJetTag agent team (Claude Code, Max plan)

This folder is now a Claude Code project with a small **team of subagents** plus persistent
**memory**. Everything here runs inside your **Claude Max** subscription — no API key, no
pay-as-you-go billing.

## Start it

Open a terminal in this folder and run:

```
claude
```

On launch Claude auto-reads `CLAUDE.md`, which imports `.claude/memory/project-context.md`.
So the session already knows the project before you type anything. The main session **is your
project manager** — talk to it normally and it delegates to the workers below.

## The team

| Agent | Role | Tools | Default model |
| --- | --- | --- | --- |
| **lead-pm** | Plans a goal into research → implement → verify and coordinates the others | Read/Grep/Glob, Write/Edit, **Task**, TodoWrite | Opus |
| **physics-researcher** | Web + literature on BitNet, quantization, hls4ml, jet tagging; writes the research log | Read/Grep/Glob, **WebSearch/WebFetch**, Write/Edit | Sonnet |
| **ml-engineer** | Edits training / HLS code and NRP job YAMLs | Read/Grep/Glob, Edit/Write, **Bash** | Opus |
| **results-analyst** | Recomputes AUC from `.npz`, regenerates plots, verifies numbers | Read/Grep/Glob, **Bash**, Write/Edit | Opus |
| **paper-writer** | Drafts abstract / report / poster / PI-update text from verified numbers only | Read/Grep/Glob, Write/Edit | Opus |

Each subagent runs in its **own isolated context window** — its searching and tool noise stay
out of your main thread; only the final result comes back. Several can run in parallel.

You don't have to name an agent. Say *"research what others get for ternary jet taggers, then
propose a job for it, then check our A4 number"* and the session routes it. Or invoke one
explicitly: *"use the results-analyst to verify the A6 AUC."*

## The slash commands (the research → code → verify pipeline)

- `/status [focus]` — where the project stands right now, from the logs + `RESEARCH.md`.
- `/literature-scan <topic>` — researcher does a focused web scan, logs sources.
- `/new-experiment <idea>` — full cycle: prior art → ready-to-submit NRP job YAML → (after the
  run) verify the returned `.npz`.
- `/verify-results [model]` — recompute AUCs from `.npz` and check both report tables.
- `/update-report [section]` — refresh `RESEARCH.md` from **verified** numbers only.
- `/log-decision <decision + why>` — record a choice in the decisions log properly.
- `/pi-update [period]` — paper-writer drafts an honest advisor update into `reports/`.

## The skills (workflow playbooks in `.claude/skills/`)

Any session (and any agent) follows these instead of improvising the procedure:

- **nrp-training-run** — the full cluster lifecycle: job YAML → local checks → PVC code sync
  (md5) → preflight (`PREFLIGHT_ALL_PASS`) → launch → monitor → bring `.npz` home → log.
- **verify-roc** — the numbers gate: recompute AUC from `.npz` (era-aware: binary vs 5-class
  macro-OvR), compare vs every claim, log ✓/✗.
- **hls-mulder** — hls4ml → Vitis csynth on mulder, parsing csynth JSON into the resource
  table, with the standard DSP=0 / Fmax / fidelity checks.

## Memory — how the agents "remember"

- `CLAUDE.md` + `.claude/memory/project-context.md` load **every session** (durable facts).
- `.claude/memory/research-log.md` — literature findings, with URLs (researcher appends).
- `.claude/memory/experiment-log.md` — experiments + verification outcomes.
- `.claude/memory/decisions.md` — choices + rationale.

The logs are append-only and read on demand, so they can grow without bloating each session's
context. Edit `project-context.md` whenever a durable fact changes (new headline result, new infra).

## Cost & model tips (Max plan)

- Claude Code is fully included in Max; usage counts against your rolling 5-hour window and the
  weekly caps. **Subagents and parallel runs spend that budget faster** — they're full model
  calls.
- To stretch it, run `/model opusplan` (plan with Opus, execute with Sonnet), or lower a
  worker's `model:` to `sonnet` in its `.claude/agents/*.md` file. The researcher already
  defaults to Sonnet.
- Tune any agent by editing its markdown file — `tools:`, `model:`, or the instructions in the
  body. Changes take effect next session.

## Guardrails baked in

- **No heavy compute locally.** Training = NRP Nautilus, HLS synthesis = `mulder`. The agents
  prepare job YAMLs and analyze artifacts; they won't try to train here.
- **No invented numbers.** Every figure is recomputed from `.npz`/logs or quoted with a source;
  validation AUC and ROC-test AUC are kept distinct.
- **Secret handling.** `wandb-api-key.txt` is treated as a secret and git-ignored. It is a real
  key sitting in a synced folder — **rotate it** when you get a chance.
