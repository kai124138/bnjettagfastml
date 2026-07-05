---
description: Run a full research → implement → verify cycle for a new BNJetTag experiment idea.
argument-hint: <experiment idea, e.g. "try W1A3 activations">
---
Plan and run a complete experiment cycle for: **$ARGUMENTS**

1. Use the **physics-researcher** subagent to check prior art — has anyone reported results in
   this regime? Log findings (with URLs) to `.claude/memory/research-log.md`.
2. Use the **ml-engineer** subagent to propose the minimal code / job-YAML change that realizes
   it, mirroring the existing `kai-bn-train-*.yaml` knob pattern. **Do not run training** —
   produce a ready-to-submit NRP Nautilus job file.
3. Summarize exactly what to submit on Nautilus and which artifact to bring back.
4. Once results exist, use the **results-analyst** subagent to recompute AUC from the returned
   `.npz` and compare against the FP32 / W8A8 baselines.

Keep me in the loop at each handoff. Append the plan and outcome to
`.claude/memory/experiment-log.md`.
