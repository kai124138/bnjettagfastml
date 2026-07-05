---
description: Refresh RESEARCH.md (the living research doc) from the latest verified results only.
argument-hint: "[optional: section]"
---
Update `RESEARCH.md` $ARGUMENTS from the latest **verified** results only.

First have the **results-analyst** subagent confirm the numbers against the `.npz` / csynth
data, then edit `RESEARCH.md` to match. Never write a figure that has not been recomputed
this session. Always label validation AUC vs ROC-test AUC, and pre-migration (private
2-class) vs post-migration (HLS4ML 5-class) numbers — they are not comparable.
Frozen deliverables in `reports/` are historical records: do not silently rewrite them;
add a dated correction note instead. Note what changed in `.claude/memory/experiment-log.md`.
