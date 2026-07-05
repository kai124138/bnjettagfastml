---
description: Draft a progress update for the PI/advisor from verified state only.
argument-hint: "[optional: period or focus, e.g. 'this week' or 'round-5']"
---
Use the **paper-writer** subagent to draft a progress update for my advisor $ARGUMENTS.

Source material: `RESEARCH.md` (§2 status, §5–6 results, §7 open questions) and the newest
`.claude/memory/experiment-log.md` / `decisions.md` entries. Verified numbers only, each
labeled (val vs ROC-test, era, single-run vs seed-averaged).

Shape: short and honest — (1) what moved since the last update, (2) current numbers that
matter (with caveats stated, not hidden), (3) what's running / blocked right now,
(4) decisions taken and why, (5) what I plan next and any question I have for them.
Plain prose, no hype, no bullet-point wall — it should read like me writing to my advisor.
Save as `reports/pi-update-YYYY-MM-DD.md` (today's date), and tell me the 2–3 things most
worth saying out loud in the meeting.
