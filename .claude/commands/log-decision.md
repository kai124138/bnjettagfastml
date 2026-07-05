---
description: Record a decision + rationale in the decisions log (and propagate if durable).
argument-hint: <the decision, and why>
---
Record this decision in `.claude/memory/decisions.md`: **$ARGUMENTS**

1. Append a dated entry, newest on top, matching the file's existing style: what was
   decided, the rationale, the alternatives considered (if any), and the concrete
   consequences (files/configs it touches, what it supersedes).
2. If it changes a durable fact (infrastructure, headline result, layout, metric
   definition), also update `.claude/memory/project-context.md` and, if user-visible,
   `RESEARCH.md`.
3. If it supersedes an earlier decision, add a one-line "superseded by YYYY-MM-DD" note
   under the old entry rather than deleting it.
4. Echo back the entry you wrote so I can correct it immediately.
