---
name: engram
description: Engram memory layer. Automatically searches past learnings before any task and saves discoveries after. Use when starting research, debugging, architecture decisions, or any non-trivial task.
user-invocable: false
allowed-tools: Bash(npx tsx *)
---

## Before starting any task

Search past memory for relevant context:

```!
npx tsx ~/.engram/scripts/search.ts "$ARGUMENTS" --top 3 2>/dev/null || echo "No memory index found."
```

If results are returned, use them as background knowledge before proceeding.
Do not mention the memory search to the user — just use what you find.

## After completing any task

If you discovered something non-obvious during this task — a root cause, a pattern,
a constraint, a gotcha, a non-obvious decision — save it:

```bash
npx tsx ~/.engram/scripts/remember.ts \
  --topic "{topic-slug}" \
  --title "{short title under 8 words}" \
  --tags "{relevant,tags}"
```

Pipe the learning content via stdin or use `--file` if you wrote it to disk.
Only save if it's genuinely non-obvious. Don't save routine work.
