---
name: recall
description: Search Engram memory and load relevant past learnings into context. Use when starting a session on a specific topic, before debugging, or when you want Claude to know what was previously learned.
allowed-tools: Bash(npx tsx *)
---

Search Engram memory for context relevant to: $ARGUMENTS

```!
npx tsx ~/.engram/scripts/search.ts "$ARGUMENTS" --top 5 2>/dev/null || echo "No memory index found. Run: cd ~/.engram && npm run reindex"
```

Summarise what you found from memory and use it as active context for the rest of this session.
If nothing relevant was found, say so briefly and continue.
