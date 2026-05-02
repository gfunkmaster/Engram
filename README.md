# Engram

**Claude forgets everything. Engram doesn't.**

Engram is a portable semantic memory layer for Claude Code. Every session leaves a trace. Every trace makes the next session smarter.

---

## The Problem

Every Claude Code session starts from zero. You re-explain context. You rediscover patterns. You make decisions you already made three weeks ago because there's no record of them.

Claude Code has CLAUDE.md for rules and skills for workflows — but neither is a *learning* system. They hold what you put in manually. They don't accumulate knowledge from work you've already done.

Engram fills that gap.

---

## How It Works

```
You type a prompt
  → on-prompt.ts searches memory → relevant context injected silently
  → Claude responds
  → on-stop.ts checks response for learnings → saves automatically
```

Fully automatic in both directions. You just work.

- **Markdown** is the source of truth — human-readable, git-tracked, portable forever
- **sqlite-vec** is the search layer — a single `.db` file, no server process
- **all-MiniLM-L6-v2** generates embeddings locally via ONNX — works fully offline after first download

---

## Setup

```bash
git clone https://github.com/gfunkmaster/Engram ~/.engram
cd ~/.engram && npm install
```

On a new machine, after cloning:

```bash
npm run reindex   # rebuilds the vector index from your markdown files
```

---

## Usage

### Save a learning

```bash
# From a file
npm run remember -- --topic "auth" --title "JWT refresh token rotation" --tags "auth,jwt" --file output.md

# From stdin
echo "Always validate the refresh token family to prevent reuse attacks" | \
  npm run remember -- --topic "auth" --title "Token family validation"
```

### Search past memory

```bash
npm run search -- "how did we handle JWT auth"
npm run search -- "websocket reconnection strategy" --top 3
```

### Rebuild the full index

```bash
npm run reindex
```

---

## Scripts

| Script | Command | Purpose |
|---|---|---|
| `reindex.ts` | `npm run reindex` | Rebuild vector index from all markdown files |
| `search.ts` | `npm run search -- "<query>"` | Semantic search over memory |
| `remember.ts` | `npm run remember -- --topic ... --title ...` | Write and immediately index a new memory |

---

## File Structure

```
~/.engram/
├── package.json
├── tsconfig.json
├── scripts/
│   ├── reindex.ts
│   ├── search.ts
│   └── remember.ts
└── memory/
    ├── raw/          ← markdown files (git-tracked, source of truth)
    └── memory.db     ← vector index (gitignored, regeneratable)
```

Memory is organized by topic:

```
memory/raw/
├── auth/
├── patterns/
├── decisions/
└── learnings/
```

---

## Automatic Mode — Claude Code Hooks

Two hooks make Engram fully automatic. Add both to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "npx tsx /path/to/Engram/hooks/on-prompt.ts"
          }
        ]
      }
    ],
    "Stop": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "npx tsx /path/to/Engram/hooks/on-stop.ts"
          }
        ]
      }
    ]
  }
}
```

### `hooks/on-prompt.ts` — auto-search (on every prompt)

1. Embeds your prompt locally
2. Searches memory for relevant past learnings
3. Injects matches as silent context before Claude sees your message
4. Exits in milliseconds if nothing relevant found

### `hooks/on-stop.ts` — auto-remember (after every response)

1. Reads Claude's last response from the session transcript
2. Runs a fast signal-phrase check (`"turns out"`, `"root cause"`, `"the gotcha"`, etc.)
3. If no signals → exits immediately, zero cost
4. If signals found → sends response to Claude Haiku: *"is this worth saving?"*
5. If yes → checks for duplicates → embeds → saves to `memory/raw/` → indexes

**Both hooks fail silently.** Nothing ever blocks Claude or surfaces errors to you.

### Why Haiku for the judgment call

Haiku is fast (~1-2s) and costs ~$0.0001 per call. Most responses get filtered by the signal-phrase check and never reach Haiku. For the ones that do, Haiku has the judgment to distinguish a genuine learning from routine output — something no regex can do reliably.

---

## Integration with Claude Code Skills

Skills call Engram at two points in every pipeline:

**At the start (research/analyze skills):**
```bash
npx tsx ~/.engram/scripts/search.ts "$ARGUMENTS" --top 5
```

**At the end (review/deliver skills):**
```bash
npx tsx ~/.engram/scripts/remember.ts \
  --topic "{domain}" \
  --title "{what was learned}" \
  --file ~/.claude/context/output/final.md
```

Over time, Claude arrives at each session pre-loaded with everything it has learned from every previous session in that domain.

---

## Dependencies

```json
{
  "@huggingface/transformers": "^3.0.0",
  "better-sqlite3": "^9.4.3",
  "sqlite-vec": "^0.1.6",
  "tsx": "^4.7.0"
}
```

Pure TypeScript. No Python. The `@anthropic-ai/sdk` is only used by `on-stop.ts` for the Haiku judgment call — your existing `ANTHROPIC_API_KEY` from Claude Code covers it.

---

## New Machine Setup

```bash
git clone https://github.com/gfunkmaster/Engram ~/.engram
cd ~/.engram && npm install
npm run reindex
```

The markdown files travel with you via git. The vector index is regenerated on each machine in seconds.

---

## Why "Engram"

In neuroscience, an engram is the physical trace a memory leaves in the brain — the stored residue of a learned experience. That's exactly what this system writes after every session.
