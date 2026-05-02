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
Session 1  →  work happens  →  remember.ts saves the learning
Session 2  →  search.ts finds it  →  Claude already knows
```

Three scripts. No servers. No API keys. Works on any machine.

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

Pure TypeScript. No Python. No external services. No API keys for memory.

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
