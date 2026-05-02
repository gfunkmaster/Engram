#!/usr/bin/env tsx
/**
 * Claude Code Stop hook — auto-remember.
 *
 * After every Claude response:
 *   1. Fast signal-phrase pre-filter (instant, zero cost)
 *   2. If signals found → Claude Haiku decides if worth saving (1-2s)
 *   3. If worth saving → embed + duplicate-check + write to memory
 *
 * Completely silent. Never outputs to stdout. Never blocks Claude.
 * Fails gracefully on any error.
 */

import Anthropic from '@anthropic-ai/sdk';
import Database from 'better-sqlite3';
import * as sqliteVec from 'sqlite-vec';
import { pipeline } from '@huggingface/transformers';
import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { execSync } from 'child_process';

const ENGRAM_DIR = join(dirname(fileURLToPath(import.meta.url)), '..');
const RAW_DIR = join(ENGRAM_DIR, 'memory', 'raw');
const DB_PATH = join(ENGRAM_DIR, 'memory', 'memory.db');
const EMBED_MODEL = 'Xenova/all-MiniLM-L6-v2';
const HAIKU = 'claude-haiku-4-5-20251001';
const DUPLICATE_THRESHOLD = 0.2; // cosine distance — lower = more similar

const SIGNAL_PHRASES = [
  'turns out', 'it turns out',
  'discovered that', 'found that', 'realized that',
  'the issue was', 'the problem was', 'root cause',
  'important to note', 'worth noting',
  'the key insight', 'the trick is', 'the trick here',
  'learned that', 'the gotcha',
  'fixed by', 'resolved by', 'solved by',
  'always ensure', 'never do', 'avoid doing',
  'non-obvious', 'counterintuitive',
  'caveat', 'edge case', 'the catch is',
  'the reason this', 'what makes this',
];

function hasSignal(text: string): boolean {
  const lower = text.toLowerCase();
  return SIGNAL_PHRASES.some(p => lower.includes(p));
}

function getLastAssistantMessage(transcriptPath: string): string {
  try {
    const lines = readFileSync(transcriptPath, 'utf-8').trim().split('\n');
    for (let i = lines.length - 1; i >= 0; i--) {
      const entry = JSON.parse(lines[i]);
      if (entry.role !== 'assistant') continue;
      if (typeof entry.content === 'string') return entry.content;
      if (Array.isArray(entry.content)) {
        return entry.content
          .filter((b: { type: string }) => b.type === 'text')
          .map((b: { text?: string }) => b.text ?? '')
          .join('\n')
          .trim();
      }
    }
  } catch { /* silent */ }
  return '';
}

function getTopicFromGit(): string {
  try {
    const branch = execSync('git branch --show-current', {
      encoding: 'utf-8',
      stdio: ['ignore', 'pipe', 'ignore'],
    }).trim();
    if (!branch || branch === 'main' || branch === 'master') return 'general';
    const parts = branch.split('/');
    const slug = parts.length > 1 ? parts[1] : branch;
    return slug.split('-')[0] || 'general';
  } catch {
    return 'general';
  }
}

function today(): string {
  return new Date().toISOString().split('T')[0];
}

function serialize(vector: number[]): Buffer {
  const buf = Buffer.allocUnsafe(vector.length * 4);
  new Float32Array(buf.buffer).set(vector);
  return buf;
}

async function saveMemory(title: string, topic: string, content: string): Promise<void> {
  // Embed content once — used for both duplicate check and indexing
  const extractor = await pipeline('feature-extraction', EMBED_MODEL, { dtype: 'fp32' });
  const out = await extractor(content, { pooling: 'mean', normalize: true });
  const embedding = serialize(Array.from(out.data as Float32Array));

  // Duplicate check — skip if near-identical memory already exists
  if (existsSync(DB_PATH)) {
    const dbCheck = new Database(DB_PATH, { readonly: true });
    sqliteVec.load(dbCheck);
    const rows = dbCheck.prepare(`
      SELECT distance FROM memory_embeddings
      WHERE embedding MATCH ? AND k = 1
      ORDER BY distance
    `).all(embedding, 1) as Array<{ distance: number }>;
    dbCheck.close();
    if (rows.length > 0 && rows[0].distance < DUPLICATE_THRESHOLD) return;
  }

  // Write markdown file
  const topicDir = join(RAW_DIR, topic);
  mkdirSync(topicDir, { recursive: true });
  const slug = title.toLowerCase().replace(/[^a-z0-9]+/g, '-').slice(0, 50);
  const filename = `${today()}-${slug}.md`;
  const relPath = `${topic}/${filename}`;

  writeFileSync(join(topicDir, filename), `---
title: ${title}
topic: ${topic}
date: ${today()}
source: auto
---

${content}
`, 'utf-8');

  // Index into DB
  if (!existsSync(DB_PATH)) return;

  const db = new Database(DB_PATH);
  sqliteVec.load(db);
  const { lastInsertRowid } = db.prepare(
    'INSERT INTO memories (path, title, tags, topic, chunk) VALUES (?, ?, ?, ?, ?)'
  ).run(relPath, title, 'auto', topic, content);
  db.prepare(
    'INSERT INTO memory_embeddings (id, embedding) VALUES (?, ?)'
  ).run(lastInsertRowid, embedding);
  db.close();
}

async function main(): Promise<void> {
  // Read hook JSON from stdin
  const chunks: Buffer[] = [];
  for await (const chunk of process.stdin) chunks.push(chunk as Buffer);
  const raw = Buffer.concat(chunks).toString('utf-8').trim();
  if (!raw) return;

  let transcriptPath = '';
  try {
    transcriptPath = JSON.parse(raw).transcript_path ?? '';
  } catch { return; }

  if (!transcriptPath || !existsSync(transcriptPath)) return;

  const lastResponse = getLastAssistantMessage(transcriptPath);
  if (!lastResponse || lastResponse.length < 100) return;

  // Fast pre-filter — exit immediately on most responses
  if (!hasSignal(lastResponse)) return;

  // Haiku decides if it's genuinely worth saving
  const client = new Anthropic();
  const result = await client.messages.create({
    model: HAIKU,
    max_tokens: 200,
    messages: [{
      role: 'user',
      content: `Does this response contain a non-obvious technical learning worth saving?

SAVE: non-obvious discoveries, bug root causes, patterns, constraints, gotchas, non-obvious reasoning.
SKIP: routine code generation, obvious explanations, status updates, conversational filler.

Response:
${lastResponse.slice(0, 2000)}

JSON only:
{"worth_saving": true, "title": "under 8 words", "content": "1-3 sentences max"}
or
{"worth_saving": false}`,
    }],
  });

  const text = result.content[0].type === 'text' ? result.content[0].text.trim() : '';
  if (!text) return;

  let parsed: { worth_saving: boolean; title?: string; content?: string };
  try {
    parsed = JSON.parse(text);
  } catch { return; }

  if (!parsed.worth_saving || !parsed.title || !parsed.content) return;

  const topic = getTopicFromGit();
  await saveMemory(parsed.title, topic, parsed.content);
}

main().catch(() => { /* never surface errors */ });
