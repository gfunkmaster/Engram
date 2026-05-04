/**
 * Core Engram memory primitives.
 * Shared by scripts, hooks, and the client wrapper.
 */

import Database from 'better-sqlite3';
import * as sqliteVec from 'sqlite-vec';
import { pipeline } from '@huggingface/transformers';
import Anthropic from '@anthropic-ai/sdk';
import { writeFileSync, mkdirSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { execSync } from 'child_process';

const ENGRAM_DIR = join(dirname(fileURLToPath(import.meta.url)), '..');
export const RAW_DIR = join(ENGRAM_DIR, 'memory', 'raw');
export const DB_PATH = join(ENGRAM_DIR, 'memory', 'memory.db');

const EMBED_MODEL = 'Xenova/all-MiniLM-L6-v2';
const HAIKU = 'claude-haiku-4-5-20251001';
const DUPLICATE_THRESHOLD = 0.2;

export const SIGNAL_PHRASES = [
  'turns out', 'it turns out', 'discovered that', 'found that', 'realized that',
  'the issue was', 'the problem was', 'root cause', 'important to note',
  'worth noting', 'the key insight', 'the trick is', 'learned that', 'the gotcha',
  'fixed by', 'resolved by', 'solved by', 'always ensure', 'never do',
  'non-obvious', 'counterintuitive', 'caveat', 'edge case', 'the catch is',
];

export interface SearchResult {
  path: string;
  title: string;
  topic: string;
  chunk: string;
  distance: number;
}

export function serialize(vector: number[]): Buffer {
  const buf = Buffer.allocUnsafe(vector.length * 4);
  new Float32Array(buf.buffer).set(vector);
  return buf;
}

export function today(): string {
  return new Date().toISOString().split('T')[0];
}

export function hasSignal(text: string): boolean {
  const lower = text.toLowerCase();
  return SIGNAL_PHRASES.some(p => lower.includes(p));
}

export function getTopicFromGit(): string {
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

/** Semantic search over Engram memory. */
export async function search(query: string, topK = 5): Promise<SearchResult[]> {
  if (!existsSync(DB_PATH)) return [];

  const extractor = await pipeline('feature-extraction', EMBED_MODEL, { dtype: 'fp32' });
  const out = await extractor(query, { pooling: 'mean', normalize: true });
  const embedding = serialize(Array.from(out.data as Float32Array));

  const db = new Database(DB_PATH, { readonly: true });
  sqliteVec.load(db);

  const rows = db.prepare(`
    SELECT m.path, m.title, m.topic, m.chunk, e.distance
    FROM memory_embeddings e
    JOIN memories m ON m.id = e.id
    WHERE e.embedding MATCH ? AND k = ?
    ORDER BY e.distance
  `).all(embedding, topK) as SearchResult[];

  db.close();
  return rows;
}

/** Write a memory to disk and index it immediately. */
export async function saveMemory(title: string, topic: string, content: string): Promise<void> {
  const extractor = await pipeline('feature-extraction', EMBED_MODEL, { dtype: 'fp32' });
  const out = await extractor(content, { pooling: 'mean', normalize: true });
  const embedding = serialize(Array.from(out.data as Float32Array));

  // Duplicate check
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

  // Write markdown
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

  // Index
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

/**
 * Use Haiku to decide if a response contains a learning worth saving.
 * Fires in background — never throws, never blocks.
 */
export async function autoRemember(responseText: string, topic?: string): Promise<void> {
  if (!responseText || responseText.length < 100) return;
  if (!hasSignal(responseText)) return;

  const client = new Anthropic();
  const result = await client.messages.create({
    model: HAIKU,
    max_tokens: 200,
    messages: [{
      role: 'user',
      content: `Does this response contain a non-obvious technical learning worth saving?

SAVE: non-obvious discoveries, bug root causes, patterns, constraints, gotchas.
SKIP: routine code, obvious explanations, status updates, conversational filler.

Response:
${responseText.slice(0, 2000)}

JSON only:
{"worth_saving": true, "title": "under 8 words", "content": "1-3 sentences"}
or
{"worth_saving": false}`,
    }],
  });

  const text = result.content[0].type === 'text' ? result.content[0].text.trim() : '';
  if (!text) return;

  let parsed: { worth_saving: boolean; title?: string; content?: string };
  try { parsed = JSON.parse(text); } catch { return; }

  if (!parsed.worth_saving || !parsed.title || !parsed.content) return;

  await saveMemory(parsed.title, topic ?? getTopicFromGit(), parsed.content);
}
