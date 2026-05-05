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

// Distance thresholds (cosine distance — lower = more similar)
const DUPLICATE_THRESHOLD = 0.15;
const SUPERSESSION_THRESHOLD = 0.35;
export const INJECTION_THRESHOLD = 0.75;

const MIN_RESPONSE_LENGTH = 200;
export const PROMOTE_ACCESS_THRESHOLD = 3; // short→long after this many distinct accesses

export const SIGNAL_PHRASES = [
  // Discovery
  'turns out', 'it turns out', 'discovered that', 'found that', 'realized that',
  'interestingly', 'surprisingly', 'unexpectedly', 'what i found',
  // Problems & root causes
  'the issue was', 'the problem was', 'root cause', 'the bug was', 'the culprit',
  'what broke', 'why it failed', 'the reason it', 'the cause',
  // Solutions
  'fixed by', 'resolved by', 'solved by', 'the fix is', 'the solution is',
  'the workaround', 'what worked',
  // Patterns & insights
  'the trick is', 'the trick here', 'the key insight', 'important to note',
  'worth noting', 'the pattern here', 'the pattern is',
  // Constraints & warnings
  'always ensure', 'never do', 'avoid', 'make sure to', 'be careful',
  'watch out', 'non-obvious', 'counterintuitive', 'caveat', 'edge case',
  'the catch is', 'gotcha', 'pitfall',
  // Learnings
  'learned that', 'this means', 'the implication', 'takeaway', 'lesson',
  'what this means', 'worth remembering',
];

export type MemoryTier = 'short' | 'long';

export interface SearchResult {
  id: number;
  path: string;
  title: string;
  topic: string;
  chunk: string;
  distance: number;
  is_active: number;
  superseded_by: number | null;
  memory_tier: MemoryTier;
  project_scope: string | null;
  confidence: number;
  access_count: number;
}

export interface SaveOptions {
  sessionId?: string;
  sourceExcerpt?: string;
  tags?: string;
  tier?: MemoryTier;
  projectScope?: string | null;
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

/** Returns the git remote origin URL for the current working directory, or null. */
export function getProjectScope(): string | null {
  try {
    const remote = execSync('git remote get-url origin', {
      encoding: 'utf-8',
      stdio: ['ignore', 'pipe', 'ignore'],
    }).trim();
    return remote || null;
  } catch {
    return null;
  }
}

/**
 * Sliding window chunker with heading-aware splitting.
 */
export function chunkText(text: string, size = 400, overlap = 80): string[] {
  const sections = text.split(/(?=^#{1,3}\s)/m).filter(s => s.trim().length > 20);
  const chunks: string[] = [];

  for (const section of sections) {
    const words = section.split(/\s+/).filter(Boolean);
    if (words.length <= size) {
      chunks.push(section.trim());
      continue;
    }
    let i = 0;
    while (i < words.length) {
      const chunk = words.slice(i, i + size).join(' ');
      if (chunk.trim()) chunks.push(chunk);
      i += size - overlap;
      if (i + overlap >= words.length) break;
    }
  }

  return chunks.filter(c => c.trim().length > 20);
}

/**
 * Tier-aware semantic search.
 *
 * Returns long-term memories (global) + short-term memories scoped to the
 * current project. Unscoped short-term memories (project_scope IS NULL) are
 * included regardless. Access count is incremented for every returned result.
 */
export async function search(
  query: string,
  topK = 5,
  projectScope?: string | null
): Promise<SearchResult[]> {
  if (!existsSync(DB_PATH)) return [];

  const scope = projectScope !== undefined ? projectScope : getProjectScope();

  const extractor = await pipeline('feature-extraction', EMBED_MODEL, { dtype: 'fp32' });
  const out = await extractor(query, { pooling: 'mean', normalize: true });
  const embedding = serialize(Array.from(out.data as Float32Array));

  const db = new Database(DB_PATH);
  sqliteVec.load(db);

  // Fetch a wider candidate set and filter in JS — sqlite-vec applies KNN before WHERE filters,
  // so we ask for more than needed to avoid missing relevant tier-filtered results.
  const candidates = db.prepare(`
    SELECT m.id, m.path, m.title, m.topic, m.chunk, m.is_active, m.superseded_by,
           m.memory_tier, m.project_scope, m.confidence, m.access_count, e.distance
    FROM memory_embeddings e
    JOIN memories m ON m.id = e.id
    WHERE e.embedding MATCH ? AND k = ?
      AND m.is_active = 1
    ORDER BY e.distance
  `).all(embedding, topK * 4) as SearchResult[];

  // Filter: long-term is always global; short-term must match project scope
  const rows = candidates
    .filter(r =>
      r.memory_tier === 'long' ||
      r.project_scope === null ||
      (scope !== null && r.project_scope === scope)
    )
    .slice(0, topK);

  // Track access for all returned rows
  if (rows.length > 0) {
    const now = Math.floor(Date.now() / 1000);
    const update = db.prepare(
      'UPDATE memories SET access_count = access_count + 1, last_accessed_at = ? WHERE id = ?'
    );
    db.transaction(() => { for (const r of rows) update.run(now, r.id); })();
  }

  db.close();
  return rows;
}

/** Search including superseded memories — used by the why CLI. */
export async function searchAll(query: string, topK = 20): Promise<SearchResult[]> {
  if (!existsSync(DB_PATH)) return [];

  const extractor = await pipeline('feature-extraction', EMBED_MODEL, { dtype: 'fp32' });
  const out = await extractor(query, { pooling: 'mean', normalize: true });
  const embedding = serialize(Array.from(out.data as Float32Array));

  const db = new Database(DB_PATH, { readonly: true });
  sqliteVec.load(db);

  const rows = db.prepare(`
    SELECT m.id, m.path, m.title, m.topic, m.chunk, m.is_active, m.superseded_by,
           m.memory_tier, m.project_scope, m.confidence, m.access_count, e.distance
    FROM memory_embeddings e
    JOIN memories m ON m.id = e.id
    WHERE e.embedding MATCH ? AND k = ?
    ORDER BY e.distance
  `).all(embedding, topK) as SearchResult[];

  db.close();
  return rows;
}

/** Write a memory to disk, handle supersession, and index. */
export async function saveMemory(
  title: string,
  topic: string,
  content: string,
  opts: SaveOptions = {}
): Promise<void> {
  const tier: MemoryTier = opts.tier ?? 'short';
  const projectScope = opts.projectScope !== undefined
    ? opts.projectScope
    : (tier === 'short' ? getProjectScope() : null);

  const mergedOpts: SaveOptions & { tier: MemoryTier; projectScope: string | null } = {
    ...opts,
    tier,
    projectScope,
  };

  if (!existsSync(DB_PATH)) {
    _writeMarkdown(title, topic, content, mergedOpts);
    return;
  }

  const extractor = await pipeline('feature-extraction', EMBED_MODEL, { dtype: 'fp32' });
  const out = await extractor(content, { pooling: 'mean', normalize: true });
  const embedding = serialize(Array.from(out.data as Float32Array));

  const db = new Database(DB_PATH);
  sqliteVec.load(db);

  const candidates = db.prepare(`
    SELECT m.id, m.path, m.title, e.distance
    FROM memory_embeddings e
    JOIN memories m ON m.id = e.id
    WHERE e.embedding MATCH ? AND k = 5
      AND m.is_active = 1
    ORDER BY e.distance
  `).all(embedding, 5) as Array<{ id: number; path: string; title: string; distance: number }>;

  db.close();

  if (candidates.length > 0) {
    const nearest = candidates[0];
    if (nearest.distance < DUPLICATE_THRESHOLD) return;
    if (nearest.distance < SUPERSESSION_THRESHOLD) {
      const filepath = _writeMarkdown(title, topic, content, mergedOpts);
      _indexAndSupersede(title, topic, content, embedding, filepath, nearest.id, mergedOpts);
      return;
    }
  }

  const filepath = _writeMarkdown(title, topic, content, mergedOpts);
  _indexNew(title, topic, content, embedding, filepath, mergedOpts);
}

function _writeMarkdown(
  title: string,
  topic: string,
  content: string,
  opts: SaveOptions & { tier: MemoryTier; projectScope: string | null }
): string {
  const topicDir = join(RAW_DIR, topic);
  mkdirSync(topicDir, { recursive: true });
  const slug = title.toLowerCase().replace(/[^a-z0-9]+/g, '-').slice(0, 50);
  const filename = `${today()}-${slug}.md`;
  const filepath = join(topicDir, filename);

  writeFileSync(filepath, `---
title: ${title}
topic: ${topic}
tier: ${opts.tier}
project_scope: ${opts.projectScope ?? ''}
tags: ${opts.tags ?? 'auto'}
date: ${today()}
source: auto
session_id: ${opts.sessionId ?? ''}
---

${content}
`, 'utf-8');

  return `${topic}/${filename}`;
}

function _indexNew(
  title: string,
  topic: string,
  content: string,
  embedding: Buffer,
  relPath: string,
  opts: SaveOptions & { tier: MemoryTier; projectScope: string | null }
): void {
  if (!existsSync(DB_PATH)) return;
  const db = new Database(DB_PATH);
  sqliteVec.load(db);

  const { lastInsertRowid } = db.prepare(`
    INSERT INTO memories
      (path, title, tags, topic, chunk, session_id, source_excerpt,
       memory_tier, project_scope, confidence, decay_rate, is_active)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1.0, 0.02, 1)
  `).run(
    relPath, title, opts.tags ?? 'auto', topic, content,
    opts.sessionId ?? null, opts.sourceExcerpt ?? null,
    opts.tier, opts.projectScope
  );

  db.prepare('INSERT INTO memory_embeddings (id, embedding) VALUES (?, ?)').run(lastInsertRowid, embedding);
  db.close();
}

function _indexAndSupersede(
  title: string,
  topic: string,
  content: string,
  embedding: Buffer,
  relPath: string,
  supersededId: number,
  opts: SaveOptions & { tier: MemoryTier; projectScope: string | null }
): void {
  if (!existsSync(DB_PATH)) return;
  const db = new Database(DB_PATH);
  sqliteVec.load(db);

  const { lastInsertRowid } = db.prepare(`
    INSERT INTO memories
      (path, title, tags, topic, chunk, session_id, source_excerpt,
       memory_tier, project_scope, confidence, decay_rate, supersedes, is_active)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1.0, 0.02, ?, 1)
  `).run(
    relPath, title, opts.tags ?? 'auto', topic, content,
    opts.sessionId ?? null, opts.sourceExcerpt ?? null,
    opts.tier, opts.projectScope, supersededId
  );

  db.prepare('INSERT INTO memory_embeddings (id, embedding) VALUES (?, ?)').run(lastInsertRowid, embedding);
  db.prepare('UPDATE memories SET is_active = 0, superseded_by = ? WHERE id = ?').run(lastInsertRowid, supersededId);
  db.close();
}

/**
 * Use Haiku to decide if a response contains a learning worth saving.
 * Saves as short-term memory scoped to the current project.
 */
export async function autoRemember(
  responseText: string,
  topic?: string,
  sessionId?: string,
  projectScope?: string | null
): Promise<void> {
  if (!responseText || responseText.length < MIN_RESPONSE_LENGTH) return;
  if (!hasSignal(responseText)) return;

  const client = new Anthropic();
  const result = await client.messages.create({
    model: HAIKU,
    max_tokens: 200,
    messages: [{
      role: 'user',
      content: `Does this response contain a non-obvious technical learning worth saving to long-term memory?

SAVE: non-obvious discoveries, bug root causes, patterns, constraints, gotchas, non-obvious decisions.
SKIP: routine code generation, obvious explanations, status updates, conversational filler.

Response:
${responseText.slice(0, 2000)}

JSON only — no other text:
{"worth_saving": true, "title": "under 8 words", "content": "1-3 sentences", "excerpt": "verbatim sentence that triggered this"}
or
{"worth_saving": false}`,
    }],
  });

  const text = result.content[0].type === 'text' ? result.content[0].text.trim() : '';
  if (!text) return;

  let parsed: { worth_saving: boolean; title?: string; content?: string; excerpt?: string };
  try { parsed = JSON.parse(text); } catch { return; }
  if (!parsed.worth_saving || !parsed.title || !parsed.content) return;

  const scope = projectScope !== undefined ? projectScope : getProjectScope();

  await saveMemory(parsed.title, topic ?? getTopicFromGit(), parsed.content, {
    sessionId,
    sourceExcerpt: parsed.excerpt,
    tier: 'short',
    projectScope: scope,
  });
}
