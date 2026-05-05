/**
 * Core Engram memory primitives.
 * Shared by scripts, hooks, and the client wrapper.
 */

import Database from 'better-sqlite3';
import * as sqliteVec from 'sqlite-vec';
import { pipeline } from '@huggingface/transformers';
import Anthropic from '@anthropic-ai/sdk';
import { writeFileSync, mkdirSync, existsSync } from 'fs';
import { join, dirname, basename } from 'path';
import { fileURLToPath } from 'url';
import { execSync } from 'child_process';
import { ensureSchema } from './migrate.ts';

const ENGRAM_DIR = join(dirname(fileURLToPath(import.meta.url)), '..');
export const RAW_DIR = join(ENGRAM_DIR, 'memory', 'raw');
export const DB_PATH = join(ENGRAM_DIR, 'memory', 'memory.db');

const EMBED_MODEL = 'Xenova/all-MiniLM-L6-v2';
// Task 10: allow override via env var
const HAIKU = process.env.ENGRAM_MODEL ?? 'claude-haiku-4-5-20251001';

// Distance thresholds (cosine distance — lower = more similar)
export const DUPLICATE_THRESHOLD = 0.15;
export const SUPERSESSION_THRESHOLD = 0.35;
export const INJECTION_THRESHOLD = 0.75;

const MIN_RESPONSE_LENGTH = 200;
export const PROMOTE_ACCESS_THRESHOLD = parseInt(process.env.ENGRAM_PROMOTE_THRESHOLD ?? '10', 10);

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

// Task 6: sanitize topic to prevent path traversal
export function sanitizeTopic(topic: string): string {
  const sanitized = topic.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
  if (!sanitized || sanitized.includes('..')) return 'general';
  return sanitized;
}

export function getTopicFromGit(): string {
  try {
    const branch = execSync('git branch --show-current', {
      encoding: 'utf-8',
      stdio: ['ignore', 'pipe', 'ignore'],
    }).trim();

    if (!branch || branch === 'main' || branch === 'master') {
      // On main/master, use the repo name as the topic
      const repoPath = execSync('git rev-parse --show-toplevel', {
        encoding: 'utf-8',
        stdio: ['ignore', 'pipe', 'ignore'],
      }).trim();
      return sanitizeTopic(basename(repoPath));
    }

    // Use the full branch name sanitized — ESHM-1234-fix-jwt → eshm-1234-fix-jwt
    return sanitizeTopic(branch);
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
      // Task 1: fix dropping the last segment — was `if (i + overlap >= words.length) break;`
      if (i >= words.length) break;
    }
  }

  return chunks.filter(c => c.trim().length > 20);
}

// Task 3: helper to strip markdown code fences before JSON.parse
export function stripJsonFences(text: string): string {
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/);
  return fenced ? fenced[1].trim() : text.trim();
}

// Task 22: decision logic extracted for testability
export type SaveDecision = 'skip' | 'new' | { supersede: number };
export function decideSave(candidates: Array<{ id: number; distance: number }>): SaveDecision {
  if (candidates.length === 0) return 'new';
  const nearest = candidates[0];
  if (nearest.distance < DUPLICATE_THRESHOLD) return 'skip';
  if (nearest.distance < SUPERSESSION_THRESHOLD) return { supersede: nearest.id };
  return 'new';
}

/** Open the DB, load sqlite-vec, and run schema migrations. */
function openDb(path: string = DB_PATH, options?: Database.Options): Database.Database {
  const db = new Database(path, options);
  sqliteVec.load(db);
  ensureSchema(db);
  return db;
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

  // Two separate queries guarantee best results from each tier regardless of mix.
  // sqlite-vec applies KNN before WHERE filters, so a single combined query can miss
  // relevant short-term results when long-term memories dominate the top-K candidates.
  const kPerTier = Math.max(topK * 2, 20);
  const db = openDb(DB_PATH, { readonly: true });

  const SELECT = `
    SELECT m.id, m.path, m.title, m.topic, m.chunk, m.is_active, m.superseded_by,
           m.memory_tier, m.project_scope, m.confidence, m.access_count, e.distance
    FROM memory_embeddings e
    JOIN memories m ON m.id = e.id
  `;

  // Query 1: long-term (global, no scope filter)
  const longTerm = db.prepare(`
    ${SELECT}
    WHERE e.embedding MATCH ? AND k = ?
      AND m.is_active = 1 AND m.memory_tier = 'long'
    ORDER BY e.distance
  `).all(embedding, kPerTier) as SearchResult[];

  // Query 2: short-term scoped to current project (or unscoped)
  const shortTerm = scope !== null
    ? db.prepare(`
        ${SELECT}
        WHERE e.embedding MATCH ? AND k = ?
          AND m.is_active = 1 AND m.memory_tier = 'short'
          AND (m.project_scope = ? OR m.project_scope IS NULL)
        ORDER BY e.distance
      `).all(embedding, kPerTier, scope) as SearchResult[]
    : db.prepare(`
        ${SELECT}
        WHERE e.embedding MATCH ? AND k = ?
          AND m.is_active = 1 AND m.memory_tier = 'short'
        ORDER BY e.distance
      `).all(embedding, kPerTier) as SearchResult[];

  db.close();

  // Merge, deduplicate by id, sort by distance
  const seen = new Set<number>();
  const rows: SearchResult[] = [];
  for (const r of [...longTerm, ...shortTerm].sort((a, b) => a.distance - b.distance)) {
    if (!seen.has(r.id)) { seen.add(r.id); rows.push(r); }
  }
  const result = rows.slice(0, topK);

  // Write phase — separate write DB only when there are results to track
  if (result.length > 0) {
    const writeDb = openDb(DB_PATH);
    const now = Math.floor(Date.now() / 1000);
    const update = writeDb.prepare(
      'UPDATE memories SET access_count = access_count + 1, last_accessed_at = ? WHERE id = ?'
    );
    writeDb.transaction(() => { for (const r of result) update.run(now, r.id); })();
    writeDb.close();
  }

  return result;
}

/** Search including superseded memories — used by the why CLI. */
export async function searchAll(query: string, topK = 20): Promise<SearchResult[]> {
  if (!existsSync(DB_PATH)) return [];

  const extractor = await pipeline('feature-extraction', EMBED_MODEL, { dtype: 'fp32' });
  const out = await extractor(query, { pooling: 'mean', normalize: true });
  const embedding = serialize(Array.from(out.data as Float32Array));

  const db = openDb(DB_PATH, { readonly: true });

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

  // Task 14: single DB connection with WAL mode
  const db = openDb(DB_PATH);
  db.pragma('journal_mode = WAL');

  try {
    let supersededId: number | null = null;
    let shouldSkip = false;

    // Read + decide inside a transaction
    db.transaction(() => {
      const candidates = db.prepare(`
        SELECT m.id, e.distance FROM memory_embeddings e
        JOIN memories m ON m.id = e.id
        WHERE e.embedding MATCH ? AND k = 5 AND m.is_active = 1
        ORDER BY e.distance
      `).all(embedding) as Array<{ id: number; distance: number }>;

      const decision = decideSave(candidates);
      if (decision === 'skip') { shouldSkip = true; return; }
      if (decision !== 'new') { supersededId = (decision as { supersede: number }).supersede; }
    })();

    if (shouldSkip) return;

    // File I/O outside transaction (unavoidable)
    const filepath = _writeMarkdown(title, topic, content, mergedOpts);

    // Write + supersede inside a transaction
    db.transaction(() => {
      const { lastInsertRowid } = db.prepare(`
        INSERT INTO memories
          (path, title, tags, topic, chunk, session_id, source_excerpt,
           memory_tier, project_scope, confidence, decay_rate, supersedes, is_active, file_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1.0, 0.02, ?, 1, NULL)
      `).run(
        filepath, title, mergedOpts.tags ?? 'auto', topic, content,
        mergedOpts.sessionId ?? null, mergedOpts.sourceExcerpt ?? null,
        mergedOpts.tier, mergedOpts.projectScope, supersededId
      );

      db.prepare('INSERT INTO memory_embeddings (id, embedding) VALUES (?, ?)').run(lastInsertRowid, embedding);

      if (supersededId !== null) {
        db.prepare('UPDATE memories SET is_active = 0, superseded_by = ? WHERE id = ?').run(lastInsertRowid, supersededId);
      }
    })();
  } finally {
    db.close();
  }
}

function _writeMarkdown(
  title: string,
  topic: string,
  content: string,
  opts: SaveOptions & { tier: MemoryTier; projectScope: string | null }
): string {
  // Task 6: sanitize topic for directory/path construction
  const safeTopic = sanitizeTopic(topic);
  const topicDir = join(RAW_DIR, safeTopic);
  mkdirSync(topicDir, { recursive: true });
  const slug = title.toLowerCase().replace(/[^a-z0-9]+/g, '-').slice(0, 50);
  // Task 2: add timestamp suffix to prevent filename collisions
  const filename = `${today()}-${slug}-${Date.now().toString(36).slice(-6)}.md`;
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
confidence: 1.0
access_count: 0
---

${content}
`, 'utf-8');

  return `${safeTopic}/${filename}`;
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

  // Task 9: fail visibly if API key is missing
  if (!process.env.ANTHROPIC_API_KEY) {
    process.stderr.write('[Engram] ANTHROPIC_API_KEY not set — auto-remember disabled\n');
    return;
  }

  // Task 26: allow disabling all Haiku API calls
  if (process.env.ENGRAM_DISABLE_HAIKU === '1') return;

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
  // Task 3: strip markdown code fences before parsing
  try { parsed = JSON.parse(stripJsonFences(text)); } catch { return; }
  if (!parsed.worth_saving || !parsed.title || !parsed.content) return;

  const scope = projectScope !== undefined ? projectScope : getProjectScope();

  await saveMemory(parsed.title, topic ?? getTopicFromGit(), parsed.content, {
    sessionId,
    sourceExcerpt: parsed.excerpt,
    tier: 'short',
    projectScope: scope,
  });
}
