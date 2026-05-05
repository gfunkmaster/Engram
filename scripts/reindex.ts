#!/usr/bin/env tsx
/**
 * Rebuild the semantic memory index from memory/raw/.
 * Run once on any new machine after cloning, or after schema changes.
 *
 * Usage:
 *   npm run reindex
 *   npx tsx scripts/reindex.ts
 */

import Database from 'better-sqlite3';
import * as sqliteVec from 'sqlite-vec';
import { pipeline } from '@huggingface/transformers';
import { readFileSync, readdirSync, statSync } from 'fs';
import { join, relative, dirname } from 'path';
import { fileURLToPath } from 'url';
import { chunkText, serialize } from '../lib/memory.ts';
import type { MemoryTier } from '../lib/memory.ts';

const ENGRAM_DIR = join(dirname(fileURLToPath(import.meta.url)), '..');
const MEMORY_DIR = join(ENGRAM_DIR, 'memory', 'raw');
const DB_PATH = join(ENGRAM_DIR, 'memory', 'memory.db');
const MODEL = 'Xenova/all-MiniLM-L6-v2';

interface MemoryRow {
  path: string;
  title: string;
  tags: string;
  topic: string;
  chunk: string;
  session_id: string | null;
  memory_tier: MemoryTier;
  project_scope: string | null;
}

function extractFrontmatter(content: string): { meta: Record<string, string>; body: string } {
  const meta: Record<string, string> = {};
  let body = content;

  if (content.startsWith('---')) {
    const end = content.indexOf('---', 3);
    if (end !== -1) {
      const fm = content.slice(3, end).trim();
      body = content.slice(end + 3).trim();
      for (const line of fm.split('\n')) {
        const colon = line.indexOf(':');
        if (colon !== -1) {
          meta[line.slice(0, colon).trim()] = line.slice(colon + 1).trim();
        }
      }
    }
  }

  return { meta, body };
}

function walkMarkdown(dir: string): string[] {
  const files: string[] = [];
  try {
    for (const entry of readdirSync(dir)) {
      const full = join(dir, entry);
      if (statSync(full).isDirectory()) {
        files.push(...walkMarkdown(full));
      } else if (entry.endsWith('.md')) {
        files.push(full);
      }
    }
  } catch { /* directory may not exist yet */ }
  return files;
}

async function main() {
  console.log('Loading model...');
  const extractor = await pipeline('feature-extraction', MODEL, { dtype: 'fp32' });

  const db = new Database(DB_PATH);
  sqliteVec.load(db);

  db.exec(`
    DROP TABLE IF EXISTS memory_embeddings;
    DROP TABLE IF EXISTS memories;

    CREATE TABLE memories (
      id              INTEGER PRIMARY KEY,
      path            TEXT NOT NULL,
      title           TEXT,
      tags            TEXT,
      topic           TEXT,
      chunk           TEXT NOT NULL,
      session_id      TEXT,
      source_excerpt  TEXT,
      created_at      INTEGER DEFAULT (unixepoch()),
      supersedes      INTEGER,
      superseded_by   INTEGER,
      is_active       INTEGER DEFAULT 1,
      memory_tier     TEXT DEFAULT 'short',
      project_scope   TEXT,
      confidence      REAL DEFAULT 1.0,
      decay_rate      REAL DEFAULT 0.02,
      access_count    INTEGER DEFAULT 0,
      last_accessed_at INTEGER
    );

    CREATE VIRTUAL TABLE memory_embeddings USING vec0(
      id        INTEGER PRIMARY KEY,
      embedding float[384]
    );
  `);

  const insertMemory = db.prepare(`
    INSERT INTO memories
      (path, title, tags, topic, chunk, session_id, is_active, memory_tier, project_scope)
    VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)
  `);
  const insertEmbedding = db.prepare(
    'INSERT INTO memory_embeddings (id, embedding) VALUES (?, ?)'
  );

  const files = walkMarkdown(MEMORY_DIR);
  console.log(`Indexing ${files.length} files...\n`);

  const insertBatch = db.transaction((rows: MemoryRow[], embeddings: Buffer[]) => {
    for (let i = 0; i < rows.length; i++) {
      const r = rows[i];
      const { lastInsertRowid } = insertMemory.run(
        r.path, r.title, r.tags, r.topic, r.chunk, r.session_id,
        r.memory_tier, r.project_scope
      );
      insertEmbedding.run(lastInsertRowid, embeddings[i]);
    }
  });

  for (const file of files) {
    const content = readFileSync(file, 'utf-8');
    const { meta, body } = extractFrontmatter(content);
    const relPath = relative(MEMORY_DIR, file);
    const topic = meta['topic'] ?? file.split('/').at(-2) ?? 'general';
    const title = meta['title'] ?? file.split('/').at(-1)?.replace('.md', '') ?? '';
    const tags = meta['tags'] ?? '';
    const sessionId = meta['session_id'] ?? null;
    const tier = (meta['tier'] === 'long' ? 'long' : 'short') as MemoryTier;
    const projectScope = meta['project_scope'] ? meta['project_scope'] : null;

    const chunks = chunkText(body).filter(c => c.trim());
    const rows: MemoryRow[] = [];
    const embeddings: Buffer[] = [];

    for (const c of chunks) {
      const output = await extractor(c, { pooling: 'mean', normalize: true });
      rows.push({ path: relPath, title, tags, topic, chunk: c, session_id: sessionId, memory_tier: tier, project_scope: projectScope });
      embeddings.push(serialize(Array.from(output.data as Float32Array)));
    }

    insertBatch(rows, embeddings);
    const tierLabel = tier === 'long' ? '[long]' : '[short]';
    console.log(`  ✓ ${relPath} ${tierLabel} (${chunks.length} chunk${chunks.length !== 1 ? 's' : ''})`);
  }

  db.close();
  console.log(`\nDone. ${files.length} files indexed → ${DB_PATH}`);
}

main().catch(console.error);
