#!/usr/bin/env tsx
/**
 * Rebuild the semantic memory index from ~/.claude/memory/raw/
 * Run once on any new machine after cloning your ~/.claude repo.
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

const ENGRAM_DIR = join(dirname(fileURLToPath(import.meta.url)), '..');
const MEMORY_DIR = join(ENGRAM_DIR, 'memory', 'raw');
const DB_PATH = join(ENGRAM_DIR, 'memory', 'memory.db');
const MODEL = 'Xenova/all-MiniLM-L6-v2';
const CHUNK_SIZE = 512;

interface Memory {
  path: string;
  title: string;
  tags: string;
  topic: string;
  chunk: string;
}

function serialize(vector: number[]): Buffer {
  const buf = Buffer.allocUnsafe(vector.length * 4);
  new Float32Array(buf.buffer).set(vector);
  return buf;
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

function chunk(text: string): string[] {
  const words = text.split(/\s+/);
  const chunks: string[] = [];
  for (let i = 0; i < words.length; i += CHUNK_SIZE) {
    chunks.push(words.slice(i, i + CHUNK_SIZE).join(' '));
  }
  return chunks;
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
  } catch {
    // directory may not exist yet
  }
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
      id    INTEGER PRIMARY KEY,
      path  TEXT NOT NULL,
      title TEXT,
      tags  TEXT,
      topic TEXT,
      chunk TEXT NOT NULL
    );

    CREATE VIRTUAL TABLE memory_embeddings USING vec0(
      id        INTEGER PRIMARY KEY,
      embedding float[384]
    );
  `);

  const insertMemory = db.prepare(
    'INSERT INTO memories (path, title, tags, topic, chunk) VALUES (?, ?, ?, ?, ?)'
  );
  const insertEmbedding = db.prepare(
    'INSERT INTO memory_embeddings (id, embedding) VALUES (?, ?)'
  );

  const files = walkMarkdown(MEMORY_DIR);
  console.log(`Indexing ${files.length} files...`);

  const insertMany = db.transaction(async (memories: Memory[], embeddings: Buffer[]) => {
    for (let i = 0; i < memories.length; i++) {
      const m = memories[i];
      const { lastInsertRowid } = insertMemory.run(m.path, m.title, m.tags, m.topic, m.chunk);
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

    const chunks = chunk(body).filter(c => c.trim());
    const memories: Memory[] = [];
    const embeddings: Buffer[] = [];

    for (const c of chunks) {
      const output = await extractor(c, { pooling: 'mean', normalize: true });
      const vector = Array.from(output.data as Float32Array);
      memories.push({ path: relPath, title, tags, topic, chunk: c });
      embeddings.push(serialize(vector));
    }

    await insertMany(memories, embeddings);
    console.log(`  ✓ ${relPath} (${chunks.length} chunks)`);
  }

  db.close();
  console.log(`\nDone. Index saved to ${DB_PATH}`);
}

main().catch(console.error);
