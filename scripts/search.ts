#!/usr/bin/env tsx
/**
 * Semantic search over ~/.claude/memory/raw/
 * Called by skills before starting any research or analysis task.
 *
 * Usage:
 *   npm run search -- "how do we handle auth"
 *   npx tsx scripts/search.ts "JWT refresh flow" --top 3
 */

import Database from 'better-sqlite3';
import * as sqliteVec from 'sqlite-vec';
import { pipeline } from '@huggingface/transformers';
import { existsSync } from 'fs';
import { join } from 'path';
import { homedir } from 'os';

const DB_PATH = join(homedir(), '.claude', 'memory', 'memory.db');
const MODEL = 'Xenova/all-MiniLM-L6-v2';

interface SearchResult {
  path: string;
  title: string;
  topic: string;
  chunk: string;
  distance: number;
}

function serialize(vector: number[]): Buffer {
  const buf = Buffer.allocUnsafe(vector.length * 4);
  new Float32Array(buf.buffer).set(vector);
  return buf;
}

function parseArgs(): { query: string; top: number } {
  const args = process.argv.slice(2);
  const topIndex = args.indexOf('--top');
  const top = topIndex !== -1 ? parseInt(args[topIndex + 1], 10) : 5;
  const query = args.filter((_, i) => i !== topIndex && i !== topIndex + 1).join(' ');
  return { query, top };
}

async function search(query: string, topK: number): Promise<SearchResult[]> {
  const extractor = await pipeline('feature-extraction', MODEL, { dtype: 'fp32' });
  const output = await extractor(query, { pooling: 'mean', normalize: true });
  const vector = Array.from(output.data as Float32Array);

  const db = new Database(DB_PATH, { readonly: true });
  sqliteVec.load(db);

  const rows = db.prepare(`
    SELECT m.path, m.title, m.topic, m.chunk, e.distance
    FROM memory_embeddings e
    JOIN memories m ON m.id = e.id
    WHERE e.embedding MATCH ?
      AND k = ?
    ORDER BY e.distance
  `).all(serialize(vector), topK) as SearchResult[];

  db.close();
  return rows;
}

async function main() {
  const { query, top } = parseArgs();

  if (!query) {
    console.error('Usage: tsx search.ts "<query>" [--top N]');
    process.exit(1);
  }

  if (!existsSync(DB_PATH)) {
    console.error('No index found. Run: npm run reindex');
    process.exit(1);
  }

  const results = await search(query, top);

  if (results.length === 0) {
    console.log('No results found.');
    return;
  }

  for (const [i, r] of results.entries()) {
    console.log(`\n${'='.repeat(60)}`);
    console.log(`[${i + 1}] ${r.title}  (topic: ${r.topic})  score: ${r.distance.toFixed(4)}`);
    console.log(`    ${r.path}`);
    console.log(`\n${r.chunk.slice(0, 400)}${r.chunk.length > 400 ? '...' : ''}`);
  }
}

main().catch(console.error);
