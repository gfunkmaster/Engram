#!/usr/bin/env tsx
/**
 * Claude Code UserPromptSubmit hook.
 * Reads the user's prompt from stdin (JSON), searches Engram memory,
 * and outputs relevant context for Claude to use.
 *
 * Wired up in ~/.claude/settings.json — fires automatically on every prompt.
 * Exits silently if: no DB, short prompt, or no relevant results found.
 */

import Database from 'better-sqlite3';
import * as sqliteVec from 'sqlite-vec';
import { pipeline } from '@huggingface/transformers';
import { existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const ENGRAM_DIR = join(dirname(fileURLToPath(import.meta.url)), '..');
const DB_PATH = join(ENGRAM_DIR, 'memory', 'memory.db');
const MODEL = 'Xenova/all-MiniLM-L6-v2';
const TOP_K = 3;
const RELEVANCE_THRESHOLD = 0.5; // lower = more similar, only inject high-confidence results

function serialize(vector: number[]): Buffer {
  const buf = Buffer.allocUnsafe(vector.length * 4);
  new Float32Array(buf.buffer).set(vector);
  return buf;
}

async function main() {
  // Read stdin
  const chunks: Buffer[] = [];
  for await (const chunk of process.stdin) chunks.push(chunk as Buffer);
  const raw = Buffer.concat(chunks).toString('utf-8').trim();

  if (!raw) process.exit(0);

  let prompt = '';
  try {
    const input = JSON.parse(raw);
    prompt = input.prompt ?? '';
  } catch {
    process.exit(0);
  }

  // Skip trivial prompts
  if (prompt.length < 20) process.exit(0);

  // Skip if no index
  if (!existsSync(DB_PATH)) process.exit(0);

  try {
    const extractor = await pipeline('feature-extraction', MODEL, { dtype: 'fp32' });
    const output = await extractor(prompt, { pooling: 'mean', normalize: true });
    const vector = Array.from(output.data as Float32Array);

    const db = new Database(DB_PATH, { readonly: true });
    sqliteVec.load(db);

    const rows = db.prepare(`
      SELECT m.title, m.topic, m.chunk, e.distance
      FROM memory_embeddings e
      JOIN memories m ON m.id = e.id
      WHERE e.embedding MATCH ? AND k = ?
      ORDER BY e.distance
    `).all(serialize(vector), TOP_K) as Array<{
      title: string;
      topic: string;
      chunk: string;
      distance: number;
    }>;

    db.close();

    const relevant = rows.filter(r => r.distance < RELEVANCE_THRESHOLD);
    if (relevant.length === 0) process.exit(0);

    const lines = [
      '---',
      '## Relevant context from Engram memory',
      '',
      ...relevant.map(r =>
        `**${r.title}** *(${r.topic})*\n${r.chunk.slice(0, 400)}${r.chunk.length > 400 ? '...' : ''}`
      ),
      '---',
      '',
    ];

    process.stdout.write(lines.join('\n'));
  } catch {
    // Never block Claude — fail silently
    process.exit(0);
  }
}

main();
