#!/usr/bin/env tsx
/**
 * Write a new memory and immediately index it.
 * Called by skills at the end of every pipeline run.
 *
 * Usage:
 *   npm run remember -- --topic "auth" --title "JWT refresh flow" --tags "auth,jwt"
 *   cat output.md | npx tsx scripts/remember.ts --topic "auth" --title "JWT refresh flow"
 *   npx tsx scripts/remember.ts --topic "auth" --title "JWT refresh flow" --file output.md
 */

import Database from 'better-sqlite3';
import * as sqliteVec from 'sqlite-vec';
import { pipeline } from '@huggingface/transformers';
import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'fs';
import { join } from 'path';
import { homedir } from 'os';

const RAW_DIR = join(homedir(), '.claude', 'memory', 'raw');
const DB_PATH = join(homedir(), '.claude', 'memory', 'memory.db');
const MODEL = 'Xenova/all-MiniLM-L6-v2';
const CHUNK_SIZE = 512;

function serialize(vector: number[]): Buffer {
  const buf = Buffer.allocUnsafe(vector.length * 4);
  new Float32Array(buf.buffer).set(vector);
  return buf;
}

function chunk(text: string): string[] {
  const words = text.split(/\s+/);
  const chunks: string[] = [];
  for (let i = 0; i < words.length; i += CHUNK_SIZE) {
    chunks.push(words.slice(i, i + CHUNK_SIZE).join(' '));
  }
  return chunks.filter(c => c.trim());
}

function parseArgs() {
  const args = process.argv.slice(2);
  const get = (flag: string) => {
    const i = args.indexOf(flag);
    return i !== -1 ? args[i + 1] : undefined;
  };
  return {
    topic: get('--topic'),
    title: get('--title'),
    tags: get('--tags') ?? '',
    file: get('--file'),
  };
}

function today(): string {
  return new Date().toISOString().split('T')[0];
}

async function remember(topic: string, title: string, tags: string, content: string) {
  // Write markdown to raw/
  const topicDir = join(RAW_DIR, topic);
  mkdirSync(topicDir, { recursive: true });

  const slug = title.toLowerCase().replace(/[^a-z0-9]+/g, '-');
  const filename = `${today()}-${slug}.md`;
  const filepath = join(topicDir, filename);

  const markdown = `---
title: ${title}
topic: ${topic}
tags: ${tags}
date: ${today()}
---

${content}
`;
  writeFileSync(filepath, markdown, 'utf-8');
  console.log(`Saved: ${filepath}`);

  // Index it immediately
  if (!existsSync(DB_PATH)) {
    console.log('No index found — run npm run reindex first. File saved, will be included on next reindex.');
    return;
  }

  console.log('Indexing...');
  const extractor = await pipeline('feature-extraction', MODEL, { dtype: 'fp32' });
  const db = new Database(DB_PATH);
  sqliteVec.load(db);

  const insertMemory = db.prepare(
    'INSERT INTO memories (path, title, tags, topic, chunk) VALUES (?, ?, ?, ?, ?)'
  );
  const insertEmbedding = db.prepare(
    'INSERT INTO memory_embeddings (id, embedding) VALUES (?, ?)'
  );

  const relPath = `${topic}/${filename}`;
  const chunks = chunk(content);

  const insertAll = db.transaction(async () => {
    for (const c of chunks) {
      const output = await extractor(c, { pooling: 'mean', normalize: true });
      const vector = Array.from(output.data as Float32Array);
      const { lastInsertRowid } = insertMemory.run(relPath, title, tags, topic, c);
      insertEmbedding.run(lastInsertRowid, serialize(vector));
    }
  });

  await insertAll();
  db.close();
  console.log(`Indexed ${chunks.length} chunk(s).`);
}

async function main() {
  const { topic, title, tags, file } = parseArgs();

  if (!topic || !title) {
    console.error('Usage: tsx remember.ts --topic <topic> --title <title> [--tags <tags>] [--file <path>]');
    console.error('       Or pipe content via stdin.');
    process.exit(1);
  }

  let content: string;
  if (file) {
    content = readFileSync(file, 'utf-8');
  } else {
    const chunks: Buffer[] = [];
    for await (const chunk of process.stdin) chunks.push(chunk as Buffer);
    content = Buffer.concat(chunks).toString('utf-8');
  }

  if (!content.trim()) {
    console.error('No content provided.');
    process.exit(1);
  }

  await remember(topic, title, tags, content);
}

main().catch(console.error);
