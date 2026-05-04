#!/usr/bin/env tsx
/**
 * Claude Code UserPromptSubmit hook — auto-search.
 * Searches Engram memory on every prompt and injects relevant context.
 * Exits silently if nothing relevant found. Never blocks Claude.
 */

import { existsSync } from 'fs';
import { search } from '../lib/memory.ts';

const RELEVANCE_THRESHOLD = 0.5;

async function main(): Promise<void> {
  const chunks: Buffer[] = [];
  for await (const chunk of process.stdin) chunks.push(chunk as Buffer);
  const raw = Buffer.concat(chunks).toString('utf-8').trim();
  if (!raw) return;

  let prompt = '';
  try { prompt = JSON.parse(raw).prompt ?? ''; } catch { return; }
  if (prompt.length < 20) return;

  try {
    const results = await search(prompt, 3);
    const relevant = results.filter(r => r.distance < RELEVANCE_THRESHOLD);
    if (relevant.length === 0) return;

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
  } catch { /* silent */ }
}

main();
