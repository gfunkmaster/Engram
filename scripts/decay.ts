#!/usr/bin/env tsx
/**
 * Apply confidence decay to short-term memories.
 *
 * Each run multiplies confidence by (1 - decay_rate). Memories that fall below
 * the threshold (default: 0.10) are deactivated. Long-term memories are never
 * decayed. Run this on a schedule (e.g. daily cron or weekly manually).
 *
 * Usage:
 *   npm run decay               # dry-run — shows what would change
 *   npm run decay -- --apply    # write changes to the DB
 *   npm run decay -- --cutoff 0.05   # custom deactivation cutoff
 */

import Database from 'better-sqlite3';
import * as sqliteVec from 'sqlite-vec';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';
import { DB_PATH } from '../lib/memory.ts';

const DEACTIVATE_CUTOFF = 0.10;

const RESET  = '\x1b[0m';
const BOLD   = '\x1b[1m';
const GREEN  = '\x1b[32m';
const YELLOW = '\x1b[33m';
const RED    = '\x1b[31m';
const CYAN   = '\x1b[36m';
const DIM    = '\x1b[2m';
const GREY   = '\x1b[90m';

interface MemoryRow {
  id: number;
  title: string;
  topic: string;
  project_scope: string | null;
  confidence: number;
  decay_rate: number;
  access_count: number;
  created_at: number;
}

async function main() {
  const args = process.argv.slice(2);
  const apply = args.includes('--apply');
  const cutoffIdx = args.indexOf('--cutoff');
  const cutoff = cutoffIdx !== -1 ? parseFloat(args[cutoffIdx + 1]) : DEACTIVATE_CUTOFF;

  if (!apply) {
    console.log(`\n${YELLOW}Dry-run mode — pass --apply to commit changes${RESET}\n`);
  }

  const db = new Database(DB_PATH);
  sqliteVec.load(db);

  const memories = db.prepare(`
    SELECT id, title, topic, project_scope, confidence, decay_rate, access_count, created_at
    FROM memories
    WHERE is_active = 1
      AND memory_tier = 'short'
    ORDER BY confidence ASC
  `).all() as MemoryRow[];

  if (memories.length === 0) {
    console.log(`${GREY}No active short-term memories to decay.${RESET}\n`);
    db.close();
    return;
  }

  let decayed = 0;
  let deactivated = 0;

  console.log(`\n${BOLD}Short-term memory decay report${RESET}\n`);
  console.log(`${DIM}Deactivation cutoff: ${cutoff} | Processing ${memories.length} memories${RESET}\n`);

  for (const m of memories) {
    const newConfidence = m.confidence * (1 - m.decay_rate);
    const willDeactivate = newConfidence < cutoff;

    if (willDeactivate) {
      console.log(`  ${RED}✕${RESET} ${BOLD}${m.title}${RESET} ${GREY}(${m.topic})${RESET}`);
      console.log(`  ${DIM}confidence: ${m.confidence.toFixed(3)} → ${newConfidence.toFixed(3)} [DEACTIVATE]${RESET}`);
      if (apply) {
        db.prepare('UPDATE memories SET confidence = ?, is_active = 0 WHERE id = ?').run(newConfidence, m.id);
      }
      deactivated++;
    } else {
      const age = Math.round((Date.now() / 1000 - m.created_at) / 86400);
      const bar = _confidenceBar(newConfidence);
      console.log(`  ${bar}  ${m.confidence.toFixed(3)} → ${newConfidence.toFixed(3)}  ${DIM}${m.title} · ${age}d old${RESET}`);
      if (apply) {
        db.prepare('UPDATE memories SET confidence = ? WHERE id = ?').run(newConfidence, m.id);
      }
      decayed++;
    }
  }

  db.close();

  console.log(`\n${CYAN}Summary: ${decayed} decayed, ${deactivated} deactivated out of ${memories.length} short-term memories${RESET}`);
  if (!apply) console.log(`${YELLOW}Run with --apply to commit these changes.${RESET}`);
  console.log();
}

function _confidenceBar(confidence: number, width = 20): string {
  const filled = Math.round(confidence * width);
  const color = confidence > 0.5 ? GREEN : confidence > 0.25 ? YELLOW : RED;
  return color + '█'.repeat(filled) + GREY + '░'.repeat(width - filled) + RESET;
}

main().catch(console.error);
