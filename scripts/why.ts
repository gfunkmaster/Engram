#!/usr/bin/env tsx
/**
 * Debug CLI — shows the full retrieval pipeline for a query.
 * Reveals what would be injected, what was filtered, and why.
 *
 * Usage:
 *   npm run why -- "JWT refresh flow"
 *   npx tsx scripts/why.ts "how did we handle auth"
 */

import { searchAll, INJECTION_THRESHOLD } from '../lib/memory.ts';

const RESET  = '\x1b[0m';
const BOLD   = '\x1b[1m';
const DIM    = '\x1b[2m';
const GREEN  = '\x1b[32m';
const YELLOW = '\x1b[33m';
const RED    = '\x1b[31m';
const CYAN   = '\x1b[36m';
const GREY   = '\x1b[90m';

function bar(distance: number, width = 30): string {
  const filled = Math.round((1 - distance) * width);
  return GREEN + '█'.repeat(filled) + GREY + '░'.repeat(width - filled) + RESET;
}

async function main() {
  const query = process.argv.slice(2).join(' ');

  if (!query) {
    console.error('Usage: npx tsx scripts/why.ts "<query>"');
    process.exit(1);
  }

  console.log(`\n${BOLD}Engram retrieval trace${RESET}`);
  console.log(`${DIM}Query: "${query}"${RESET}`);
  console.log(`${DIM}Threshold: ${INJECTION_THRESHOLD} (lower distance = more similar)${RESET}\n`);

  console.log('Searching...');
  const results = await searchAll(query, 20);

  if (results.length === 0) {
    console.log('\nNo memories found. Run npm run reindex if you have memories stored.');
    return;
  }

  const willInject  = results.filter(r => r.distance < INJECTION_THRESHOLD && r.is_active === 1);
  const filtered    = results.filter(r => r.distance >= INJECTION_THRESHOLD && r.is_active === 1);
  const superseded  = results.filter(r => r.is_active === 0);

  // Would inject
  if (willInject.length > 0) {
    console.log(`${BOLD}${GREEN}✓ Would inject (${willInject.length})${RESET}\n`);
    for (const r of willInject) {
      console.log(`  ${bar(r.distance)}  ${r.distance.toFixed(4)}  ${BOLD}${r.title}${RESET} ${GREY}(${r.topic})${RESET}`);
      console.log(`  ${GREY}${r.path}${RESET}`);
      console.log(`  ${r.chunk.slice(0, 200).replace(/\n/g, ' ')}${r.chunk.length > 200 ? '...' : ''}\n`);
    }
  } else {
    console.log(`${YELLOW}✗ Nothing would be injected — no results below threshold ${INJECTION_THRESHOLD}${RESET}\n`);
  }

  // Filtered out
  if (filtered.length > 0) {
    console.log(`${BOLD}${YELLOW}◌ Filtered out — below threshold (${filtered.length})${RESET}\n`);
    for (const r of filtered) {
      console.log(`  ${bar(r.distance)}  ${r.distance.toFixed(4)}  ${DIM}${r.title} (${r.topic})${RESET}`);
    }
    console.log();
  }

  // Superseded
  if (superseded.length > 0) {
    console.log(`${BOLD}${RED}⊘ Superseded — skipped (${superseded.length})${RESET}\n`);
    for (const r of superseded) {
      console.log(`  ${DIM}${r.distance.toFixed(4)}  ${r.title} (${r.topic}) → replaced by id:${r.superseded_by}${RESET}`);
    }
    console.log();
  }

  console.log(`${CYAN}Summary: ${willInject.length} inject / ${filtered.length} filtered / ${superseded.length} superseded out of ${results.length} candidates${RESET}\n`);
}

main().catch(console.error);
