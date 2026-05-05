/**
 * Thin client for the Engram daemon.
 * Falls back to direct lib/memory.ts calls if daemon is not running.
 */

import http from 'node:http';
import { search as directSearch, saveMemory as directSaveMemory, getProjectScope } from '../lib/memory.ts';
import type { SearchResult, SaveOptions } from '../lib/memory.ts';

const PORT = parseInt(process.env.ENGRAM_PORT ?? '7700', 10);

async function daemonRequest<T>(path: string, body: Record<string, unknown>): Promise<T> {
  return new Promise((resolve, reject) => {
    const data = JSON.stringify(body);
    const req = http.request({
      hostname: '127.0.0.1',
      port: PORT,
      path,
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(data) },
      timeout: 10000,
    }, res => {
      const chunks: Buffer[] = [];
      res.on('data', c => chunks.push(c as Buffer));
      res.on('end', () => {
        try { resolve(JSON.parse(Buffer.concat(chunks).toString('utf-8'))); }
        catch (e) { reject(e); }
      });
    });
    req.on('error', reject);
    req.on('timeout', () => { req.destroy(); reject(new Error('timeout')); });
    req.write(data);
    req.end();
  });
}

export async function daemonSearch(
  query: string,
  topK = 5,
  projectScope?: string | null
): Promise<SearchResult[]> {
  try {
    const scope = projectScope !== undefined ? projectScope : getProjectScope();
    return await daemonRequest<SearchResult[]>('/search', { query, topK, projectScope: scope });
  } catch {
    // Daemon not running — fall back to direct call
    return directSearch(query, topK, projectScope);
  }
}

export async function daemonSaveMemory(
  title: string,
  topic: string,
  content: string,
  opts: SaveOptions = {}
): Promise<void> {
  try {
    await daemonRequest<{ ok: boolean }>('/remember', { title, topic, content, opts });
  } catch {
    return directSaveMemory(title, topic, content, opts);
  }
}
