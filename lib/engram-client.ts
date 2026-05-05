/**
 * Engram-aware Anthropic client wrapper.
 *
 * Drop-in replacement for the Anthropic client. Every message.create call
 * automatically searches memory before the request and auto-remembers after.
 *
 * Usage:
 *   import { EngramClient } from './lib/engram-client.ts';
 *   const client = new EngramClient();
 *   const response = await client.messages.create({ ... }); // same API
 */

import Anthropic from '@anthropic-ai/sdk';
import { search, autoRemember, getProjectScope, INJECTION_THRESHOLD } from './memory.ts';
import type { SearchResult } from './memory.ts';

// Use the non-streaming overload only — streaming responses are passed through unchanged.
type MessageCreateParamsNonStreaming = Parameters<Anthropic['messages']['create']>[0] & { stream?: false };
type MessageCreateParams = Parameters<Anthropic['messages']['create']>[0];
type MessageResponse = Awaited<ReturnType<Anthropic['messages']['create']>>;

function formatMemories(memories: SearchResult[]): string {
  return memories
    .map(m => `**${m.title}** (${m.topic})\n${m.chunk.slice(0, 300)}`)
    .join('\n\n');
}

function extractUserQuery(params: MessageCreateParams): string {
  const messages = params.messages ?? [];
  const lastUser = [...messages].reverse().find(m => m.role === 'user');
  if (!lastUser) return '';
  if (typeof lastUser.content === 'string') return lastUser.content;
  if (Array.isArray(lastUser.content)) {
    return lastUser.content
      .filter((b): b is { type: 'text'; text: string } => b.type === 'text')
      .map(b => b.text)
      .join(' ');
  }
  return '';
}

function extractResponseText(response: MessageResponse): string {
  if (!('content' in response)) return '';
  return response.content
    .filter(b => b.type === 'text')
    .map(b => (b as { type: 'text'; text: string }).text)
    .join('\n');
}

export class EngramClient {
  private client: Anthropic;

  constructor(options?: ConstructorParameters<typeof Anthropic>[0]) {
    this.client = new Anthropic(options);
  }

  messages = {
    create: async (params: MessageCreateParams): Promise<MessageResponse> => {
      // Streaming responses can't be buffered for memory extraction — pass through unchanged.
      if (params.stream === true) {
        return this.client.messages.create(params) as Promise<MessageResponse>;
      }

      const query = extractUserQuery(params);

      // 1. Search memory and inject as system context
      let enrichedParams = { ...params } as MessageCreateParams;
      if (query.length > 20) {
        try {
          const memories = await search(query, 3);
          // Task 5: use INJECTION_THRESHOLD from memory.ts instead of hardcoded 0.5
          const relevant = memories.filter(m => m.distance < INJECTION_THRESHOLD);

          if (relevant.length > 0) {
            const memoryContext = `## Relevant context from Engram memory\n\n${formatMemories(relevant)}\n\n---\n\n`;
            const existingSystem = typeof params.system === 'string' ? params.system : '';
            enrichedParams = {
              ...params,
              system: memoryContext + existingSystem,
            };
          }
        } catch { /* never block on memory failure */ }
      }

      // 2. Call Claude
      const response = await this.client.messages.create(enrichedParams) as MessageResponse;

      // 3. Auto-remember in background — fire and forget
      const responseText = extractResponseText(response);
      // Task 8: pass projectScope so autoRemember is properly scoped
      autoRemember(responseText, undefined, undefined, getProjectScope()).catch(() => {});

      return response;
    },
  };
}

/** Factory function for simpler instantiation. */
export function createEngramClient(options?: ConstructorParameters<typeof Anthropic>[0]): EngramClient {
  return new EngramClient(options);
}
