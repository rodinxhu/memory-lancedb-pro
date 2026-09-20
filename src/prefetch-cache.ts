/**
 * TM3 — Async Prefetch Recall (total-memory project)
 *
 * Moves auto-recall off the critical path. At agent_end (the user is reading
 * the reply) the plugin speculatively runs full retrieval for the session's
 * active topic and caches the results per session. At before_agent_start the
 * incoming query's embedding is compared to the cached query's embedding —
 * similar enough means the cached results are served instantly; a topic
 * switch falls back to live retrieval with a tighter rerank timeout.
 *
 * Inspired by Hermes' MemoryManager.queue_prefetch().
 */

// ============================================================================
// Config
// ============================================================================

export interface PrefetchConfig {
  enabled?: boolean;
  /** Cosine similarity (query vs cached query) required to serve from cache. */
  similarityGate?: number;
  /** Rerank API timeout for the live-fallback path (background prefetch keeps the default 5s). */
  liveRerankTimeoutMs?: number;
  /** Cache entry lifetime in minutes. */
  ttlMinutes?: number;
  /** Max sessions cached at once (oldest evicted). */
  maxSessions?: number;
}

export const DEFAULT_PREFETCH_CONFIG: Required<PrefetchConfig> = {
  enabled: true,
  similarityGate: 0.85,
  liveRerankTimeoutMs: 1500,
  ttlMinutes: 30,
  maxSessions: 50,
};

export function parsePrefetchConfig(value: unknown): Required<PrefetchConfig> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return { ...DEFAULT_PREFETCH_CONFIG };
  }
  const cfg = value as Record<string, unknown>;
  return {
    enabled: cfg.enabled !== false,
    similarityGate:
      typeof cfg.similarityGate === "number" && cfg.similarityGate > 0 && cfg.similarityGate <= 1
        ? cfg.similarityGate
        : DEFAULT_PREFETCH_CONFIG.similarityGate,
    liveRerankTimeoutMs:
      typeof cfg.liveRerankTimeoutMs === "number" && cfg.liveRerankTimeoutMs >= 200
        ? Math.floor(cfg.liveRerankTimeoutMs)
        : DEFAULT_PREFETCH_CONFIG.liveRerankTimeoutMs,
    ttlMinutes:
      typeof cfg.ttlMinutes === "number" && cfg.ttlMinutes >= 1
        ? Math.floor(cfg.ttlMinutes)
        : DEFAULT_PREFETCH_CONFIG.ttlMinutes,
    maxSessions:
      typeof cfg.maxSessions === "number" && cfg.maxSessions >= 1
        ? Math.floor(cfg.maxSessions)
        : DEFAULT_PREFETCH_CONFIG.maxSessions,
  };
}

// ============================================================================
// Similarity
// ============================================================================

export function cosineSimilarity(a: number[], b: number[]): number {
  if (!a?.length || !b?.length || a.length !== b.length) return 0;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// ============================================================================
// Cache
// ============================================================================

interface CacheEntry<T> {
  query: string;
  queryVector: number[];
  results: T[];
  expiresAt: number;
}

export interface PrefetchHit<T> {
  results: T[];
  cachedQuery: string;
  similarity: number;
}

/**
 * Per-session prefetch cache. Generic over the result type so it stays
 * decoupled from the retriever module.
 */
export class PrefetchCache<T> {
  private entries = new Map<string, CacheEntry<T>>();

  constructor(
    private ttlMs: number,
    private maxSessions: number
  ) {}

  /** Store the latest speculative retrieval for a session (replaces previous). */
  set(sessionKey: string, query: string, queryVector: number[], results: T[]): void {
    // Map preserves insertion order — delete + set keeps eviction LRU-ish.
    this.entries.delete(sessionKey);
    this.entries.set(sessionKey, {
      query,
      queryVector,
      results,
      expiresAt: Date.now() + this.ttlMs,
    });
    while (this.entries.size > this.maxSessions) {
      const oldest = this.entries.keys().next().value;
      if (oldest === undefined) break;
      this.entries.delete(oldest);
    }
  }

  /** Serve cached results when the incoming query is close enough to the prefetched one. */
  match(sessionKey: string, queryVector: number[], similarityGate: number): PrefetchHit<T> | null {
    const entry = this.entries.get(sessionKey);
    if (!entry) return null;
    if (Date.now() > entry.expiresAt) {
      this.entries.delete(sessionKey);
      return null;
    }
    const similarity = cosineSimilarity(queryVector, entry.queryVector);
    if (similarity < similarityGate) return null;
    return { results: entry.results, cachedQuery: entry.query, similarity };
  }

  invalidate(sessionKey: string): void {
    this.entries.delete(sessionKey);
  }

  clear(): void {
    this.entries.clear();
  }

  get size(): number {
    return this.entries.size;
  }
}
