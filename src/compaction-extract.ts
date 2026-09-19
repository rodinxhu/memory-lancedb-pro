/**
 * TM2 — Pre-compaction Extraction (total-memory project)
 *
 * When OpenClaw compacts a session, the summarized-away history is gone for
 * good unless something harvested it first. This module runs in the public
 * `before_compaction` hook (event carries the full mirrored message list) and
 * sweeps that history for:
 *
 *   1. Turn pairs the live observation layer (TM1) may have missed — multi-
 *      message turns, turns from before TM1 deployed, crashed agent_end hooks.
 *      Stored to the observation scope; the same near-dup gate that protects
 *      live capture makes the sweep idempotent (already-observed turns skip).
 *
 *   2. Trigger-matched durable facts (same heuristics as live auto-capture),
 *      stored to the agent's default scope at normal importance.
 *
 * Inspired by Hermes' MemoryProvider.on_pre_compress().
 */

import type { MemoryStore } from "./store.js";
import type { Embedder } from "./embedder.js";
import {
  extractAllTurnPairs,
  storeObservationPair,
  type ObservationConfig,
} from "./observations.js";

// ============================================================================
// Types & Defaults
// ============================================================================

export interface CompactionExtractConfig {
  enabled?: boolean;
  /** Max turn pairs swept into the observation scope per compaction (oldest first). */
  maxPairs?: number;
  /** Max trigger-matched facts stored per compaction. */
  maxFacts?: number;
}

export const DEFAULT_COMPACTION_EXTRACT_CONFIG: Required<CompactionExtractConfig> = {
  enabled: true,
  maxPairs: 40,
  maxFacts: 10,
};

export function parseCompactionExtractConfig(value: unknown): Required<CompactionExtractConfig> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return { ...DEFAULT_COMPACTION_EXTRACT_CONFIG };
  }
  const cfg = value as Record<string, unknown>;
  return {
    enabled: cfg.enabled !== false,
    maxPairs:
      typeof cfg.maxPairs === "number" && cfg.maxPairs >= 1
        ? Math.floor(cfg.maxPairs)
        : DEFAULT_COMPACTION_EXTRACT_CONFIG.maxPairs,
    maxFacts:
      typeof cfg.maxFacts === "number" && cfg.maxFacts >= 0
        ? Math.floor(cfg.maxFacts)
        : DEFAULT_COMPACTION_EXTRACT_CONFIG.maxFacts,
  };
}

// ============================================================================
// Sweep
// ============================================================================

export interface SweepDeps {
  store: MemoryStore;
  embedder: Embedder;
  logger: { info: (msg: string) => void; warn: (msg: string) => void; debug?: (msg: string) => void };
  /** Live auto-capture quality gate (injected to avoid an import cycle with index.ts). */
  shouldCapture: (text: string) => boolean;
  detectCategory: (text: string) => "preference" | "fact" | "decision" | "entity" | "other";
  /** Importance for a captured turn; explicit triggers outrank incidental chatter. */
  captureImportance?: (text: string) => number;
  /** TM5: store-time threat neutralization (identity when disabled). */
  sanitize?: (text: string) => string;
}

export interface SweepResult {
  pairsScanned: number;
  pairsDropped: number; // over maxPairs cap
  observationsStored: number;
  factsStored: number;
}

/**
 * Harvest the history that is about to be compacted. Never throws — designed
 * to run inside a hook on the live gateway.
 */
export async function sweepCompactingHistory(
  messages: unknown[],
  agentId: string,
  defaultScope: string,
  obsConfig: Required<ObservationConfig>,
  config: Required<CompactionExtractConfig>,
  deps: SweepDeps
): Promise<SweepResult> {
  const result: SweepResult = { pairsScanned: 0, pairsDropped: 0, observationsStored: 0, factsStored: 0 };
  if (!config.enabled) return result;

  try {
    const allPairs = extractAllTurnPairs(messages);
    result.pairsScanned = allPairs.length;

    // Oldest pairs are the ones dying; the newest were just observed live by TM1.
    const pairs = allPairs.slice(0, config.maxPairs);
    result.pairsDropped = allPairs.length - pairs.length;

    // 1. Sweep turn pairs into the observation scope (dedup makes this idempotent).
    if (obsConfig.enabled) {
      for (const pair of pairs) {
        try {
          const outcome = await storeObservationPair(pair, agentId, obsConfig, deps, {
            source: "compaction-sweep",
          });
          if (outcome === "stored") result.observationsStored++;
        } catch (err) {
          deps.logger.warn(`compaction-extract: observation store failed: ${String(err)}`);
        }
      }
    }

    // 2. Harvest trigger-matched durable facts into the default scope.
    let factsStored = 0;
    for (const pair of pairs) {
      if (factsStored >= config.maxFacts) break;
      const text = deps.sanitize ? deps.sanitize(pair.user) : pair.user;
      if (!deps.shouldCapture(text)) continue;
      try {
        const vector = await deps.embedder.embedPassage(text);
        const existing = await deps.store.vectorSearch(vector, 1, 0.1, [defaultScope]);
        if (existing.length > 0 && existing[0].score > 0.95) continue; // same gate as live auto-capture
        await deps.store.store({
          text,
          vector,
          category: deps.detectCategory(text),
          scope: defaultScope,
          importance: deps.captureImportance ? deps.captureImportance(text) : 0.7,
          metadata: JSON.stringify({ source: "compaction-sweep", agentId }),
        });
        factsStored++;
      } catch (err) {
        deps.logger.warn(`compaction-extract: fact store failed: ${String(err)}`);
      }
    }
    result.factsStored = factsStored;

    if (result.pairsDropped > 0) {
      deps.logger.info(
        `compaction-extract: capped sweep at ${config.maxPairs} pairs (${result.pairsDropped} older pairs not swept)`
      );
    }
    return result;
  } catch (err) {
    deps.logger.warn(`compaction-extract: sweep failed: ${String(err)}`);
    return result;
  }
}
