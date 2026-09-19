/**
 * TM1 — Observation Layer (total-memory project)
 *
 * Stores every turn pair as a low-importance "observation" entry in a
 * dedicated scope (`observations:agent:<id>`), so no conversational fact is
 * ever silently lost. Curated scopes stay clean: the observation scope is
 * never part of an agent's accessible scopes, so auto-recall and agent tools
 * never surface raw observations — only the nightly promotion pass and manual
 * `memx` queries read them.
 *
 * Inspired by Hermes' MemoryProvider.sync_turn() / Honcho observations.
 */

import type { MemoryStore } from "./store.js";
import type { Embedder } from "./embedder.js";

// ============================================================================
// Types & Defaults
// ============================================================================

export interface ObservationConfig {
  enabled?: boolean;
  /** Importance assigned to observation entries (kept low so promoted memories outrank raw turns). */
  importance?: number;
  /** Observations older than this are expired by the maintenance pass. */
  retentionDays?: number;
  /** Vector similarity above which a new observation is considered a duplicate and skipped. */
  dedupThreshold?: number;
  /** Hard cap on stored observation text length (chars). */
  maxTextLength?: number;
}

export const DEFAULT_OBSERVATION_CONFIG: Required<ObservationConfig> = {
  enabled: true,
  importance: 0.3,
  retentionDays: 45,
  dedupThreshold: 0.97,
  maxTextLength: 1600,
};

export function observationScope(agentId: string): string {
  return `observations:agent:${agentId}`;
}

export function isObservationScope(scope: string): boolean {
  return scope.startsWith("observations:");
}

// ============================================================================
// Turn-pair extraction
// ============================================================================

/** Patterns for turns that carry no durable information worth observing. */
const SKIP_USER_PATTERNS = [
  /^\/[a-z]/i, // slash commands
  /^(ok|okay|yes|no|thanks|thank you|thx|sure|cool|nice|good|great|haha|lol|hi|hello|hey)[.!?]*$/i,
  /^(好的?|嗯|是的?|谢谢|可以|行|哈哈|你好)[。!?]*$/,
  /HEARTBEAT/,
  /^\[cron:/i, // scheduled-job prompts are system noise, not user conversation
  /^\[(system|queued)/i,
  /^\[OpenClaw heartbeat poll\]/i, // gateway liveness ping, not a user turn
];

function extractText(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((block) => {
        if (
          block &&
          typeof block === "object" &&
          (block as Record<string, unknown>).type === "text" &&
          typeof (block as Record<string, unknown>).text === "string"
        ) {
          return (block as Record<string, unknown>).text as string;
        }
        return "";
      })
      .filter(Boolean)
      .join("\n");
  }
  return "";
}

/**
 * Strip injected memory context, channel plumbing, and collapse whitespace.
 *
 * Channel adapters prepend envelope blocks ("Conversation info (untrusted
 * metadata): ```json …```", "Sender (untrusted metadata): …", delivery hints)
 * to every inbound message. Left in, they consume the whole user budget in
 * formatObservation() and the real question gets truncated away — every
 * observation ends up looking like a chat_id dump.
 */
function cleanText(text: string): string {
  return text
    .replace(/<relevant-memories>[\s\S]*?<\/relevant-memories>/g, "")
    .replace(/^[^\n:]{0,40}\(untrusted metadata\):\s*```(?:json)?[\s\S]*?```/gim, "")
    .replace(/Delivery:[^\n]*?use the `[a-z_]+` tool\.?/gi, "")
    .replace(/\s+/g, " ")
    .trim();
}

export interface TurnPair {
  user: string;
  assistant: string;
}

/** Apply the per-pair quality gate shared by live capture and the TM2 compaction sweep. */
function isObservablePair(user: string, assistant: string): boolean {
  if (!user || !assistant) return false;
  const hasCJK = /[一-鿿぀-ゟ゠-ヿ가-힯]/.test(user);
  if (user.length < (hasCJK ? 2 : 5)) return false;
  if (SKIP_USER_PATTERNS.some((r) => r.test(user))) return false;
  return true;
}

/**
 * Pull the final user/assistant exchange out of an agent_end messages array.
 * Returns null when the turn is not worth observing (commands, ack-only,
 * heartbeats, empty).
 */
export function extractTurnPair(messages: unknown[]): TurnPair | null {
  let user = "";
  let assistant = "";

  // Walk backwards: last assistant text, then the user text preceding it.
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (!msg || typeof msg !== "object") continue;
    const { role, content } = msg as Record<string, unknown>;
    const text = cleanText(extractText(content));
    if (!text) continue;
    if (!assistant && role === "assistant") {
      assistant = text;
    } else if (assistant && role === "user") {
      user = text;
      break;
    }
  }

  if (!isObservablePair(user, assistant)) return null;
  return { user, assistant };
}

/**
 * Walk the FULL history and pair every user message with the assistant reply
 * that follows it (TM2 pre-compaction sweep). Oldest first — those are the
 * messages about to be summarized away.
 */
export function extractAllTurnPairs(messages: unknown[]): TurnPair[] {
  const pairs: TurnPair[] = [];
  let pendingUser = "";
  for (const msg of messages) {
    if (!msg || typeof msg !== "object") continue;
    const { role, content } = msg as Record<string, unknown>;
    const text = cleanText(extractText(content));
    if (!text) continue;
    if (role === "user") {
      pendingUser = text; // a newer user message supersedes an unanswered one
    } else if (role === "assistant" && pendingUser) {
      if (isObservablePair(pendingUser, text)) {
        pairs.push({ user: pendingUser, assistant: text });
      }
      pendingUser = "";
    }
  }
  return pairs;
}

/** Format a turn pair as the stored observation document. */
export function formatObservation(pair: TurnPair, maxLength: number, dateStr: string): string {
  // Budget: user gets up to 45% of the cap, assistant the rest.
  const userBudget = Math.floor(maxLength * 0.45);
  const user = pair.user.length > userBudget ? pair.user.slice(0, userBudget) + "…" : pair.user;
  const assistantBudget = maxLength - user.length - 32; // header + labels overhead
  const assistant =
    pair.assistant.length > assistantBudget
      ? pair.assistant.slice(0, assistantBudget) + "…"
      : pair.assistant;
  return `[${dateStr}] user: ${user}\nassistant: ${assistant}`;
}

// ============================================================================
// Capture
// ============================================================================

export interface ObservationDeps {
  store: MemoryStore;
  embedder: Embedder;
  logger: { info: (msg: string) => void; warn: (msg: string) => void; debug?: (msg: string) => void };
  /** TM5: store-time threat neutralization (identity function when disabled). */
  sanitize?: (text: string) => string;
}

export interface CaptureResult {
  stored: boolean;
  reason?: "disabled" | "no-turn-pair" | "duplicate" | "error";
}

/**
 * Embed + dedup + store one turn pair into the observation scope.
 * Shared by live capture (agent_end) and the TM2 pre-compaction sweep.
 */
export async function storeObservationPair(
  pair: TurnPair,
  agentId: string,
  config: Required<ObservationConfig>,
  deps: ObservationDeps,
  extraMetadata?: Record<string, unknown>
): Promise<"stored" | "duplicate"> {
  const scope = observationScope(agentId);
  const dateStr = new Date().toISOString().split("T")[0];
  let text = formatObservation(pair, config.maxTextLength, dateStr);
  if (deps.sanitize) text = deps.sanitize(text); // TM5: neutralize injection patterns at the door

  const vector = await deps.embedder.embedPassage(text);

  // Near-dup rejection within the observation scope only.
  const existing = await deps.store.vectorSearch(vector, 1, 0.1, [scope]);
  if (existing.length > 0 && existing[0].score > config.dedupThreshold) {
    return "duplicate";
  }

  await deps.store.store({
    text,
    vector,
    category: "other",
    scope,
    importance: config.importance,
    metadata: JSON.stringify({ type: "observation", agentId, date: dateStr, ...extraMetadata }),
  });
  return "stored";
}

/**
 * Capture one turn pair as an observation. Designed to run inside the
 * agent_end hook — must never throw (errors are logged and swallowed).
 */
export async function captureObservation(
  messages: unknown[],
  agentId: string,
  config: Required<ObservationConfig>,
  deps: ObservationDeps
): Promise<CaptureResult> {
  if (!config.enabled) return { stored: false, reason: "disabled" };

  try {
    const pair = extractTurnPair(messages);
    if (!pair) return { stored: false, reason: "no-turn-pair" };

    const outcome = await storeObservationPair(pair, agentId, config, deps);
    if (outcome === "duplicate") return { stored: false, reason: "duplicate" };

    deps.logger.debug?.(`observations: captured turn pair for agent ${agentId}`);
    return { stored: true };
  } catch (err) {
    deps.logger.warn(`observations: capture failed: ${String(err)}`);
    return { stored: false, reason: "error" };
  }
}

// ============================================================================
// Maintenance: expiry + promotion candidates
// ============================================================================

/** Delete observations older than retentionDays for the given agent. Returns count deleted. */
export async function expireObservations(
  store: MemoryStore,
  agentId: string,
  retentionDays: number
): Promise<number> {
  const cutoff = Date.now() - retentionDays * 24 * 60 * 60 * 1000;
  return store.bulkDelete([observationScope(agentId)], cutoff);
}

export interface PromotionCandidate {
  /** Representative (most recent) observation text. */
  text: string;
  /** IDs of all observations in the recurring cluster. */
  memberIds: string[];
  /** Cluster size — how many distinct turns touched this topic. */
  count: number;
}

/**
 * Find recurring topics among recent observations: greedy similarity
 * clustering over stored vectors. Clusters with >= minClusterSize members are
 * promotion candidates (to be distilled into real memories by the nightly
 * dreaming cron — the cron's agent does the wording, this just finds clusters).
 */
export async function findPromotionCandidates(
  store: MemoryStore,
  agentId: string,
  options: { lookbackDays?: number; minClusterSize?: number; similarityThreshold?: number; maxCandidates?: number } = {}
): Promise<PromotionCandidate[]> {
  const lookbackDays = options.lookbackDays ?? 14;
  const minClusterSize = options.minClusterSize ?? 3;
  // Normalized BM25 neighbor score required to join a cluster (0..1 sigmoid scale).
  const similarityThreshold = options.similarityThreshold ?? 0.5;
  const maxCandidates = options.maxCandidates ?? 5;

  const scope = observationScope(agentId);
  const since = Date.now() - lookbackDays * 24 * 60 * 60 * 1000;

  // list() omits vectors for performance, so re-query each candidate via
  // vectorSearch seeded by its own embedding-neighborhood: instead we cluster
  // by querying the store per observation. To keep this cheap we cap input.
  const recent = (await store.list([scope], undefined, 500, 0)).filter(
    (m) => m.timestamp >= since && JSON.parse(m.metadata || "{}").promoted !== true
  );
  if (recent.length < minClusterSize) return [];

  const assigned = new Set<string>();
  const candidates: PromotionCandidate[] = [];

  for (const seed of recent) {
    if (assigned.has(seed.id)) continue;
    // Use BM25-free pure vector neighborhood around this observation's text.
    let neighbors;
    try {
      const seedFull = await store.bm25Search(seed.text.slice(0, 200), 25, [scope]);
      neighbors = seedFull.filter((n) => n.score >= similarityThreshold && !assigned.has(n.entry.id));
    } catch {
      continue;
    }
    const memberIds = new Set<string>([seed.id]);
    for (const n of neighbors) {
      if (n.entry.id !== seed.id && n.entry.timestamp >= since) memberIds.add(n.entry.id);
    }
    if (memberIds.size >= minClusterSize) {
      for (const id of memberIds) assigned.add(id);
      candidates.push({ text: seed.text, memberIds: [...memberIds], count: memberIds.size });
      if (candidates.length >= maxCandidates) break;
    }
  }

  return candidates.sort((a, b) => b.count - a.count);
}

// ============================================================================
// Config parsing
// ============================================================================

export function parseObservationConfig(value: unknown): Required<ObservationConfig> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return { ...DEFAULT_OBSERVATION_CONFIG };
  }
  const cfg = value as Record<string, unknown>;
  return {
    enabled: cfg.enabled !== false,
    importance:
      typeof cfg.importance === "number" && cfg.importance >= 0 && cfg.importance <= 1
        ? cfg.importance
        : DEFAULT_OBSERVATION_CONFIG.importance,
    retentionDays:
      typeof cfg.retentionDays === "number" && cfg.retentionDays >= 1
        ? Math.floor(cfg.retentionDays)
        : DEFAULT_OBSERVATION_CONFIG.retentionDays,
    dedupThreshold:
      typeof cfg.dedupThreshold === "number" && cfg.dedupThreshold > 0 && cfg.dedupThreshold <= 1
        ? cfg.dedupThreshold
        : DEFAULT_OBSERVATION_CONFIG.dedupThreshold,
    maxTextLength:
      typeof cfg.maxTextLength === "number" && cfg.maxTextLength >= 200
        ? Math.floor(cfg.maxTextLength)
        : DEFAULT_OBSERVATION_CONFIG.maxTextLength,
  };
}
