/**
 * Cold Archive
 * ============
 *
 * Memories evicted from the hot store are moved here, never deleted. A memory
 * that has not been retrieved in months is still history worth keeping — it
 * just should not occupy the vector index, dilute retrieval, or cost embedding
 * work on every rebuild.
 *
 * Deliberately a plain append-only JSONL file rather than another database:
 * - survives any corruption of the LanceDB table (the September 2026 outage
 *   would not have touched it)
 * - greppable and readable without this plugin, or without any tooling at all
 * - trivially backed up and diffed
 *
 * Nothing in here is ever rewritten in place. Restoring a memory appends a
 * `restored` record rather than removing the original line, so the archive is
 * a complete history of what left the hot store and when.
 */

import { readFile, appendFile, mkdir, readdir } from "node:fs/promises";
import { join } from "node:path";
import type { MemoryEntry, MemoryStore } from "./store.js";

export type ArchiveReason =
  | "disuse"        // never retrieved within the disuse window
  | "consolidated"  // merged into a combined memory
  | "expired"       // observation past its retention window
  | "manual";       // operator moved it by hand

export interface ArchivedMemory {
  /** Original memory id, so a restore can be traced back. */
  id: string;
  text: string;
  category: string;
  scope: string;
  importance: number;
  /** Original creation time — NOT the archive time. */
  timestamp: number;
  metadata?: string;
  archivedAt: number;
  archivedReason: ArchiveReason;
  /** For "consolidated": the id of the memory that replaced this one. */
  supersededBy?: string;
  /** Bookkeeping when a row is pulled back into the hot store. */
  restoredAt?: number;
}

/** One file per month keeps any single file small enough to read by hand. */
function archiveFileFor(archiveDir: string, when: Date): string {
  const month = when.toISOString().slice(0, 7); // YYYY-MM
  return join(archiveDir, `memories-archive-${month}.jsonl`);
}

export function archiveDirFor(dbPath: string): string {
  return join(dbPath, "..", "archive");
}

/**
 * Append entries to the cold archive. Returns how many were written.
 *
 * Callers MUST append before deleting from the hot store — if the append
 * throws, the delete must not happen.
 */
export async function archiveMemories(
  archiveDir: string,
  entries: MemoryEntry[],
  reason: ArchiveReason,
  supersededBy?: string
): Promise<number> {
  if (entries.length === 0) return 0;
  await mkdir(archiveDir, { recursive: true });

  const now = Date.now();
  const lines = entries.map((m) => {
    const rec: ArchivedMemory = {
      id: m.id,
      text: m.text,
      category: m.category,
      scope: m.scope,
      importance: m.importance,
      timestamp: m.timestamp,
      metadata: m.metadata,
      archivedAt: now,
      archivedReason: reason,
    };
    if (supersededBy) rec.supersededBy = supersededBy;
    return JSON.stringify(rec);
  });

  await appendFile(archiveFileFor(archiveDir, new Date(now)), lines.join("\n") + "\n", "utf-8");
  return lines.length;
}

/** Read every archived record, oldest file first. */
export async function readArchive(archiveDir: string): Promise<ArchivedMemory[]> {
  let files: string[];
  try {
    files = (await readdir(archiveDir))
      .filter((f) => f.startsWith("memories-archive-") && f.endsWith(".jsonl"))
      .sort();
  } catch {
    return []; // no archive yet
  }

  const out: ArchivedMemory[] = [];
  for (const f of files) {
    const raw = await readFile(join(archiveDir, f), "utf-8");
    for (const line of raw.split("\n")) {
      if (!line.trim()) continue;
      try {
        out.push(JSON.parse(line) as ArchivedMemory);
      } catch {
        // A truncated final line (power loss mid-append) must not make the
        // whole archive unreadable.
      }
    }
  }
  return out;
}

/**
 * Plain substring search over archived text. No embeddings on purpose: the
 * archive must stay readable when the embedder or the vector store is down,
 * which is exactly when someone is most likely to go looking in it.
 */
export async function searchArchive(
  archiveDir: string,
  query: string,
  limit = 20
): Promise<ArchivedMemory[]> {
  const all = await readArchive(archiveDir);
  const needles = query.toLowerCase().split(/\s+/).filter(Boolean);
  if (needles.length === 0) return all.slice(-limit).reverse();

  const scored = all
    .map((rec) => {
      const hay = rec.text.toLowerCase();
      const hits = needles.filter((n) => hay.includes(n)).length;
      return { rec, hits };
    })
    .filter((s) => s.hits > 0)
    .sort((a, b) => b.hits - a.hits || b.rec.archivedAt - a.rec.archivedAt);

  return scored.slice(0, limit).map((s) => s.rec);
}

/**
 * Pull archived memories back into the hot store. Re-embeds, because the
 * archive stores text only — a vector from a previous embedding model would
 * be meaningless after a model switch (as happened on 2026-09-20).
 */
export async function restoreFromArchive(
  archiveDir: string,
  store: MemoryStore,
  embedPassage: (text: string) => Promise<number[]>,
  ids: string[]
): Promise<{ restored: number; notFound: string[] }> {
  const all = await readArchive(archiveDir);
  const wanted = new Set(ids);
  const found = all.filter((r) => wanted.has(r.id) || ids.some((id) => r.id.startsWith(id)));
  const notFound = ids.filter((id) => !found.some((r) => r.id === id || r.id.startsWith(id)));

  let restored = 0;
  for (const rec of found) {
    const vector = await embedPassage(rec.text);
    await store.store({
      text: rec.text,
      vector,
      category: rec.category as MemoryEntry["category"],
      scope: rec.scope,
      importance: rec.importance,
      metadata: JSON.stringify({
        ...safeParse(rec.metadata),
        restoredFromArchive: true,
        originalId: rec.id,
        restoredAt: Date.now(),
      }),
    });
    restored++;
  }

  // Record the restore as a new append; the original line stays put so the
  // archive remains a full history rather than a mutable store.
  if (found.length > 0) {
    const now = Date.now();
    const lines = found.map((r) => JSON.stringify({ ...r, restoredAt: now }));
    await appendFile(archiveFileFor(archiveDir, new Date(now)), lines.join("\n") + "\n", "utf-8");
  }

  return { restored, notFound };
}

function safeParse(s?: string): Record<string, unknown> {
  try {
    return s ? (JSON.parse(s) as Record<string, unknown>) : {};
  } catch {
    return {};
  }
}

export interface ArchiveStats {
  total: number;
  byReason: Record<string, number>;
  oldestArchivedAt: number | null;
  newestArchivedAt: number | null;
  restored: number;
}

export async function archiveStats(archiveDir: string): Promise<ArchiveStats> {
  const all = await readArchive(archiveDir);
  const byReason: Record<string, number> = {};
  let oldest: number | null = null;
  let newest: number | null = null;
  let restored = 0;

  for (const r of all) {
    byReason[r.archivedReason] = (byReason[r.archivedReason] ?? 0) + 1;
    if (oldest === null || r.archivedAt < oldest) oldest = r.archivedAt;
    if (newest === null || r.archivedAt > newest) newest = r.archivedAt;
    if (r.restoredAt) restored++;
  }

  return { total: all.length, byReason, oldestArchivedAt: oldest, newestArchivedAt: newest, restored };
}
