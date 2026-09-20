/**
 * TM4 — Session Search (total-memory project)
 *
 * Verbatim full-text search over OpenClaw session transcripts. The semantic
 * stack (LanceDB) answers "things about topic X"; this answers "what EXACTLY
 * was said, and when". Inspired by Hermes' SQLite FTS5 message index (with
 * the same trigram trick for CJK substring matching).
 *
 * - Sidecar DB: SQLite via node:sqlite (no native npm dependency to survive
 *   upgrades), default at ~/.openclaw/memory/session-fts.db
 * - Incremental: tracks per-file byte offsets; only new bytes are parsed.
 * - Indexes user/assistant message text from agents/<id>/sessions/*.jsonl
 *   (trajectory traces excluded; rotated .reset. files included).
 */

import { DatabaseSync } from "node:sqlite";
import { open, readdir, stat, mkdir } from "node:fs/promises";
import { join, dirname, basename } from "node:path";

// ============================================================================
// Config
// ============================================================================

export interface SessionSearchConfig {
  enabled?: boolean;
  /** Sidecar DB path (default: <stateDir>/memory/session-fts.db, resolved by caller). */
  dbPath?: string;
  /** Default result limit for the tool. */
  maxResults?: number;
  /** Background incremental index interval (minutes). */
  indexIntervalMinutes?: number;
  /** Max chars of one message stored/indexed. */
  maxMessageLength?: number;
}

export const DEFAULT_SESSION_SEARCH_CONFIG: Required<SessionSearchConfig> = {
  enabled: true,
  dbPath: "", // resolved by caller against the OpenClaw state dir
  maxResults: 8,
  indexIntervalMinutes: 10,
  maxMessageLength: 4000,
};

export function parseSessionSearchConfig(value: unknown): Required<SessionSearchConfig> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return { ...DEFAULT_SESSION_SEARCH_CONFIG };
  }
  const cfg = value as Record<string, unknown>;
  return {
    enabled: cfg.enabled !== false,
    dbPath: typeof cfg.dbPath === "string" ? cfg.dbPath : "",
    maxResults:
      typeof cfg.maxResults === "number" && cfg.maxResults >= 1 && cfg.maxResults <= 50
        ? Math.floor(cfg.maxResults)
        : DEFAULT_SESSION_SEARCH_CONFIG.maxResults,
    indexIntervalMinutes:
      typeof cfg.indexIntervalMinutes === "number" && cfg.indexIntervalMinutes >= 1
        ? Math.floor(cfg.indexIntervalMinutes)
        : DEFAULT_SESSION_SEARCH_CONFIG.indexIntervalMinutes,
    maxMessageLength:
      typeof cfg.maxMessageLength === "number" && cfg.maxMessageLength >= 500
        ? Math.floor(cfg.maxMessageLength)
        : DEFAULT_SESSION_SEARCH_CONFIG.maxMessageLength,
  };
}

// ============================================================================
// Text extraction (session JSONL line → indexable text)
// ============================================================================

function extractMessageText(content: unknown, maxLength: number): string {
  let text = "";
  if (typeof content === "string") {
    text = content;
  } else if (Array.isArray(content)) {
    text = content
      .map((b) =>
        b && typeof b === "object" && (b as Record<string, unknown>).type === "text"
          ? String((b as Record<string, unknown>).text ?? "")
          : ""
      )
      .filter(Boolean)
      .join("\n");
  }
  return text
    .replace(/<relevant-memories>[\s\S]*?<\/relevant-memories>/g, "")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, maxLength);
}

/** Transcripts mix ISO strings and epoch-ms numbers; normalize to ISO for display/sorting. */
function normalizeTimestamp(value: unknown): string {
  if (typeof value === "number" && Number.isFinite(value)) {
    return new Date(value).toISOString();
  }
  if (typeof value === "string") {
    if (/^\d{12,}(\.\d+)?$/.test(value.trim())) {
      return new Date(parseFloat(value)).toISOString();
    }
    return value;
  }
  return "";
}

export interface SessionSearchResult {
  snippet: string;
  role: string;
  ts: string;
  sessionId: string;
  agentId: string;
  file: string;
}

export interface SessionFtsLogger {
  info: (msg: string) => void;
  warn: (msg: string) => void;
  debug?: (msg: string) => void;
}

// ============================================================================
// Index
// ============================================================================

const HAS_CJK = /[一-鿿぀-ゟ゠-ヿ가-힯]/;

export class SessionFtsIndex {
  private db: DatabaseSync | null = null;

  constructor(
    private dbPath: string,
    private logger: SessionFtsLogger,
    private maxMessageLength = 4000
  ) {}

  private async ensureOpen(): Promise<DatabaseSync> {
    if (this.db) return this.db;
    await mkdir(dirname(this.dbPath), { recursive: true });
    const db = new DatabaseSync(this.dbPath);
    db.exec(`
      PRAGMA journal_mode = WAL;
      CREATE TABLE IF NOT EXISTS files (
        path TEXT PRIMARY KEY,
        bytes_indexed INTEGER NOT NULL DEFAULT 0
      );
      CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file TEXT NOT NULL,
        agent_id TEXT NOT NULL,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        ts TEXT NOT NULL,
        text TEXT NOT NULL
      );
      CREATE INDEX IF NOT EXISTS idx_messages_file ON messages(file);
      CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
        text, content='messages', content_rowid='id'
      );
      CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts_tri USING fts5(
        text, content='messages', content_rowid='id', tokenize='trigram'
      );
      CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
        INSERT INTO messages_fts(rowid, text) VALUES (new.id, new.text);
        INSERT INTO messages_fts_tri(rowid, text) VALUES (new.id, new.text);
      END;
      CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
        INSERT INTO messages_fts(messages_fts, rowid, text) VALUES ('delete', old.id, old.text);
        INSERT INTO messages_fts_tri(messages_fts_tri, rowid, text) VALUES ('delete', old.id, old.text);
      END;
    `);
    this.db = db;
    return db;
  }

  /** Index new bytes of one session file. Returns messages added. */
  async indexFile(path: string, agentId: string): Promise<number> {
    const db = await this.ensureOpen();
    let fileSize: number;
    try {
      fileSize = (await stat(path)).size;
    } catch {
      return 0; // file vanished
    }

    const row = db.prepare("SELECT bytes_indexed FROM files WHERE path = ?").get(path) as
      | { bytes_indexed: number }
      | undefined;
    const offset = row?.bytes_indexed ?? 0;
    if (fileSize <= offset) return 0;

    // Read only the new bytes.
    const fh = await open(path, "r");
    let chunk: string;
    try {
      const buf = Buffer.alloc(fileSize - offset);
      await fh.read(buf, 0, buf.length, offset);
      chunk = buf.toString("utf-8");
    } finally {
      await fh.close();
    }

    // Only consume up to the last complete line — the writer may be mid-append.
    const lastNewline = chunk.lastIndexOf("\n");
    if (lastNewline === -1) return 0;
    const consumedBytes = offset + Buffer.byteLength(chunk.slice(0, lastNewline + 1), "utf-8");
    const lines = chunk.slice(0, lastNewline).split("\n");

    const sessionId = basename(path).replace(/\.jsonl.*$/, "").replace(/\.reset.*$/, "");
    const insert = db.prepare(
      "INSERT INTO messages (file, agent_id, session_id, role, ts, text) VALUES (?, ?, ?, ?, ?, ?)"
    );
    const upsertFile = db.prepare(
      "INSERT INTO files (path, bytes_indexed) VALUES (?, ?) ON CONFLICT(path) DO UPDATE SET bytes_indexed = excluded.bytes_indexed"
    );

    let added = 0;
    db.exec("BEGIN");
    try {
      for (const line of lines) {
        if (!line.trim()) continue;
        let entry: Record<string, unknown>;
        try {
          entry = JSON.parse(line);
        } catch {
          continue;
        }
        if (entry.type !== "message") continue;
        const msg = entry.message as Record<string, unknown> | undefined;
        if (!msg) continue;
        const role = msg.role as string;
        if (role !== "user" && role !== "assistant") continue;
        const text = extractMessageText(msg.content, this.maxMessageLength);
        if (!text || text.length < 3) continue;
        const ts = normalizeTimestamp(msg.timestamp ?? entry.timestamp);
        insert.run(path, agentId, sessionId, role, ts, text);
        added++;
      }
      upsertFile.run(path, consumedBytes);
      db.exec("COMMIT");
    } catch (err) {
      db.exec("ROLLBACK");
      throw err;
    }
    return added;
  }

  /**
   * Incrementally index all session files under the given agents dir
   * (<agentsDir>/<agentId>/sessions/*.jsonl). Yields between files so a large
   * initial pass doesn't starve the gateway event loop.
   */
  async indexAll(agentsDir: string): Promise<{ files: number; messages: number }> {
    let files = 0;
    let messages = 0;
    let agentIds: string[];
    try {
      agentIds = await readdir(agentsDir);
    } catch {
      return { files, messages };
    }
    for (const agentId of agentIds) {
      const sessionsDir = join(agentsDir, agentId, "sessions");
      let names: string[];
      try {
        names = await readdir(sessionsDir);
      } catch {
        continue;
      }
      for (const name of names) {
        if (!name.endsWith(".jsonl") || name.includes(".trajectory.")) continue;
        try {
          const added = await this.indexFile(join(sessionsDir, name), agentId);
          if (added > 0) {
            files++;
            messages += added;
          }
        } catch (err) {
          this.logger.warn(`session-fts: failed to index ${name}: ${String(err)}`);
        }
        // Yield to the event loop between files.
        await new Promise((resolve) => setImmediate(resolve));
      }
    }
    return { files, messages };
  }

  /** Escape a user query into a safe FTS5 MATCH expression (quoted terms, AND). */
  private static toMatchExpr(query: string): string {
    const terms = query
      .split(/\s+/)
      .map((t) => t.replace(/"/g, "").trim())
      .filter(Boolean);
    if (terms.length === 0) return '""';
    return terms.map((t) => `"${t}"`).join(" ");
  }

  /** Verbatim search: word FTS first, trigram for CJK/substrings, LIKE for short CJK. */
  async search(
    query: string,
    options: { limit?: number; agentId?: string } = {}
  ): Promise<SessionSearchResult[]> {
    const db = await this.ensureOpen();
    const limit = Math.min(Math.max(options.limit ?? 8, 1), 50);
    const agentFilter = options.agentId ? "AND m.agent_id = ?" : "";
    const baseParams = options.agentId ? [options.agentId] : [];

    const seen = new Set<number>();
    const results: SessionSearchResult[] = [];

    const collect = (rows: Array<Record<string, unknown>>) => {
      for (const r of rows) {
        const id = r.id as number;
        if (seen.has(id)) continue;
        seen.add(id);
        results.push({
          snippet: r.snippet as string,
          role: r.role as string,
          ts: r.ts as string,
          sessionId: r.session_id as string,
          agentId: r.agent_id as string,
          file: r.file as string,
        });
      }
    };

    // 1. Word-tokenized FTS (best ranking via bm25).
    try {
      const rows = db
        .prepare(
          `SELECT m.id, m.role, m.ts, m.session_id, m.agent_id, m.file,
                  snippet(messages_fts, 0, '«', '»', '…', 18) AS snippet
           FROM messages_fts JOIN messages m ON m.id = messages_fts.rowid
           WHERE messages_fts MATCH ? ${agentFilter}
           ORDER BY bm25(messages_fts) LIMIT ?`
        )
        .all(SessionFtsIndex.toMatchExpr(query), ...baseParams, limit) as Array<Record<string, unknown>>;
      collect(rows);
    } catch (err) {
      this.logger.debug?.(`session-fts: word search failed: ${String(err)}`);
    }

    // 2. Trigram FTS for CJK / substring matching (query must be >= 3 chars).
    if (results.length < limit && [...query.trim()].length >= 3) {
      try {
        const rows = db
          .prepare(
            `SELECT m.id, m.role, m.ts, m.session_id, m.agent_id, m.file,
                    snippet(messages_fts_tri, 0, '«', '»', '…', 18) AS snippet
             FROM messages_fts_tri JOIN messages m ON m.id = messages_fts_tri.rowid
             WHERE messages_fts_tri MATCH ? ${agentFilter}
             ORDER BY bm25(messages_fts_tri) LIMIT ?`
          )
          .all(SessionFtsIndex.toMatchExpr(query), ...baseParams, limit - results.length) as Array<
          Record<string, unknown>
        >;
        collect(rows);
      } catch (err) {
        this.logger.debug?.(`session-fts: trigram search failed: ${String(err)}`);
      }
    }

    // 3. LIKE fallback for very short CJK queries (trigram needs >= 3 chars).
    if (results.length === 0 && HAS_CJK.test(query) && [...query.trim()].length < 3) {
      try {
        const rows = db
          .prepare(
            `SELECT m.id, m.role, m.ts, m.session_id, m.agent_id, m.file,
                    substr(m.text, 1, 200) AS snippet
             FROM messages m
             WHERE m.text LIKE ? ${agentFilter}
             ORDER BY m.ts DESC LIMIT ?`
          )
          .all(`%${query.trim()}%`, ...baseParams, limit) as Array<Record<string, unknown>>;
        collect(rows);
      } catch (err) {
        this.logger.debug?.(`session-fts: LIKE fallback failed: ${String(err)}`);
      }
    }

    // Most recent first reads better for "when did we discuss X".
    return results.sort((a, b) => (b.ts || "").localeCompare(a.ts || "")).slice(0, limit);
  }

  async stats(): Promise<{ files: number; messages: number; dbPath: string }> {
    const db = await this.ensureOpen();
    const f = db.prepare("SELECT COUNT(*) AS c FROM files").get() as { c: number };
    const m = db.prepare("SELECT COUNT(*) AS c FROM messages").get() as { c: number };
    return { files: f.c, messages: m.c, dbPath: this.dbPath };
  }

  /** Drop and rebuild everything (recovery path). */
  async reindex(agentsDir: string): Promise<{ files: number; messages: number }> {
    const db = await this.ensureOpen();
    db.exec("DELETE FROM messages; DELETE FROM files;");
    return this.indexAll(agentsDir);
  }

  close(): void {
    try {
      this.db?.close();
    } catch {
      /* already closed */
    }
    this.db = null;
  }
}
