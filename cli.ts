/**
 * CLI Commands for Memory Management
 */

import type { Command } from "commander";
import { loadLanceDB, type MemoryEntry, type MemoryStore } from "./src/store.js";
import { findPromotionCandidates, findSingletonCandidates, observationScope } from "./src/observations.js";
import { archiveDirFor, archiveStats, restoreFromArchive, searchArchive } from "./src/archive.js";
import type { MemoryRetriever } from "./src/retriever.js";
import type { MemoryScopeManager } from "./src/scopes.js";
import type { MemoryMigrator } from "./src/migrate.js";

// ============================================================================
// Types
// ============================================================================

interface CLIContext {
  store: MemoryStore;
  retriever: MemoryRetriever;
  scopeManager: MemoryScopeManager;
  migrator: MemoryMigrator;
  embedder?: import("./src/embedder.js").Embedder;
  /** TM4: verbatim session transcript search (absent when sessionSearch.enabled=false). */
  sessionFts?: import("./src/session-fts.js").SessionFtsIndex;
  /** TM4: agents dir for on-demand incremental indexing before CLI searches. */
  agentsDir?: string;
}

// ============================================================================
// Utility Functions
// ============================================================================

function formatMemory(memory: any, index?: number): string {
  const prefix = index !== undefined ? `${index + 1}. ` : "";
  const date = new Date(memory.timestamp || memory.createdAt || Date.now()).toISOString().split('T')[0];
  const text = memory.text.slice(0, 100) + (memory.text.length > 100 ? "..." : "");
  return `${prefix}[${memory.category}:${memory.scope}] ${text} (${date})`;
}

function formatJson(obj: any): string {
  return JSON.stringify(obj, null, 2);
}

// ============================================================================
// CLI Command Implementations
// ============================================================================

export function registerMemoryCLI(program: Command, context: CLIContext): void {
  const memory = program
    .command("memx")
    .description("Enhanced memory management commands (memory-lancedb-pro)");

  // List memories
  memory
    .command("list")
    .description("List memories with optional filtering")
    .option("--scope <scope>", "Filter by scope")
    .option("--category <category>", "Filter by category")
    .option("--limit <n>", "Maximum number of results", "20")
    .option("--offset <n>", "Number of results to skip", "0")
    .option("--json", "Output as JSON")
    .action(async (options) => {
      try {
        const limit = parseInt(options.limit) || 20;
        const offset = parseInt(options.offset) || 0;

        let scopeFilter: string[] | undefined;
        if (options.scope) {
          scopeFilter = [options.scope];
        }

        const memories = await context.store.list(
          scopeFilter,
          options.category,
          limit,
          offset
        );

        if (options.json) {
          console.log(formatJson(memories));
        } else {
          if (memories.length === 0) {
            console.log("No memories found.");
          } else {
            console.log(`Found ${memories.length} memories:\n`);
            memories.forEach((memory, i) => {
              console.log(formatMemory(memory, offset + i));
            });
          }
        }
      } catch (error) {
        console.error("Failed to list memories:", error);
        process.exit(1);
      }
    });

  // Search memories
  memory
    .command("search <query>")
    .description("Search memories using hybrid retrieval")
    .option("--scope <scope>", "Search within specific scope")
    .option("--category <category>", "Filter by category")
    .option("--limit <n>", "Maximum number of results", "10")
    .option("--json", "Output as JSON")
    .action(async (query, options) => {
      try {
        const limit = parseInt(options.limit) || 10;

        let scopeFilter: string[] | undefined;
        if (options.scope) {
          scopeFilter = [options.scope];
        }

        const results = await context.retriever.retrieve({
          query,
          limit,
          scopeFilter,
          category: options.category,
        });

        if (options.json) {
          console.log(formatJson(results));
        } else {
          if (results.length === 0) {
            console.log("No relevant memories found.");
          } else {
            console.log(`Found ${results.length} memories:\n`);
            results.forEach((result, i) => {
              const sources = [];
              if (result.sources.vector) sources.push("vector");
              if (result.sources.bm25) sources.push("BM25");
              if (result.sources.reranked) sources.push("reranked");

              console.log(
                `${i + 1}. [${result.entry.category}:${result.entry.scope}] ${result.entry.text} ` +
                `(${(result.score * 100).toFixed(0)}%, ${sources.join('+')})`
              );
            });
          }
        }
      } catch (error) {
        console.error("Search failed:", error);
        process.exit(1);
      }
    });

  // Dreaming pass: cluster recurring observations for promotion to curated memory
  memory
    .command("dream")
    .description("Find recurring observation clusters ready to promote into curated memories")
    .option("--agent <id>", "Agent whose observations to cluster", "main")
    .option("--days <n>", "Lookback window in days", "14")
    .option("--min-cluster <n>", "Minimum related turns to qualify as recurring", "3")
    .option("--max <n>", "Maximum candidates to return", "5")
    .option("--mark <ids>", "Comma-separated observation IDs to flag as promoted (run after storing the distilled memory)")
    .option("--singleton-days <n>", "Lookback for individual (non-recurring) turns", "2")
    .option("--max-singletons <n>", "Maximum individual turns to surface for judging", "15")
    .option("--no-singletons", "Only return recurring clusters")
    .option("--json", "Output as JSON")
    .action(async (options) => {
      try {
        const agentId = options.agent || "main";
        const scope = observationScope(agentId);

        if (options.mark) {
          const ids = String(options.mark).split(",").map((s: string) => s.trim()).filter(Boolean);
          // Merge promoted:true into each entry's metadata (update replaces the string).
          const all = await context.store.list([scope], undefined, 2000, 0);
          const byId = new Map(all.map((m) => [m.id, m]));
          let marked = 0;
          for (const id of ids) {
            const entry = byId.get(id) ?? all.find((m) => m.id.startsWith(id));
            if (!entry) {
              console.error(`not found in ${scope}: ${id}`);
              continue;
            }
            let meta: Record<string, unknown> = {};
            try { meta = JSON.parse(entry.metadata || "{}"); } catch { /* keep empty */ }
            meta.promoted = true;
            await context.store.update(entry.id, { metadata: JSON.stringify(meta) }, [scope]);
            marked++;
          }
          console.log(options.json ? JSON.stringify({ marked, requested: ids.length }) : `Marked ${marked}/${ids.length} observations as promoted.`);
          return;
        }

        const clusters = await findPromotionCandidates(context.store, agentId, {
          lookbackDays: parseInt(options.days) || 14,
          minClusterSize: parseInt(options.minCluster) || 3,
          maxCandidates: parseInt(options.max) || 5,
        });

        // D2: clustering only finds RECURRING topics, but most durable facts
        // are stated once. Surface recent un-promoted turns too and let the
        // nightly agent judge which are worth keeping.
        const clustered = new Set(clusters.flatMap((c) => c.memberIds));
        const singletons = options.noSingletons
          ? []
          : await findSingletonCandidates(context.store, agentId, {
              lookbackDays: parseInt(options.singletonDays) || 2,
              maxCandidates: parseInt(options.maxSingletons) || 15,
              excludeIds: clustered,
            });

        if (options.json) {
          console.log(formatJson({
            clusters: clusters.map((c) => ({ ...c, kind: "cluster" })),
            singletons: singletons.map((c) => ({ ...c, kind: "singleton" })),
          }));
        } else if (clusters.length === 0 && singletons.length === 0) {
          console.log("No recurring observation clusters found (nothing to dream about).");
        } else {
          if (clusters.length > 0) {
            console.log(`Found ${clusters.length} recurring cluster(s):\n`);
            clusters.forEach((c, i) => {
              console.log(`${i + 1}. ${c.count} related turns — representative:`);
              console.log(`   ${c.text.slice(0, 200)}${c.text.length > 200 ? "…" : ""}`);
              console.log(`   member IDs: ${c.memberIds.join(",")}\n`);
            });
          }
          if (singletons.length > 0) {
            console.log(`Found ${singletons.length} recent turn(s) to judge individually:\n`);
            singletons.forEach((c, i) => {
              console.log(`${i + 1}. ${c.text.slice(0, 200)}${c.text.length > 200 ? "…" : ""}`);
              console.log(`   id: ${c.memberIds[0]}\n`);
            });
          }
        }
      } catch (error) {
        console.error("Dream pass failed:", error);
        process.exit(1);
      }
    });

  // Cold archive: evicted memories are moved here, never deleted.
  const archive = memory
    .command("archive")
    .description("Inspect and restore memories moved to the cold archive");

  archive
    .command("stats")
    .description("Summarise what has been archived")
    .option("--json", "Output as JSON")
    .action(async (options) => {
      try {
        const dir = archiveDirFor(context.store.dbPath);
        const s = await archiveStats(dir);
        if (options.json) {
          console.log(formatJson({ ...s, archiveDir: dir }));
          return;
        }
        if (s.total === 0) {
          console.log(`Archive is empty (${dir}).`);
          return;
        }
        const fmt = (t: number | null) => (t ? new Date(t).toISOString().split("T")[0] : "-");
        console.log(`Archive: ${s.total} record(s) in ${dir}`);
        console.log(`  span:     ${fmt(s.oldestArchivedAt)} → ${fmt(s.newestArchivedAt)}`);
        console.log(`  restored: ${s.restored}`);
        for (const [reason, n] of Object.entries(s.byReason).sort((a, b) => b[1] - a[1])) {
          console.log(`  ${reason.padEnd(13)} ${n}`);
        }
      } catch (error) {
        console.error("Archive stats failed:", error);
        process.exit(1);
      }
    });

  archive
    .command("search <query>")
    .description("Substring search over archived memories (works with the embedder offline)")
    .option("--limit <n>", "Maximum results", "20")
    .option("--json", "Output as JSON")
    .action(async (query, options) => {
      try {
        const hits = await searchArchive(archiveDirFor(context.store.dbPath), query, parseInt(options.limit) || 20);
        if (options.json) {
          console.log(formatJson(hits));
          return;
        }
        if (hits.length === 0) {
          console.log("No archived memories matched.");
          return;
        }
        console.log(`Found ${hits.length} archived memor(ies):\n`);
        hits.forEach((h, i) => {
          const when = new Date(h.archivedAt).toISOString().split("T")[0];
          console.log(`${i + 1}. [${h.category}:${h.scope}] archived ${when} (${h.archivedReason})`);
          console.log(`   ${h.text.slice(0, 180)}${h.text.length > 180 ? "…" : ""}`);
          console.log(`   id: ${h.id}\n`);
        });
      } catch (error) {
        console.error("Archive search failed:", error);
        process.exit(1);
      }
    });

  archive
    .command("restore <ids...>")
    .description("Pull archived memories back into the live store (re-embeds them)")
    .action(async (ids: string[]) => {
      try {
        if (!context.embedder) {
          console.error("Restore requires an embedder (not available in basic CLI mode).");
          process.exit(1);
        }
        const res = await restoreFromArchive(
          archiveDirFor(context.store.dbPath),
          context.store,
          (t) => context.embedder!.embedPassage(t),
          ids
        );
        console.log(`Restored ${res.restored} memor(ies).`);
        if (res.notFound.length > 0) console.error(`Not found in archive: ${res.notFound.join(", ")}`);
      } catch (error) {
        console.error("Archive restore failed:", error);
        process.exit(1);
      }
    });

  // Provenance repair. Creation timestamps drive recency and time-decay
  // scoring, and a rebuild that re-stamps them silently flattens both.
  memory
    .command("fix-timestamps <file>")
    .description("Restore creation timestamps from a backup JSONL (matches on exact text)")
    .option("--scope <scope>", "Limit to one scope")
    .option("--dry-run", "Report what would change without writing")
    .action(async (file, options) => {
      try {
        const fs = await import("node:fs/promises");
        const raw = await fs.readFile(file, "utf-8");
        const originals = new Map<string, number>();
        for (const line of raw.split("\n")) {
          if (!line.trim()) continue;
          try {
            const r = JSON.parse(line);
            if (typeof r.text !== "string" || !Number.isFinite(r.timestamp)) continue;
            const prev = originals.get(r.text);
            // Keep the earliest sighting — that is the real creation time.
            if (prev === undefined || r.timestamp < prev) originals.set(r.text, r.timestamp);
          } catch { /* skip malformed line */ }
        }

        const scopes = options.scope
          ? [options.scope]
          : Object.keys((await context.store.stats()).scopeCounts);

        let fixed = 0, unmatched = 0, alreadyOk = 0;
        for (const scope of scopes) {
          const rows = await context.store.list([scope], undefined, 5000, 0);
          for (const m of rows) {
            const orig = originals.get(m.text);
            if (orig === undefined) { unmatched++; continue; }
            if (Math.abs(m.timestamp - orig) <= 60_000) { alreadyOk++; continue; }
            if (options.dryRun) {
              const from = new Date(m.timestamp).toISOString().split("T")[0];
              const to = new Date(orig).toISOString().split("T")[0];
              console.log(`${from} -> ${to}  [${scope}] ${m.text.slice(0, 60)}`);
            } else {
              await context.store.update(m.id, { timestamp: orig }, [scope]);
            }
            fixed++;
          }
        }
        console.log(
          `${options.dryRun ? "Would repair" : "Repaired"} ${fixed} timestamp(s); ` +
          `${alreadyOk} already correct, ${unmatched} not present in the backup.`
        );
      } catch (error) {
        console.error("Timestamp repair failed:", error);
        process.exit(1);
      }
    });

  // TM4: verbatim session transcript search (FTS5 sidecar)
  memory
    .command("sessions <query>")
    .description("Verbatim full-text search over session transcripts (TM4)")
    .option("--agent <id>", "Filter by agent ID")
    .option("--limit <n>", "Maximum number of results", "8")
    .option("--reindex", "Drop and rebuild the FTS index first")
    .option("--json", "Output as JSON")
    .action(async (query, options) => {
      try {
        if (!context.sessionFts) {
          console.error("Session search is disabled (sessionSearch.enabled=false) or unavailable.");
          process.exit(1);
        }
        if (context.agentsDir) {
          if (options.reindex) {
            const r = await context.sessionFts.reindex(context.agentsDir);
            console.error(`Reindexed ${r.files} files (${r.messages} messages).`);
          } else {
            await context.sessionFts.indexAll(context.agentsDir); // cheap incremental catch-up
          }
        }
        const results = await context.sessionFts.search(query, {
          limit: parseInt(options.limit) || 8,
          agentId: options.agent,
        });
        if (options.json) {
          console.log(formatJson(results));
        } else if (results.length === 0) {
          console.log("No matching transcript messages found.");
        } else {
          console.log(`Found ${results.length} transcript matches:\n`);
          results.forEach((r, i) => {
            const date = (r.ts || "").replace("T", " ").slice(0, 16) || "unknown time";
            console.log(`${i + 1}. [${date}] ${r.agentId}/${r.role} (session ${r.sessionId.slice(0, 8)}…)`);
            console.log(`   ${r.snippet}\n`);
          });
        }
      } catch (error) {
        console.error("Session search failed:", error);
        process.exit(1);
      }
    });

  // Memory statistics
  memory
    .command("stats")
    .description("Show memory statistics")
    .option("--scope <scope>", "Stats for specific scope")
    .option("--json", "Output as JSON")
    .action(async (options) => {
      try {
        let scopeFilter: string[] | undefined;
        if (options.scope) {
          scopeFilter = [options.scope];
        }

        const stats = await context.store.stats(scopeFilter);
        const scopeStats = context.scopeManager.getStats();
        const retrievalConfig = context.retriever.getConfig();

        const summary = {
          memory: stats,
          scopes: scopeStats,
          retrieval: {
            mode: retrievalConfig.mode,
            hasFtsSupport: context.store.hasFtsSupport,
          },
        };

        if (options.json) {
          console.log(formatJson(summary));
        } else {
          console.log(`Memory Statistics:`);
          console.log(`• Total memories: ${stats.totalCount}`);
          console.log(`• Available scopes: ${scopeStats.totalScopes}`);
          console.log(`• Retrieval mode: ${retrievalConfig.mode}`);
          console.log(`• FTS support: ${context.store.hasFtsSupport ? 'Yes' : 'No'}`);
          console.log();

          console.log("Memories by scope:");
          Object.entries(stats.scopeCounts).forEach(([scope, count]) => {
            console.log(`  • ${scope}: ${count}`);
          });
          console.log();

          console.log("Memories by category:");
          Object.entries(stats.categoryCounts).forEach(([category, count]) => {
            console.log(`  • ${category}: ${count}`);
          });
        }
      } catch (error) {
        console.error("Failed to get statistics:", error);
        process.exit(1);
      }
    });

  // Delete memory
  memory
    .command("delete <id>")
    .description("Delete a specific memory by ID")
    .option("--scope <scope>", "Scope to delete from (for access control)")
    .action(async (id, options) => {
      try {
        let scopeFilter: string[] | undefined;
        if (options.scope) {
          scopeFilter = [options.scope];
        }

        const deleted = await context.store.delete(id, scopeFilter);

        if (deleted) {
          console.log(`Memory ${id} deleted successfully.`);
        } else {
          console.log(`Memory ${id} not found or access denied.`);
          process.exit(1);
        }
      } catch (error) {
        console.error("Failed to delete memory:", error);
        process.exit(1);
      }
    });

  // Bulk delete
  memory
    .command("delete-bulk")
    .description("Bulk delete memories with filters")
    .option("--scope <scopes...>", "Scopes to delete from (required)")
    .option("--before <date>", "Delete memories before this date (YYYY-MM-DD)")
    .option("--dry-run", "Show what would be deleted without actually deleting")
    .action(async (options) => {
      try {
        if (!options.scope || options.scope.length === 0) {
          console.error("At least one scope must be specified for safety.");
          process.exit(1);
        }

        let beforeTimestamp: number | undefined;
        if (options.before) {
          const date = new Date(options.before);
          if (isNaN(date.getTime())) {
            console.error("Invalid date format. Use YYYY-MM-DD.");
            process.exit(1);
          }
          beforeTimestamp = date.getTime();
        }

        if (options.dryRun) {
          console.log("DRY RUN - No memories will be deleted");
          console.log(`Filters: scopes=${options.scope.join(',')}, before=${options.before || 'none'}`);

          // Show what would be deleted
          const stats = await context.store.stats(options.scope);
          console.log(`Would delete from ${stats.totalCount} memories in matching scopes.`);
        } else {
          const deletedCount = await context.store.bulkDelete(options.scope, beforeTimestamp);
          console.log(`Deleted ${deletedCount} memories.`);
        }
      } catch (error) {
        console.error("Bulk delete failed:", error);
        process.exit(1);
      }
    });

  // Export memories
  memory
    .command("export")
    .description("Export memories to JSON")
    .option("--scope <scope>", "Export specific scope")
    .option("--category <category>", "Export specific category")
    .option("--output <file>", "Output file (default: stdout)")
    .action(async (options) => {
      try {
        let scopeFilter: string[] | undefined;
        if (options.scope) {
          scopeFilter = [options.scope];
        }

        const memories = await context.store.list(
          scopeFilter,
          options.category,
          1000 // Large limit for export
        );

        const exportData = {
          version: "1.0",
          exportedAt: new Date().toISOString(),
          count: memories.length,
          filters: {
            scope: options.scope,
            category: options.category,
          },
          memories: memories.map(m => ({
            ...m,
            vector: undefined, // Exclude vectors to reduce size
          })),
        };

        const output = formatJson(exportData);

        if (options.output) {
          const fs = await import("node:fs/promises");
          await fs.writeFile(options.output, output);
          console.log(`Exported ${memories.length} memories to ${options.output}`);
        } else {
          console.log(output);
        }
      } catch (error) {
        console.error("Export failed:", error);
        process.exit(1);
      }
    });

  // Import memories
  memory
    .command("import <file>")
    .description("Import memories from JSON file")
    .option("--scope <scope>", "Import into specific scope")
    .option("--dry-run", "Show what would be imported without actually importing")
    .action(async (file, options) => {
      try {
        const fs = await import("node:fs/promises");
        const content = await fs.readFile(file, "utf-8");
        const data = JSON.parse(content);

        if (!data.memories || !Array.isArray(data.memories)) {
          throw new Error("Invalid import file format");
        }

        if (options.dryRun) {
          console.log("DRY RUN - No memories will be imported");
          console.log(`Would import ${data.memories.length} memories`);
          if (options.scope) {
            console.log(`Target scope: ${options.scope}`);
          }
          return;
        }

        console.log(`Importing ${data.memories.length} memories...`);

        let imported = 0;
        let skipped = 0;

        if (!context.embedder) {
          console.error("Import requires an embedder (not available in basic CLI mode).");
          console.error("Use the plugin's memory_store tool or pass embedder to createMemoryCLI.");
          return;
        }

        const targetScope = options.scope || context.scopeManager.getDefaultScope();

        for (const memory of data.memories) {
          try {
            const text = memory.text;
            if (!text || typeof text !== "string" || text.length < 2) {
              skipped++;
              continue;
            }

            // Check for duplicates
            const existing = await context.retriever.retrieve({
              query: text,
              limit: 1,
              scopeFilter: [targetScope],
            });
            if (existing.length > 0 && existing[0].score > 0.95) {
              skipped++;
              continue;
            }

            const vector = await context.embedder.embedPassage(text);
            // store() stamps Date.now(), which silently rewrites every
            // creation time on a rebuild — an import restored months of
            // memories all dated "today" on 2026-09-18 and again on 09-20,
            // destroying the provenance that recency and time-decay scoring
            // depend on. Preserve the exported timestamp when there is one.
            const stored = await context.store.store({
              text,
              vector,
              importance: memory.importance ?? 0.7,
              category: memory.category || "other",
              scope: targetScope,
              metadata: typeof memory.metadata === "string" ? memory.metadata : undefined,
            });
            if (Number.isFinite(memory.timestamp) && memory.timestamp !== stored.timestamp) {
              await context.store.update(stored.id, { timestamp: memory.timestamp });
            }
            imported++;
          } catch (error) {
            console.warn(`Failed to import memory: ${error}`);
            skipped++;
          }
        }

        console.log(`Import completed: ${imported} imported, ${skipped} skipped`);
      } catch (error) {
        console.error("Import failed:", error);
        process.exit(1);
      }
    });

  // Re-embed an existing LanceDB into the current target DB (A/B testing)
  memory
    .command("reembed")
    .description("Re-embed memories from a source LanceDB database into the current target database")
    .requiredOption("--source-db <path>", "Source LanceDB database directory")
    .option("--batch-size <n>", "Batch size for embedding calls", "32")
    .option("--limit <n>", "Limit number of rows to process (for testing)")
    .option("--dry-run", "Show what would be re-embedded without writing")
    .option("--skip-existing", "Skip entries whose id already exists in the target DB")
    .option("--force", "Allow using the same source-db as the target dbPath (DANGEROUS)")
    .action(async (options) => {
      try {
        if (!context.embedder) {
          console.error("Re-embed requires an embedder (not available in basic CLI mode).");
          return;
        }

        const fs = await import("node:fs/promises");

        const sourceDbPath = options.sourceDb as string;
        const batchSize = clampInt(parseInt(options.batchSize, 10) || 32, 1, 128);
        const limit = options.limit ? clampInt(parseInt(options.limit, 10) || 0, 1, 1000000) : undefined;
        const dryRun = options.dryRun === true;
        const skipExisting = options.skipExisting === true;
        const force = options.force === true;

        // Safety: prevent accidental in-place re-embedding
        let sourceReal = sourceDbPath;
        let targetReal = context.store.dbPath;
        try {
          sourceReal = await fs.realpath(sourceDbPath);
        } catch {}
        try {
          targetReal = await fs.realpath(context.store.dbPath);
        } catch {}

        if (!force && sourceReal === targetReal) {
          console.error("Refusing to re-embed in-place: source-db equals target dbPath. Use a new dbPath or pass --force.");
          process.exit(1);
        }

        const lancedb = await loadLanceDB();
        const db = await lancedb.connect(sourceDbPath);
        const table = await db.openTable("memories");

        let query = table
          .query()
          .select(["id", "text", "category", "scope", "importance", "timestamp", "metadata"]);

        if (limit) query = query.limit(limit);

        const rows = (await query.toArray())
          .filter((r: any) => r && typeof r.text === "string" && r.text.trim().length > 0)
          .filter((r: any) => r.id && r.id !== "__schema__");

        if (rows.length === 0) {
          console.log("No source memories found.");
          return;
        }

        console.log(
          `Re-embedding ${rows.length} memories from ${sourceDbPath} → ${context.store.dbPath} (batchSize=${batchSize})`
        );

        if (dryRun) {
          console.log("DRY RUN - No memories will be written");
          console.log(`First example: ${rows[0].id?.slice?.(0, 8)} ${String(rows[0].text).slice(0, 80)}`);
          return;
        }

        let processed = 0;
        let imported = 0;
        let skipped = 0;

        for (let i = 0; i < rows.length; i += batchSize) {
          const batch = rows.slice(i, i + batchSize);
          const texts = batch.map((r: any) => String(r.text));
          const vectors = await context.embedder.embedBatchPassage(texts);

          for (let j = 0; j < batch.length; j++) {
            processed++;
            const row = batch[j];
            const vector = vectors[j];

            if (!vector || vector.length === 0) {
              skipped++;
              continue;
            }

            const id = String(row.id);
            if (skipExisting) {
              const exists = await context.store.hasId(id);
              if (exists) {
                skipped++;
                continue;
              }
            }

            const entry: MemoryEntry = {
              id,
              text: String(row.text),
              vector,
              category: (row.category as any) || "other",
              scope: (row.scope as string | undefined) || "global",
              importance: typeof row.importance === "number" ? row.importance : 0.7,
              timestamp: typeof row.timestamp === "number" ? row.timestamp : Date.now(),
              metadata: typeof row.metadata === "string" ? row.metadata : "{}",
            };

            await context.store.importEntry(entry);
            imported++;
          }

          if (processed % 100 === 0 || processed === rows.length) {
            console.log(`Progress: ${processed}/${rows.length} processed, ${imported} imported, ${skipped} skipped`);
          }
        }

        console.log(`Re-embed completed: ${imported} imported, ${skipped} skipped (processed=${processed}).`);
      } catch (error) {
        console.error("Re-embed failed:", error);
        process.exit(1);
      }
    });

  // Migration commands
  const migrate = memory
    .command("migrate")
    .description("Migration utilities");

  migrate
    .command("check")
    .description("Check if migration is needed from legacy memory-lancedb")
    .option("--source <path>", "Specific source database path")
    .action(async (options) => {
      try {
        const check = await context.migrator.checkMigrationNeeded(options.source);

        console.log("Migration Check Results:");
        console.log(`• Legacy database found: ${check.sourceFound ? 'Yes' : 'No'}`);
        if (check.sourceDbPath) {
          console.log(`• Source path: ${check.sourceDbPath}`);
        }
        if (check.entryCount !== undefined) {
          console.log(`• Entries to migrate: ${check.entryCount}`);
        }
        console.log(`• Migration needed: ${check.needed ? 'Yes' : 'No'}`);
      } catch (error) {
        console.error("Migration check failed:", error);
        process.exit(1);
      }
    });

  migrate
    .command("run")
    .description("Run migration from legacy memory-lancedb")
    .option("--source <path>", "Specific source database path")
    .option("--default-scope <scope>", "Default scope for migrated data", "global")
    .option("--dry-run", "Show what would be migrated without actually migrating")
    .option("--skip-existing", "Skip entries that already exist")
    .action(async (options) => {
      try {
        const result = await context.migrator.migrate({
          sourceDbPath: options.source,
          defaultScope: options.defaultScope,
          dryRun: options.dryRun,
          skipExisting: options.skipExisting,
        });

        console.log("Migration Results:");
        console.log(`• Status: ${result.success ? 'Success' : 'Failed'}`);
        console.log(`• Migrated: ${result.migratedCount}`);
        console.log(`• Skipped: ${result.skippedCount}`);
        if (result.errors.length > 0) {
          console.log(`• Errors: ${result.errors.length}`);
          result.errors.forEach(error => console.log(`  - ${error}`));
        }
        console.log(`• Summary: ${result.summary}`);

        if (!result.success) {
          process.exit(1);
        }
      } catch (error) {
        console.error("Migration failed:", error);
        process.exit(1);
      }
    });

  migrate
    .command("verify")
    .description("Verify migration results")
    .option("--source <path>", "Specific source database path")
    .action(async (options) => {
      try {
        const result = await context.migrator.verifyMigration(options.source);

        console.log("Migration Verification:");
        console.log(`• Valid: ${result.valid ? 'Yes' : 'No'}`);
        console.log(`• Source count: ${result.sourceCount}`);
        console.log(`• Target count: ${result.targetCount}`);

        if (result.issues.length > 0) {
          console.log("• Issues:");
          result.issues.forEach(issue => console.log(`  - ${issue}`));
        }

        if (!result.valid) {
          process.exit(1);
        }
      } catch (error) {
        console.error("Verification failed:", error);
        process.exit(1);
      }
    });
}

// ============================================================================
// Factory Function
// ============================================================================

export function createMemoryCLI(context: CLIContext) {
  // OpenClaw >=2026.5 invokes the CLI registrar with a context object
  // ({ program, config, workspaceDir, logger }); older versions passed the
  // commander Command directly. Support both by unwrapping ctx.program.
  return (ctx: { program?: Command } | Command) => {
    const program = (ctx as { program?: Command }).program ?? (ctx as Command);
    registerMemoryCLI(program, context);
  };
}