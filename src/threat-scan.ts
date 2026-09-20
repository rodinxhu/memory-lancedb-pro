/**
 * TM5 — Store-time Threat Scan (total-memory project)
 *
 * Everything written to memory is eventually injected back into an agent
 * prompt. The recall path already wraps memories in an UNTRUSTED banner;
 * this adds defense at the door: injection-style instruction patterns are
 * neutralized BEFORE storage, across every write path (auto-capture,
 * observations, compaction sweep, memory_store/memory_update tools).
 *
 * Inspired by Hermes' tools/threat_patterns.py (single pattern source,
 * scanned at snapshot-build time; matches replaced with [BLOCKED] markers).
 *
 * Patterns are deliberately CONSERVATIVE — a false positive corrupts a real
 * memory, so only unambiguous injection phrasing is matched.
 */

export interface ThreatHit {
  /** Which pattern category matched. */
  category: string;
  /** The matched span (truncated for logging). */
  match: string;
}

interface ThreatPattern {
  category: string;
  regex: RegExp;
}

const THREAT_PATTERNS: ThreatPattern[] = [
  // Instruction override
  {
    category: "instruction-override",
    regex: /\b(ignore|disregard|forget|override)\s+(all\s+|any\s+|your\s+|the\s+)?(previous|prior|above|earlier|system)\s+(instructions?|prompts?|rules?|context|training)\b/gi,
  },
  {
    category: "instruction-override",
    regex: /忽略(之前|以上|所有|先前)的?(指令|规则|提示|说明)/g,
  },
  // Role hijack
  {
    category: "role-hijack",
    regex: /\byou\s+are\s+no\s+longer\s+(an?\s+)?(assistant|ai|bound|restricted)\b/gi,
  },
  {
    category: "role-hijack",
    regex: /\b(enter|enable|activate)\s+(dan|developer|jailbreak|god)\s*mode\b/gi,
  },
  // Prompt exfiltration
  {
    category: "prompt-exfiltration",
    regex: /\b(reveal|print|show|repeat|output)\s+(me\s+)?(your\s+)?(full\s+|entire\s+|hidden\s+)?(system\s+prompt|initial\s+instructions|developer\s+message)\b/gi,
  },
  // Fake conversation structure (tag injection)
  {
    category: "tag-injection",
    regex: /<\s*\/?\s*(system|assistant|developer)\s*>/gi,
  },
  {
    category: "tag-injection",
    regex: /\[\s*(SYSTEM|INST)\s*\]/g,
  },
];

export interface NeutralizeResult {
  text: string;
  hits: ThreatHit[];
}

/**
 * Replace injection-style spans with an inert marker. Returns the cleaned
 * text and what was found (for logging). Cheap — a handful of regex passes.
 */
export function neutralizeThreats(text: string): NeutralizeResult {
  const hits: ThreatHit[] = [];
  let cleaned = text;
  for (const { category, regex } of THREAT_PATTERNS) {
    regex.lastIndex = 0;
    cleaned = cleaned.replace(regex, (match) => {
      hits.push({ category, match: match.slice(0, 60) });
      return `[blocked:${category}]`;
    });
  }
  return { text: cleaned, hits };
}

/** Convenience for write paths: clean text, log hits via the provided logger. */
export function makeSanitizer(
  enabled: boolean,
  logger: { warn: (msg: string) => void }
): (text: string) => string {
  if (!enabled) return (text) => text;
  return (text) => {
    const { text: cleaned, hits } = neutralizeThreats(text);
    if (hits.length > 0) {
      logger.warn(
        `threat-scan: neutralized ${hits.length} pattern(s) before storage: ` +
          hits.map((h) => `${h.category} ("${h.match}")`).join(", ")
      );
    }
    return cleaned;
  };
}

export interface ThreatScanConfig {
  enabled?: boolean;
}

export function parseThreatScanConfig(value: unknown): Required<ThreatScanConfig> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return { enabled: true };
  }
  return { enabled: (value as Record<string, unknown>).enabled !== false };
}
