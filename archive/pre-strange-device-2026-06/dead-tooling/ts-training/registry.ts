// Model version registry — a durable record of every trained version, so a good
// checkpoint is never lost and the dashboard can show the lineage + which one is
// deployed to the browser game.
//
// Artifact: models/registry.jsonl at the repo root, one JSON line per version:
//   {id, name, created, kind, arch, params, gens, winRateVsHard, tileFrac,
//    leaf, search, path, notes, deployed}
//
// Dependency-free. Library + CLI:
//   npx vite-node training/registry.ts -- add --name p2-champ --kind policy \
//       --path rust-trainer/checkpoints/champion.json --arch 63,24,16,1 \
//       --params 1953 --gens 201 --winrate 0.15 --tilefrac 0.217 --leaf static \
//       --notes "Phase-2 spatial-local champion"
//   npx vite-node training/registry.ts -- list
//   npx vite-node training/registry.ts -- promote --id <id>   # marks deployed (clears others of same kind)

import { existsSync, readFileSync, writeFileSync, appendFileSync, mkdirSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

export const REPO_ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)));
export const REGISTRY_PATH = resolve(REPO_ROOT, 'models', 'registry.jsonl');

export type ModelKind = 'policy' | 'value' | 'alphazero';

export interface ModelEntry {
  id: string;
  name: string;
  created: string;
  kind: ModelKind;
  arch: number[];
  params: number | null;
  gens: number | null;
  /** Real win-rate vs the hard heuristic (the oracle), if benchmarked. */
  winRateVsHard: number | null;
  tileFrac: number | null;
  /** Leaf-eval mode the win-rate was measured with (none|static|value|rollout). */
  leaf: string | null;
  /** Free-form search config summary, if any. */
  search: string | null;
  /** Path to the weights file, relative to repo root. */
  path: string;
  notes: string;
  deployed: boolean;
}

export function readRegistry(): ModelEntry[] {
  if (!existsSync(REGISTRY_PATH)) return [];
  const out: ModelEntry[] = [];
  for (const line of readFileSync(REGISTRY_PATH, 'utf8').split('\n')) {
    const s = line.trim();
    if (!s) continue;
    try {
      out.push(JSON.parse(s) as ModelEntry);
    } catch {
      // skip malformed line
    }
  }
  return out;
}

function writeRegistry(entries: ModelEntry[]): void {
  mkdirSync(dirname(REGISTRY_PATH), { recursive: true });
  writeFileSync(REGISTRY_PATH, entries.map((e) => JSON.stringify(e)).join('\n') + '\n');
}

function makeId(name: string): string {
  // Sortable, human-readable, collision-resistant enough for a single-user registry.
  const stamp = new Date().toISOString().replace(/[-:T.]/g, '').slice(0, 14);
  const slug = name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
  return `${stamp}-${slug}`;
}

export function registerModel(
  entry: Omit<ModelEntry, 'id' | 'created' | 'deployed'> & Partial<Pick<ModelEntry, 'id' | 'created' | 'deployed'>>,
): ModelEntry {
  const full: ModelEntry = {
    deployed: false,
    created: new Date().toISOString(),
    id: makeId(entry.name),
    ...entry,
  } as ModelEntry;
  mkdirSync(dirname(REGISTRY_PATH), { recursive: true });
  appendFileSync(REGISTRY_PATH, JSON.stringify(full) + '\n');
  return full;
}

/** Mark one version deployed; clear the flag on others of the same kind. */
export function promoteModel(id: string): void {
  const entries = readRegistry();
  const target = entries.find((e) => e.id === id);
  if (!target) throw new Error(`no model with id ${id}`);
  for (const e of entries) {
    if (e.kind === target.kind) e.deployed = e.id === id;
  }
  writeRegistry(entries);
}

// --- CLI -------------------------------------------------------------------

function flag(args: string[], name: string): string | undefined {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : undefined;
}
function num(v: string | undefined): number | null {
  return v === undefined ? null : Number(v);
}

function runCli(argv: string[]): void {
  const args = argv.includes('--') ? argv.slice(argv.indexOf('--') + 1) : argv.slice(2);
  const cmd = args[0];
  if (cmd === 'add') {
    const name = flag(args, '--name');
    const path = flag(args, '--path');
    if (!name || !path) throw new Error('add requires --name and --path');
    const archStr = flag(args, '--arch');
    const entry = registerModel({
      name,
      kind: (flag(args, '--kind') ?? 'policy') as ModelKind,
      arch: archStr ? archStr.split(',').map(Number) : [],
      params: num(flag(args, '--params')),
      gens: num(flag(args, '--gens')),
      winRateVsHard: num(flag(args, '--winrate')),
      tileFrac: num(flag(args, '--tilefrac')),
      leaf: flag(args, '--leaf') ?? null,
      search: flag(args, '--search') ?? null,
      path,
      notes: flag(args, '--notes') ?? '',
    });
    console.log(`registered ${entry.id}`);
  } else if (cmd === 'promote') {
    const id = flag(args, '--id');
    if (!id) throw new Error('promote requires --id');
    promoteModel(id);
    console.log(`promoted ${id} (deployed)`);
  } else if (cmd === 'list') {
    for (const e of readRegistry()) {
      const wr = e.winRateVsHard != null ? `${(e.winRateVsHard * 100).toFixed(1)}% vs hard` : 'unbenched';
      console.log(`${e.deployed ? '★' : ' '} ${e.id}  [${e.kind}]  ${wr}  ${e.notes}`);
    }
  } else {
    console.error('usage: registry.ts -- <add|promote|list> [flags]');
    process.exit(1);
  }
}

// vite-node strips the script path from argv, so detect "run as CLI" by a
// recognized subcommand. When imported (e.g. by the dashboard) the leading arg
// is something else (--dir/--port) and the CLI stays dormant.
{
  const a = process.argv.includes('--') ? process.argv.slice(process.argv.indexOf('--') + 1) : process.argv.slice(2);
  if (['add', 'promote', 'list'].includes(a[0])) runCli(process.argv);
}
