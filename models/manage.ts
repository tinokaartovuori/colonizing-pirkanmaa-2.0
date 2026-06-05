/**
 * Model registry CLI — the single management surface for trained models.
 *
 * Convention (see models/README.md and CLAUDE.md "Model management"):
 *   models/<arc>/<type>/<id>/{weights.json, manifest.json, bench.json?}
 *   models/registry.jsonl   — one summary line per model (the index)
 *   models/CHAMPION.json     — { champions: {"<arc>/<type>": id}, deployed: {...} }
 *   id = <arc>-<type>-<NNN>  (e.g. sd-az-001); NNN is per-(arc,type), zero-padded.
 *
 * Usage (via `npm run models -- <cmd>`):
 *   list                                          list all registered models
 *   show <id>                                     print a model's manifest
 *   register <weights.json> --arc sd --type az [--parent <id>] [--notes "..."]
 *                                                 import a trained model + write manifest + index
 *   promote <id> [--deployed]                     mark id champion for its arc/type (+deploy pointer)
 */
import * as fs from 'node:fs';
import * as path from 'node:path';
import { execSync } from 'node:child_process';

const ROOT = path.dirname(new URL(import.meta.url).pathname);
const REGISTRY = path.join(ROOT, 'registry.jsonl');
const CHAMPION = path.join(ROOT, 'CHAMPION.json');

type Manifest = {
  id: string; arc: string; type: string; version: number;
  created_utc: string; git_commit: string | null; parent: string | null;
  training_config: Record<string, unknown>;
  benchmarks: Record<string, unknown>;
  status: 'experimental' | 'champion' | 'deployed' | 'archived';
  notes: string;
};

function gitCommit(): string | null {
  try { return execSync('git rev-parse --short HEAD', { cwd: ROOT }).toString().trim(); }
  catch { return null; }
}
function readRegistry(): any[] {
  if (!fs.existsSync(REGISTRY)) return [];
  return fs.readFileSync(REGISTRY, 'utf8').split('\n').filter(Boolean).map(l => JSON.parse(l));
}
function readChampion(): { champions: Record<string, string>; deployed: Record<string, string> } {
  if (!fs.existsSync(CHAMPION)) return { champions: {}, deployed: {} };
  return JSON.parse(fs.readFileSync(CHAMPION, 'utf8'));
}

// --- args: positional + --flags ---
const argv = process.argv.slice(2);
const cmd = argv[0];
const pos: string[] = [];
const flags: Record<string, string> = {};
for (let i = 1; i < argv.length; i++) {
  if (argv[i].startsWith('--')) { flags[argv[i].slice(2)] = argv[i + 1] ?? 'true'; i++; }
  else pos.push(argv[i]);
}

function nextId(arc: string, type: string): { id: string; version: number } {
  const prefix = `${arc}-${type}-`;
  const max = readRegistry()
    .filter(r => r.id?.startsWith(prefix))
    .map(r => parseInt(r.id.slice(prefix.length), 10))
    .reduce((a, b) => Math.max(a, b), 0);
  const version = max + 1;
  return { id: `${prefix}${String(version).padStart(3, '0')}`, version };
}

function register() {
  const weights = pos[0];
  const arc = flags.arc, type = flags.type;
  if (!weights || !arc || !type) {
    console.error('usage: register <weights.json> --arc <sd> --type <az|hardbot|ga> [--parent <id>] [--notes "..."]');
    process.exit(1);
  }
  if (!fs.existsSync(weights)) { console.error(`no such weights file: ${weights}`); process.exit(1); }
  const { id, version } = nextId(arc, type);
  const dir = path.join(ROOT, arc, type, id);
  fs.mkdirSync(dir, { recursive: true });
  fs.copyFileSync(weights, path.join(dir, 'weights.json'));
  const manifest: Manifest = {
    id, arc, type, version,
    created_utc: new Date().toISOString(),
    git_commit: gitCommit(),
    parent: flags.parent ?? null,
    training_config: {},
    benchmarks: {},
    status: 'experimental',
    notes: flags.notes ?? '',
  };
  fs.writeFileSync(path.join(dir, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
  const line = JSON.stringify({
    id, arc, type, created_utc: manifest.created_utc,
    status: manifest.status, winrate_vs_hard: null, parent: manifest.parent, notes: manifest.notes,
  });
  fs.appendFileSync(REGISTRY, line + '\n');
  console.log(`registered ${id}\n  -> ${path.relative(path.dirname(ROOT), dir)}/\n  source: ${weights}`);
  console.log(`  edit ${id}/manifest.json to fill training_config + benchmarks (or have the trainer write them).`);
}

function list() {
  const rows = readRegistry();
  const champ = readChampion();
  if (!rows.length) { console.log('registry is empty — register a model with: npm run models -- register <weights.json> --arc sd --type az'); return; }
  const champSet = new Set(Object.values(champ.champions));
  const deplSet = new Set(Object.values(champ.deployed));
  console.log('ID              ARC/TYPE     CREATED               STATUS        vsHARD  FLAGS');
  for (const r of rows) {
    const flagsStr = [champSet.has(r.id) ? 'CHAMP' : '', deplSet.has(r.id) ? 'DEPLOYED' : ''].filter(Boolean).join(' ');
    console.log(
      `${(r.id ?? '?').padEnd(15)} ${`${r.arc}/${r.type}`.padEnd(12)} ${(r.created_utc ?? '').slice(0, 19).padEnd(20)} ` +
      `${(r.status ?? '').padEnd(13)} ${String(r.winrate_vs_hard ?? '-').padEnd(6)} ${flagsStr}`
    );
  }
}

function show() {
  const id = pos[0];
  const r = readRegistry().find(x => x.id === id);
  if (!r) { console.error(`no model ${id}`); process.exit(1); }
  const m = path.join(ROOT, r.arc, r.type, id, 'manifest.json');
  console.log(fs.readFileSync(m, 'utf8'));
}

function promote() {
  const id = pos[0];
  const r = readRegistry().find(x => x.id === id);
  if (!r) { console.error(`no model ${id}`); process.exit(1); }
  const champ = readChampion();
  champ.champions[`${r.arc}/${r.type}`] = id;
  if (flags.deployed) champ.deployed['weights.ts'] = id;
  fs.writeFileSync(CHAMPION, JSON.stringify(champ, null, 2) + '\n');
  console.log(`promoted ${id} → champion for ${r.arc}/${r.type}${flags.deployed ? ' (+deployed pointer)' : ''}`);
  if (flags.deployed) console.log('  NB: this only sets the pointer; actually writing src/ai/nn/weights.ts is a separate deploy step.');
}

switch (cmd) {
  case 'register': register(); break;
  case 'list': list(); break;
  case 'show': show(); break;
  case 'promote': promote(); break;
  default:
    console.log('commands: list | show <id> | register <weights.json> --arc <sd> --type <az|hardbot|ga> [--parent <id>] [--notes ...] | promote <id> [--deployed]');
}
