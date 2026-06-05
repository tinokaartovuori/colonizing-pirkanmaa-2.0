// Build-process logger/roadmap CLI — keeps `build-status.json` (phase roadmap)
// and `build-log.jsonl` (narrative log) at the repo root, which the dashboard's
// "Rakennusprosessi" tab renders. Dependency-free (node:fs only).
//
// Usage (vite-node strips the script path + `--`, so we detect the subcommand by
// name, not argv position):
//   npx vite-node training/progress.ts log <phaseId> "message" [--level milestone|info|warn]
//   npx vite-node training/progress.ts phase <phaseId> <pending|active|done> [--progress 0.5]
//   npx vite-node training/progress.ts show
//
// The Rust trainer can also just append a build-log.jsonl line directly
// ({"ts","phase","msg","level"}) — same format.

import { existsSync, readFileSync, writeFileSync, appendFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const STATUS = resolve(REPO, 'build-status.json');
const LOG = resolve(REPO, 'build-log.jsonl');

const argv = process.argv;
const SUBS = ['log', 'phase', 'show'];
const ci = argv.findIndex((a) => SUBS.includes(a));
if (ci < 0) {
  console.error('usage: progress.ts <log|phase|show> …');
  process.exit(1);
}
const cmd = argv[ci];
const rest = argv.slice(ci + 1);
function flag(name: string): string | undefined {
  const i = rest.indexOf('--' + name);
  return i >= 0 ? rest[i + 1] : undefined;
}
const positional = rest.filter((a, i) => !a.startsWith('--') && !(i > 0 && rest[i - 1].startsWith('--')));

function readStatus(): { title?: string; phases: Record<string, unknown>[] } {
  if (!existsSync(STATUS)) return { phases: [] };
  try { return JSON.parse(readFileSync(STATUS, 'utf8')); } catch { return { phases: [] }; }
}

if (cmd === 'log') {
  const phase = positional[0] || '';
  const msg = positional.slice(1).join(' ');
  const level = flag('level') || 'info';
  const line = JSON.stringify({ ts: new Date().toISOString(), phase, msg, level });
  appendFileSync(LOG, line + '\n');
  console.log('logged:', line);
} else if (cmd === 'phase') {
  const id = positional[0];
  const status = positional[1];
  const progress = flag('progress');
  const st = readStatus();
  const p = st.phases.find((x) => (x as { id?: string }).id === id);
  if (!p) { console.error('no phase with id', id); process.exit(1); }
  if (status) (p as { status?: string }).status = status;
  if (progress !== undefined) (p as { progress?: number }).progress = Number(progress);
  else if (status === 'done') delete (p as { progress?: number }).progress;
  writeFileSync(STATUS, JSON.stringify(st, null, 2) + '\n');
  console.log('updated phase', id, '→', status, progress !== undefined ? '(' + progress + ')' : '');
} else if (cmd === 'show') {
  const st = readStatus();
  console.log(st.title || 'build');
  for (const p of st.phases) {
    const x = p as { id: string; status: string; title: string; progress?: number };
    console.log(`  [${x.status}]${x.progress != null ? ' ' + Math.round(x.progress * 100) + '%' : ''} ${x.id} — ${x.title}`);
  }
}
