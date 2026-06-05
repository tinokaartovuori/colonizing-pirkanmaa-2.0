// Build-process status + narrative log — the source of truth the dashboard
// reads to show "where the whole AI build is at".
//
// Two artifacts, both at the repo root (independent of any --dir checkpoint dir):
//   - build-status.json   : the phase roadmap. One entry per phase with a
//                           status (pending|active|done) + optional detail +
//                           optional progress (0..1). Drives the progress bar.
//   - build-log.jsonl      : append-only narrative log. One JSON line per event
//                           {ts, phase, level, msg}. Drives the log feed.
//
// Dependency-free (node:fs/path/url only). Importable as a library AND runnable
// as a CLI so Rust/TS training scripts and humans can both write to it:
//
//   npx vite-node training/progress.ts -- log  --phase <id> --level milestone --msg "text"
//   npx vite-node training/progress.ts -- phase --id <id> --status active --detail "text" [--progress 0.4]
//   npx vite-node training/progress.ts -- show
//
// The Rust trainer can also just append a JSONL line to build-log.jsonl itself.

import { existsSync, readFileSync, writeFileSync, appendFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

export const REPO_ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)));
export const STATUS_PATH = resolve(REPO_ROOT, 'build-status.json');
export const LOG_PATH = resolve(REPO_ROOT, 'build-log.jsonl');

export type PhaseStatus = 'pending' | 'active' | 'done';
export type LogLevel = 'info' | 'milestone' | 'warn';

export interface Phase {
  id: string;
  title: string;
  status: PhaseStatus;
  detail?: string;
  /** Optional fine-grained progress within an active phase, 0..1. */
  progress?: number;
}

export interface BuildStatus {
  title: string;
  updated: string;
  phases: Phase[];
}

export interface LogEvent {
  ts: string;
  phase: string;
  level: LogLevel;
  msg: string;
}

function now(): string {
  return new Date().toISOString();
}

export function readStatus(): BuildStatus | null {
  if (!existsSync(STATUS_PATH)) return null;
  try {
    return JSON.parse(readFileSync(STATUS_PATH, 'utf8')) as BuildStatus;
  } catch {
    return null;
  }
}

export function writeStatus(status: BuildStatus): void {
  status.updated = now();
  writeFileSync(STATUS_PATH, JSON.stringify(status, null, 2) + '\n');
}

/** Update one phase's status/detail/progress, creating it if unknown. */
export function setPhase(
  id: string,
  patch: Partial<Omit<Phase, 'id'>>,
): void {
  const status = readStatus() ?? { title: 'AI build', updated: now(), phases: [] };
  const phase = status.phases.find((p) => p.id === id);
  if (phase) {
    Object.assign(phase, patch);
  } else {
    status.phases.push({ id, title: patch.title ?? id, status: patch.status ?? 'pending', ...patch });
  }
  writeStatus(status);
}

export function readLogEvents(limit = Infinity): LogEvent[] {
  if (!existsSync(LOG_PATH)) return [];
  let raw: string;
  try {
    raw = readFileSync(LOG_PATH, 'utf8');
  } catch {
    return [];
  }
  const out: LogEvent[] = [];
  for (const line of raw.split('\n')) {
    const s = line.trim();
    if (!s) continue;
    try {
      out.push(JSON.parse(s) as LogEvent);
    } catch {
      // skip partial/malformed line (tolerates concurrent writers mid-flush)
    }
  }
  return limit === Infinity ? out : out.slice(-limit);
}

/** Append one narrative log event. */
export function logEvent(phase: string, msg: string, level: LogLevel = 'info'): void {
  const ev: LogEvent = { ts: now(), phase, level, msg };
  appendFileSync(LOG_PATH, JSON.stringify(ev) + '\n');
}

// --- CLI -------------------------------------------------------------------

function flag(args: string[], name: string): string | undefined {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : undefined;
}

function runCli(argv: string[]): void {
  const args = argv.includes('--') ? argv.slice(argv.indexOf('--') + 1) : argv.slice(2);
  const cmd = args[0];
  if (cmd === 'log') {
    const phase = flag(args, '--phase') ?? 'general';
    const level = (flag(args, '--level') ?? 'info') as LogLevel;
    const msg = flag(args, '--msg') ?? '';
    if (!msg) throw new Error('log requires --msg');
    logEvent(phase, msg, level);
    console.log(`logged [${phase}/${level}] ${msg}`);
  } else if (cmd === 'phase') {
    const id = flag(args, '--id');
    if (!id) throw new Error('phase requires --id');
    const patch: Partial<Phase> = {};
    const status = flag(args, '--status') as PhaseStatus | undefined;
    if (status) patch.status = status;
    const title = flag(args, '--title');
    if (title) patch.title = title;
    const detail = flag(args, '--detail');
    if (detail !== undefined) patch.detail = detail;
    const progress = flag(args, '--progress');
    if (progress !== undefined) patch.progress = Number(progress);
    setPhase(id, patch);
    console.log(`phase ${id} -> ${JSON.stringify(patch)}`);
  } else if (cmd === 'show') {
    console.log(JSON.stringify(readStatus(), null, 2));
    console.log(`\n${readLogEvents(20).map((e) => `${e.ts} [${e.phase}] ${e.msg}`).join('\n')}`);
  } else {
    console.error('usage: progress.ts -- <log|phase|show> [flags]');
    process.exit(1);
  }
}

// vite-node strips the script path from argv, so detect "run as CLI" by a
// recognized subcommand. When imported (e.g. by the dashboard) the leading arg
// is something else (--dir/--port) and the CLI stays dormant.
{
  const a = process.argv.includes('--') ? process.argv.slice(process.argv.indexOf('--') + 1) : process.argv.slice(2);
  if (['log', 'phase', 'show'].includes(a[0])) runCli(process.argv);
}
