// Live, auto-refreshing web dashboard for GA training progress.
//
// A DEPENDENCY-FREE server (node:http/fs/path/url only) that serves a
// self-contained HTML page polling /data.json every 5s and re-rendering inline
// SVG charts in place. The trainer writes (and flushes per generation)
// `log.jsonl` plus `champion.json`/`hof.json`; a benchmark sidecar writes the
// latest result to `benchmark.json` AND appends a time series to
// `benchmark-history.jsonl` (one JSON line per benchmark: {gen,winRate,...,ts}).
// Everything is read FRESH per request.
//
// Run:
//   npx vite-node training/serve-dashboard.ts -- --dir <checkpoints-dir> --port <n>
//
// Dashboard features:
//   - Win-rate vs hard AI plotted as a real CURVE over generations (from
//     benchmark-history.jsonl), not just a flat marker.
//   - Long-term vs short-term views via a window selector (All / 200 / 100 /
//     50 / 25 most-recent generations).
//   - Optional smoothing (rolling mean) to read trend through GA noise.

import { createServer } from 'node:http';
import { existsSync, readFileSync, statSync } from 'node:fs';
import { resolve, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO_ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)));
const DEFAULT_DIR = 'rust-trainer/checkpoints-cnn';
const DEFAULT_PORT = 8787;

function parseArgs(argv: string[]): { dir: string; port: number } {
  const args = argv.includes('--') ? argv.slice(argv.indexOf('--') + 1) : argv.slice(2);
  let dir = DEFAULT_DIR;
  let port = DEFAULT_PORT;
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === '--dir') dir = args[++i];
    else if (a === '--port') port = Number(args[++i]) || DEFAULT_PORT;
  }
  return { dir, port };
}

interface LogRow {
  gen: number;
  [k: string]: unknown;
}

// Read FRESH on every request; tolerate a partial final line (training mid-write).
function readLog(dir: string): { rows: LogRow[]; mtime: string | null } {
  const path = join(dir, 'log.jsonl');
  if (!existsSync(path)) return { rows: [], mtime: null };
  let raw: string;
  let mtime: string | null = null;
  try {
    raw = readFileSync(path, 'utf8');
  } catch {
    return { rows: [], mtime: null };
  }
  try {
    mtime = statSync(path).mtime.toISOString();
  } catch {
    /* ignore */
  }
  const rows: LogRow[] = [];
  for (const line of raw.split('\n')) {
    const s = line.trim();
    if (!s) continue;
    try {
      rows.push(JSON.parse(s) as LogRow);
    } catch {
      // skip malformed line (e.g. partial/incomplete final line mid-write)
    }
  }
  return { rows, mtime };
}

function readBenchmark(dir: string): unknown | null {
  const path = join(dir, 'benchmark.json');
  if (!existsSync(path)) return null;
  try {
    return JSON.parse(readFileSync(path, 'utf8'));
  } catch {
    return null;
  }
}

// Time series of benchmark results vs the hard heuristic, one JSON line per
// benchmark. Lines look like {gen, winRate, lossRate, timeoutRate, tileFrac, ts}.
function readWinHistory(dir: string): Record<string, unknown>[] {
  const path = join(dir, 'benchmark-history.jsonl');
  if (!existsSync(path)) return [];
  let raw: string;
  try {
    raw = readFileSync(path, 'utf8');
  } catch {
    return [];
  }
  const out: Record<string, unknown>[] = [];
  for (const line of raw.split('\n')) {
    const s = line.trim();
    if (!s) continue;
    try {
      out.push(JSON.parse(s));
    } catch {
      /* skip partial line */
    }
  }
  return out;
}

// --- repo-root build artifacts (independent of the --dir checkpoint dir) -----
// The build-process status/log + model registry + research writeup live at the
// repo root so the dashboard shows the whole AI build regardless of which
// checkpoint dir a training run targets.
function readJsonSafe(path: string): unknown | null {
  if (!existsSync(path)) return null;
  try {
    return JSON.parse(readFileSync(path, 'utf8'));
  } catch {
    return null;
  }
}
function readJsonlSafe(path: string): Record<string, unknown>[] {
  if (!existsSync(path)) return [];
  let raw: string;
  try {
    raw = readFileSync(path, 'utf8');
  } catch {
    return [];
  }
  const out: Record<string, unknown>[] = [];
  for (const line of raw.split('\n')) {
    const s = line.trim();
    if (!s) continue;
    try {
      out.push(JSON.parse(s));
    } catch {
      /* skip partial/malformed line */
    }
  }
  return out;
}
function readText(path: string): string | null {
  if (!existsSync(path)) return null;
  try {
    return readFileSync(path, 'utf8');
  } catch {
    return null;
  }
}

// Dedupe rows by `gen` (keeping the LAST occurrence — the most recent run's
// value) and sort ascending, so a checkpoint dir that accidentally holds lines
// from two runs still renders a clean, monotonic series instead of a jumbled
// curve that jumps backward in `gen`.
function dedupeByGen(rows: Record<string, unknown>[]): Record<string, unknown>[] {
  const byGen = new Map<number, Record<string, unknown>>();
  for (const r of rows) {
    const g = Number((r as { gen?: unknown }).gen);
    if (Number.isFinite(g)) byGen.set(g, r);
  }
  return [...byGen.keys()].sort((a, b) => a - b).map((g) => byGen.get(g) as Record<string, unknown>);
}

function buildData(dir: string): Record<string, unknown> {
  const { rows: rawRows, mtime } = readLog(dir);
  const rows = dedupeByGen(rawRows) as LogRow[];
  const benchmark = readBenchmark(dir);
  const winHistory = dedupeByGen(readWinHistory(dir));
  return {
    dir,
    updated: new Date().toISOString(),
    log: rows,
    benchmark,
    winHistory,
    // Latest recorded game replays (trainer writes both every ~5 iters): champion
    // vs the HARD bot, and champion vs champion (AI-vs-AI self-play).
    replay: readJsonSafe(join(dir, 'replay.json')),
    replaySelf: readJsonSafe(join(dir, 'replay_selfplay.json')),
    // Scripted-opponent replays (champion vs one of the 5 Lever-C training strategies).
    // The trainer writes these alongside `replay.json` every `replay_every` iters; one
    // heavy MCTS game per strategy. Empty/missing files surface as `null` so the
    // dashboard's empty-state can render a friendly hint. File names mirror the script
    // `mode` tag (see `script_mode_tag` in cnn_train.rs) so adding a new strategy is a
    // single coordinated edit across both files.
    replayVsArmyRush: readJsonSafe(join(dir, 'replay_vs_armyrush.json')),
    replayVsHqRush: readJsonSafe(join(dir, 'replay_vs_hqrush.json')),
    replayVsDeviceRush: readJsonSafe(join(dir, 'replay_vs_devicerush.json')),
    replayVsGarrison: readJsonSafe(join(dir, 'replay_vs_garrison.json')),
    replayVsExpert: readJsonSafe(join(dir, 'replay_vs_expert.json')),
    replayVsMarcher: readJsonSafe(join(dir, 'replay_vs_marcher.json')),
    // PILLAR 6 — rebuilt SD3 league opponents (the kinds the curriculum now samples).
    replayVsRusher: readJsonSafe(join(dir, 'replay_vs_rusher.json')),
    replayVsFortress: readJsonSafe(join(dir, 'replay_vs_fortress.json')),
    replayVsStrongArmy: readJsonSafe(join(dir, 'replay_vs_strongarmy.json')),
    // CNN spatial heatmap: a representative mid-game board + the net's per-tile
    // policy desirability and value. Written each benchmark by the CNN trainer;
    // null for runs/arcs that don't emit it (panel stays hidden in that case).
    spatial: readJsonSafe(join(dir, 'spatial.json')),
    latest: rows.length ? rows[rows.length - 1] : null,
    logMtime: mtime,
    // Build-process + registry + research (repo-root artifacts).
    buildStatus: readJsonSafe(join(REPO_ROOT, 'build-status.json')),
    buildLog: readJsonlSafe(join(REPO_ROOT, 'build-log.jsonl')),
    registry: readJsonlSafe(join(REPO_ROOT, 'models', 'registry.jsonl')),
    // Research/development docs shown under the Tutkimus tab (a sub-nav switches).
    research: [
      { id: 'research', title: 'Tutkimus', md: readText(join(REPO_ROOT, 'rust-trainer', 'TRAINING-RESEARCH.md')) },
      { id: 'design', title: 'AlphaZero-suunnitelma', md: readText(join(REPO_ROOT, 'rust-trainer', 'ALPHAZERO-DESIGN.md')) },
      { id: 'reward', title: 'Palkkiosignaalit', md: readText(join(REPO_ROOT, 'rust-trainer', 'REWARD-DESIGN.md')) },
    ].filter((d) => d.md != null),
  };
}

const { dir: rawDir, port } = parseArgs(process.argv);
const DIR = resolve(REPO_ROOT, rawDir);

const server = createServer((req, res) => {
  const url = (req.url || '/').split('?')[0];
  if (url === '/data.json') {
    let body: string;
    try {
      body = JSON.stringify(buildData(rawDir));
    } catch {
      body = JSON.stringify({
        dir: rawDir,
        updated: new Date().toISOString(),
        log: [],
        benchmark: null,
        winHistory: [],
        replay: null,
        replaySelf: null,
        replayVsArmyRush: null,
        replayVsHqRush: null,
        replayVsDeviceRush: null,
        replayVsGarrison: null,
        replayVsExpert: null,
        replayVsMarcher: null,
        replayVsRusher: null,
        replayVsFortress: null,
        replayVsStrongArmy: null,
        spatial: null,
        latest: null,
        logMtime: null,
        buildStatus: null,
        buildLog: [],
        registry: [],
        research: [],
      });
    }
    res.writeHead(200, {
      'content-type': 'application/json; charset=utf-8',
      'cache-control': 'no-store',
    });
    res.end(body);
    return;
  }
  if (url === '/' || url === '/index.html') {
    res.writeHead(200, { 'content-type': 'text/html; charset=utf-8', 'cache-control': 'no-store' });
    res.end(PAGE);
    return;
  }
  res.writeHead(404, { 'content-type': 'text/plain' });
  res.end('not found');
});

server.listen(port, '127.0.0.1', () => {
  console.log(`Live training dashboard serving ${DIR}`);
  console.log(`  http://127.0.0.1:${port}/`);
});

// ---------------------------------------------------------------------------
// The self-contained HTML page (inline CSS + inline JS, no CDNs). The client
// fetches /data.json every 5000ms and re-renders charts + header in place.
// ---------------------------------------------------------------------------
const PAGE = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Colonizing Pirkanmaa — AI Dashboard</title>
<style>
  :root { --bg:#0f1419; --panel:#1a2027; --ink:#e6edf3; --muted:#8b97a3;
          --grid:#2a323c; --best:#4dd2a0; --mean:#5aa9ff; --median:#c792ea;
          --win:#ffcb6b; --len:#82aaff; --bank:#ff6b6b; --div:#7fdbff;
          --sigma:#ff9e64; --wt:#b388ff; --tile:#64ffda; --live:#4dd2a0;
          --accent:#4dd2a0; --loss:#ff6b6b; --timeout:#8b97a3; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:14px/1.5 ui-monospace,Menlo,Consolas,monospace; padding:24px; }
  h1 { font-size:20px; margin:0 0 4px; }
  .sub { color:var(--muted); margin:0 0 12px; font-size:12px; }
  .status { font-size:12px; margin:0 0 16px; color:var(--muted); }
  .status .dot { color:var(--live); }
  .status.stale .dot { color:var(--bank); }
  .summary { display:flex; flex-wrap:wrap; gap:12px; margin-bottom:16px; }
  .stat { background:var(--panel); border:1px solid var(--grid); border-radius:8px;
          padding:12px 16px; min-width:140px; }
  .stat.hero { border-color:var(--win); }
  .stat .k { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.06em; }
  .stat .v { font-size:22px; font-weight:600; margin-top:4px; }
  .stat .v small { font-size:11px; color:var(--muted); font-weight:400; }
  .tipq { color:var(--muted); cursor:help; font-size:10px; }
  .stat[title], .chart[title] { cursor:help; }
  .up { color:var(--best); } .down { color:var(--bank); }
  .controls { display:flex; flex-wrap:wrap; gap:18px; align-items:center;
              background:var(--panel); border:1px solid var(--grid); border-radius:8px;
              padding:10px 14px; margin-bottom:18px; font-size:12px; }
  .controls .grp { display:flex; align-items:center; gap:6px; }
  .controls .lab { color:var(--muted); text-transform:uppercase; letter-spacing:.06em; font-size:11px; }
  .btn { background:#222b35; color:var(--ink); border:1px solid var(--grid); border-radius:6px;
         padding:4px 10px; cursor:pointer; font:inherit; font-size:12px; }
  .btn:hover { border-color:var(--muted); }
  .btn.on { background:var(--accent); color:#0f1419; border-color:var(--accent); font-weight:600; }
  .charts { display:grid; grid-template-columns:repeat(auto-fit,minmax(380px,1fr)); gap:16px; }
  .chart { background:var(--panel); border:1px solid var(--grid); border-radius:8px; padding:14px; }
  .chart.wide { grid-column:1 / -1; }
  .chart h2 { font-size:13px; margin:0 0 10px; font-weight:600; }
  .chart h2 .hint { color:var(--muted); font-weight:400; font-size:11px; }
  .legend { display:flex; flex-wrap:wrap; gap:14px; margin-top:8px; font-size:11px; color:var(--muted); }
  .legend span { display:inline-flex; align-items:center; gap:5px; }
  .swatch { width:12px; height:3px; border-radius:2px; display:inline-block; }
  .note { color:var(--muted); font-size:11px; margin-top:6px; }
  svg { width:100%; height:200px; display:block; }
  .chart.wide svg { height:260px; }
  .axis { fill:var(--muted); font-size:10px; }
  .gridline { stroke:var(--grid); stroke-width:1; }
  .marker { stroke-dasharray:4 3; }
  .empty { color:var(--muted); font-size:12px; padding:20px 0; text-align:center; }

  /* tab bar */
  .tabs { display:flex; gap:6px; margin:0 0 18px; border-bottom:1px solid var(--grid); }
  .tab { background:none; border:none; border-bottom:2px solid transparent; color:var(--muted);
         padding:8px 14px; cursor:pointer; font:inherit; font-size:13px; }
  .tab:hover { color:var(--ink); }
  .tab.on { color:var(--accent); border-bottom-color:var(--accent); font-weight:600; }
  .panel-hidden { display:none; }

  /* build process */
  .overall { background:var(--panel); border:1px solid var(--grid); border-radius:8px; padding:14px 16px; margin-bottom:16px; }
  .overall .track { height:10px; border-radius:5px; background:#222b35; overflow:hidden; margin-top:8px; }
  .overall .fill { height:100%; background:linear-gradient(90deg,var(--accent),var(--div)); border-radius:5px; transition:width .4s; }
  .phases { display:flex; flex-direction:column; gap:10px; }
  .phase { background:var(--panel); border:1px solid var(--grid); border-left:3px solid var(--grid); border-radius:8px; padding:12px 16px; }
  .phase.active { border-left-color:var(--accent); }
  .phase.done { border-left-color:var(--best); opacity:.8; }
  .phase .top { display:flex; align-items:center; gap:10px; }
  .phase .badge { font-size:10px; text-transform:uppercase; letter-spacing:.06em; padding:2px 8px; border-radius:10px; }
  .phase.pending .badge { background:#2a323c; color:var(--muted); }
  .phase.active .badge { background:var(--accent); color:#0f1419; font-weight:600; }
  .phase.done .badge { background:var(--best); color:#0f1419; font-weight:600; }
  .phase .title { font-size:14px; font-weight:600; }
  .phase .detail { color:var(--muted); font-size:12px; margin-top:6px; }
  .phase .track { height:6px; border-radius:3px; background:#222b35; overflow:hidden; margin-top:8px; }
  .phase .fill { height:100%; background:var(--accent); border-radius:3px; }

  /* narrative log */
  .logfeed { background:var(--panel); border:1px solid var(--grid); border-radius:8px; padding:8px 0; max-height:520px; overflow:auto; }
  .logrow { display:flex; gap:12px; padding:6px 16px; border-bottom:1px solid #20272f; font-size:12px; }
  .logrow:last-child { border-bottom:none; }
  .logrow .when { color:var(--muted); white-space:nowrap; font-size:11px; }
  .logrow .ph { color:var(--div); white-space:nowrap; min-width:96px; }
  .logrow .msg { color:var(--ink); }
  .logrow.milestone .msg { color:var(--best); }
  .logrow.warn .msg { color:var(--win); }

  /* registry table */
  table { width:100%; border-collapse:collapse; font-size:12px; background:var(--panel);
          border:1px solid var(--grid); border-radius:8px; overflow:hidden; }
  th, td { text-align:left; padding:9px 12px; border-bottom:1px solid #20272f; }
  th { color:var(--muted); text-transform:uppercase; font-size:10px; letter-spacing:.06em; }
  tr:last-child td { border-bottom:none; }
  td.wr { color:var(--win); font-weight:600; }
  .pill { font-size:10px; padding:2px 7px; border-radius:10px; background:#2a323c; color:var(--muted); }
  .pill.live { background:var(--best); color:#0f1419; font-weight:600; }

  /* markdown (research) */
  .md { background:var(--panel); border:1px solid var(--grid); border-radius:8px; padding:20px 26px; max-width:980px; line-height:1.6; }
  .md h1 { font-size:20px; } .md h2 { font-size:16px; margin-top:24px; } .md h3 { font-size:14px; }
  .md code { background:#222b35; padding:1px 5px; border-radius:4px; font-size:12px; }
  .md table { margin:10px 0; } .md a { color:var(--div); }
  .md hr { border:none; border-top:1px solid var(--grid); margin:18px 0; }
  .md blockquote { border-left:3px solid var(--accent); margin:10px 0; padding:4px 14px; color:var(--muted); }

  /* live game replay viewer */
  .replay { background:var(--panel); border:1px solid var(--grid); border-radius:8px; padding:14px; margin-bottom:16px; }
  .replay h2 { font-size:13px; margin:0 0 10px; font-weight:600; }
  .replay h2 .hint { color:var(--muted); font-weight:400; font-size:11px; }
  .replay .stage { display:flex; gap:16px; flex-wrap:wrap; align-items:flex-start; }
  .replay canvas { background:#0b0f14; border:1px solid var(--grid); border-radius:6px; image-rendering:pixelated; }
  .replay .side { font-size:12px; color:var(--muted); min-width:180px; }
  .replay .side .big { font-size:15px; color:var(--ink); font-weight:600; margin-bottom:6px; }
  .replay .ctl { display:flex; align-items:center; gap:10px; margin-top:10px; flex-wrap:wrap; }
  .replay .ctl input[type=range] { flex:1; min-width:160px; accent-color:var(--accent); }
  .replay .blue { color:var(--mean); } .replay .red { color:var(--bank); }
  .replay .leg { margin-top:8px; font-size:11px; color:var(--muted); line-height:1.7; }
  /* CNN spatial heatmap (mirrors .replay dark-theme panel style) */
  .spatial { background:var(--panel); border:1px solid var(--grid); border-radius:8px; padding:14px; margin-bottom:16px; }
  .spatial h2 { font-size:13px; margin:0 0 10px; font-weight:600; }
  .spatial h2 .hint { color:var(--muted); font-weight:400; font-size:11px; }
  .spatial .stage { display:flex; gap:16px; flex-wrap:wrap; align-items:flex-start; }
  .spatial canvas { background:#0b0f14; border:1px solid var(--grid); border-radius:6px; image-rendering:pixelated;
                    transition:opacity .18s ease; }
  .spatial .side { font-size:12px; color:var(--muted); width:240px; flex:0 0 240px; }
  .spatial .side .big { font-size:15px; color:var(--ink); font-weight:600; margin-bottom:6px; }
  .spatial .ctl { display:flex; align-items:center; gap:10px; margin:0 0 10px; flex-wrap:wrap; }
  .spatial .ctlgrp { display:flex; align-items:center; gap:6px; flex-wrap:wrap; }
  .spatial .ctlgrp .lab { color:var(--muted); text-transform:uppercase; letter-spacing:.06em; font-size:11px; margin-right:2px; }
  .spatial .blue { color:var(--mean); } .spatial .red { color:var(--bank); }
  .spatial .leg { margin-top:10px; font-size:11px; color:var(--muted); line-height:1.8; }
  .spatial .leg .sw { display:inline-block; width:11px; height:11px; border-radius:2px; vertical-align:middle; margin:0 4px 2px 0; }
  .spatial .leg .grad { display:inline-block; width:120px; height:11px; border-radius:2px; vertical-align:middle; margin:0 4px 2px 0; border:1px solid var(--grid); }
  /* value bullet gauge */
  .spatial .gauge { margin:4px 0 14px; }
  .spatial .gauge .bar { position:relative; height:16px; border-radius:8px; overflow:hidden;
                         background:linear-gradient(90deg,#ff6b6b 0%,#3a414b 50%,#35d26b 100%); border:1px solid var(--grid); }
  .spatial .gauge .zero { position:absolute; left:50%; top:-2px; width:1px; height:20px; background:var(--muted); }
  .spatial .gauge .mark { position:absolute; top:-3px; width:3px; height:22px; background:var(--ink);
                          box-shadow:0 0 0 1px #0b0f14; border-radius:2px; transform:translateX(-1.5px); }
  .spatial .gauge .ends { display:flex; justify-content:space-between; color:var(--muted); font-size:10px; margin-top:3px; }
  .spatial .valbig { font-size:26px; font-weight:700; font-variant-numeric:tabular-nums;
                     font-family:ui-monospace,Menlo,Consolas,monospace; line-height:1.1; }
  .spatial .chip { display:inline-block; font-size:10px; padding:2px 7px; border-radius:10px; font-weight:600; vertical-align:middle; }
  .spatial .chip.s0 { background:rgba(90,169,255,.22); color:#9fcbff; }
  .spatial .chip.s1 { background:rgba(255,107,107,.22); color:#ff9d9d; }
  /* top-moves table */
  .spatial .moves { width:100%; border-collapse:collapse; font-size:11px; margin-top:6px; background:none; border:none; }
  .spatial .moves td { padding:4px 5px; border-bottom:1px solid #20272f; vertical-align:middle; }
  .spatial .moves tr { cursor:pointer; }
  .spatial .moves tr:hover td { background:#222b35; }
  .spatial .moves tr.top1 td { background:rgba(255,203,107,.10); }
  .spatial .moves tr.top1:hover td { background:rgba(255,203,107,.18); }
  .spatial .moves .rk { color:var(--muted); width:18px; text-align:right; font-variant-numeric:tabular-nums; }
  .spatial .moves .num { font-variant-numeric:tabular-nums; text-align:right; white-space:nowrap; color:var(--ink); }
  .spatial .moves .pbar { width:46px; }
  .spatial .moves .pbar .track { height:7px; border-radius:4px; background:#222b35; overflow:hidden; }
  .spatial .moves .pbar .fill { height:100%; border-radius:4px; background:var(--mean); }
  .spatial .ichip { display:inline-block; font-size:10px; padding:1px 6px; border-radius:9px; font-weight:600; white-space:nowrap; }
</style>
</head>
<body>
<h1>Colonizing Pirkanmaa — AI Dashboard <span style="font-size:12px;color:var(--muted)">· live</span></h1>
<p class="sub" id="sub">Colonizing Pirkanmaa</p>
<p class="status" id="status"><span class="dot">●</span> connecting…</p>

<div class="tabs" id="tabs"></div>

<!-- BUILD PROCESS -->
<div id="tab-build">
  <div class="overall" id="overall"></div>
  <h2 style="font-size:13px;margin:18px 0 8px;color:var(--muted)">VAIHEET</h2>
  <div class="phases" id="phases"></div>
  <h2 style="font-size:13px;margin:22px 0 8px;color:var(--muted)">RAKENNUSLOKI</h2>
  <div class="logfeed" id="logfeed"></div>
</div>

<!-- MODEL REGISTRY -->
<div id="tab-models" class="panel-hidden">
  <div id="registry"></div>
</div>

<!-- RESEARCH -->
<div id="tab-research" class="panel-hidden">
  <div class="tabs" id="resnav" style="border-bottom:none;margin-bottom:10px"></div>
  <div class="md" id="research"></div>
</div>

<!-- TRAINING (existing) -->
<div id="tab-training" class="panel-hidden">
  <div id="spatialPanel"></div>
  <div id="replayPanel"></div>
  <div class="summary" id="summary"></div>
  <div class="controls" id="controls">
    <div class="grp"><span class="lab">Ikkuna</span><span id="winBtns"></span></div>
    <div class="grp"><span class="lab">Tasoitus</span><span id="smoothBtns"></span></div>
    <div class="grp" id="windowInfo" style="color:var(--muted)"></div>
  </div>
  <div class="charts" id="charts"></div>
</div>

<script>
const POLL_MS = 5000;
let STATE = { dir: '', updated: null, log: [], benchmark: null, winHistory: [], replay: null, replaySelf: null,
              replayVsArmyRush: null, replayVsHqRush: null, replayVsDeviceRush: null, replayVsGarrison: null, replayVsExpert: null, replayVsMarcher: null,
              replayVsRusher: null, replayVsFortress: null, replayVsStrongArmy: null,
              spatial: null, latest: null, logMtime: null,
              buildStatus: null, buildLog: [], registry: [], research: [] };
// CNN spatial-heatmap overlay selection. 'policy' = where the net wants to act,
// 'delta' = value GAIN over doing nothing (valueAfter − root value; the key view),
// 'valueMap' = raw 1-ply value after acting on each tile. Persists across 5s polls.
let SPATIAL_MAP = 'policy';
// Selected frame index within sp.frames (early/mid/late). null = pick the MIDDLE
// frame on first load so the panel opens on something with content. A row-hover on
// a top-moves row temporarily rings that tile via SPATIAL_HOVER.
let SPATIAL_FRAME = null;
let SPATIAL_HOVER = -1;     // tile idx ringed on hover (-1 = none)
let SPATIAL_KEY = '';       // identity of the loaded spatial data (iter:frames) for fade-on-change
// View controls: window = number of most-recent gens ('all' = whole run); smooth = rolling-mean radius (0 = off).
const CTRL = { window: 'all', smooth: 0 };
// Active tab. 'build' is the landing view: "where is the whole AI build at".
let TAB = 'build';
let RES_TAB = 'research'; // which doc is shown under the Tutkimus tab
// Live game-replay viewer state (animation persists across 5s polls).
let REPLAY_KEY = '';      // identity of the currently-loaded replay (iter:seed:frames)
let REPLAY_FRAME = 0;     // current frame index
let REPLAY_PLAYING = true;
let REPLAY_FPS = 24;      // playback speed (frames/sec); 24 = 4× (6 = 1×)
let REPLAY_SRC = 'rusher';  // which match to watch — see REPLAY_SOURCES below (league first)
let REPLAY_TIMER = null;  // setInterval handle
let REPLAY_IDX = 0;       // which of the 5 FRESH games (this iteration) is shown
let REPLAY_BATCH = '';    // identity of the loaded 5-game batch (snap to game 0 on change)

// All replay sources surfaced in the viewer's toggle row. The order is the order the
// buttons render in. Each source writes replay_games fresh games per replay tick.
// Each entry: [src-id, button-label, STATE.<field>, side-panel-label, group].
//   group 'league' = the rebuilt SD3 league the curriculum trains against (PRIMARY) +
//   the AI-vs-Hard / AI-vs-AI references; group 'legacy' = the old-kind opponents the
//   curriculum no longer samples (kept for replay continuity, shown muted).
const REPLAY_SOURCES = [
  ['rusher',     'vs Rusher',          'replayVsRusher',     'Rusher',     'league'],
  ['fortress',   'vs Fortress',        'replayVsFortress',   'Fortress',   'league'],
  ['devicerush', 'vs Device Rush',     'replayVsDeviceRush', 'Device Rush','league'],
  ['strongarmy', 'vs Strong Army',     'replayVsStrongArmy', 'Strong Army','league'],
  ['hard',       'AI vs Hard CPU',     'replay',             'Hard CPU',   'league'],
  ['self',       'AI vs AI',           'replaySelf',         'AI #2',      'league'],
  ['armyrush',   'vs Army Rush (old)', 'replayVsArmyRush',   'Army Rush',  'legacy'],
  ['hqrush',     'vs HQ Rush (old)',   'replayVsHqRush',     'HQ Rush',    'legacy'],
  ['garrison',   'vs Garrison (old)',  'replayVsGarrison',   'Garrison Fortress','legacy'],
  ['expert',     'vs Econ Expert (old)','replayVsExpert',    'Econ Expert','legacy'],
  ['marcher',    'vs Marcher (old)',   'replayVsMarcher',    'Marcher',    'legacy'],
];
function replaySrcMeta(src) {
  for (const row of REPLAY_SOURCES) if (row[0] === src) return row;
  return REPLAY_SOURCES[0];
}

// The trainer records FIVE fresh games per source each ~5-iter cycle, written as a
// JSON array, so STATE.replay / STATE.replaySelf are arrays — the user browses the
// five games from the CURRENT checkpoint with "Seuraava peli". (Old single-object
// replays are tolerated by wrapping them in a 1-element array.) Scripted-opponent
// sources currently write 1-element arrays (one game per opponent per iter).
function gamesFor(src) {
  const meta = replaySrcMeta(src);
  const raw = STATE[meta[2]];
  if (Array.isArray(raw)) return raw.filter((g) => g && g.frames && g.frames.length);
  return raw && raw.frames && raw.frames.length ? [raw] : [];
}
// Identity of the current 5-game batch (iteration); when it changes, snap to game 0
// so the viewer always shows FRESH games and never stale ones.
function batchKeyOf(src) {
  const gs = gamesFor(src);
  return gs.length ? (src + ':' + gs[0].iter + ':' + gs.length) : src + ':none';
}
const TABS = [['build','Rakennusprosessi'],['models','Mallit'],['research','Tutkimus'],['training','Koulutus']];
const WINDOWS = [['all','All'],[200,'200'],[100,'100'],[50,'50'],[25,'25']];
const SMOOTHS = [[0,'off'],[3,'3'],[5,'5'],[9,'9']];

// --- tiny inline-SVG line chart -------------------------------------------
const PAD = { l: 44, r: 12, t: 10, b: 24 };
function getColor(varName) {
  return getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
}
function extent(series) {
  let lo = Infinity, hi = -Infinity;
  for (const s of series) for (const v of s.values) {
    if (v == null) continue;
    if (v < lo) lo = v; if (v > hi) hi = v;
  }
  if (!isFinite(lo)) { lo = 0; hi = 1; }
  if (lo === hi) { lo -= 1; hi += 1; }
  return [lo, hi];
}
function svgEl(tag, attrs, children) {
  const e = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (children) for (const c of children) e.appendChild(c);
  return e;
}
// Build a chart: series = [{label, color, values:[number|null], dashed?}]
// data rows each have a .gen used for the x position.
function chart(title, data, series, opts) {
  opts = opts || {};
  // Wide charts span the full grid row (~4:1). Match the viewBox aspect ratio to
  // the rendered box so preserveAspectRatio:none doesn't stretch text/lines.
  const W = opts.wide ? 960 : 380;
  const H = opts.wide ? 240 : 200;
  const card = document.createElement('div');
  card.className = 'chart' + (opts.wide ? ' wide' : '');
  const h = document.createElement('h2');
  h.textContent = title;
  if (opts.hint) { const sp = document.createElement('span'); sp.className = 'hint'; sp.textContent = ' · ' + opts.hint; h.appendChild(sp); }
  if (opts.tip) { const q = document.createElement('span'); q.className = 'tipq'; q.textContent = ' ⓘ'; q.title = opts.tip; h.appendChild(q); card.title = opts.tip; }
  card.appendChild(h);

  const hasData = series.some(s => s.values.some(v => v != null));
  if (!hasData || !data.length) {
    const e = document.createElement('div');
    e.className = 'empty';
    e.textContent = opts.emptyText || 'no data for this metric';
    card.appendChild(e);
    return card;
  }

  const gens = data.map(d => d.gen);
  const gMin = Math.min.apply(null, gens), gMax = Math.max.apply(null, gens);
  let [yLo, yHi] = opts.range || extent(series);
  const x = g => gMax === gMin ? PAD.l + (W - PAD.l - PAD.r) / 2
    : PAD.l + (g - gMin) / (gMax - gMin) * (W - PAD.l - PAD.r);
  const y = v => PAD.t + (1 - (v - yLo) / (yHi - yLo)) * (H - PAD.t - PAD.b);

  const svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H, preserveAspectRatio: 'none' });

  const TICKS = 4;
  for (let i = 0; i <= TICKS; i++) {
    const v = yLo + (yHi - yLo) * i / TICKS;
    const yy = y(v);
    svg.appendChild(svgEl('line', { class: 'gridline', x1: PAD.l, y1: yy, x2: W - PAD.r, y2: yy }));
    const t = svgEl('text', { class: 'axis', x: PAD.l - 6, y: yy + 3, 'text-anchor': 'end' });
    t.textContent = (opts.pct ? (v * 100).toFixed(0) + '%' : Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2));
    svg.appendChild(t);
  }
  const xt0 = svgEl('text', { class: 'axis', x: PAD.l, y: H - 8, 'text-anchor': 'start' });
  xt0.textContent = 'gen ' + gMin;
  const xt1 = svgEl('text', { class: 'axis', x: W - PAD.r, y: H - 8, 'text-anchor': 'end' });
  xt1.textContent = 'gen ' + gMax;
  svg.appendChild(xt0); svg.appendChild(xt1);

  if (opts.band) {
    let d = '';
    const top = [], bot = [];
    data.forEach((row, i) => {
      const c = opts.band.center[i], w = opts.band.width[i];
      if (c == null || w == null) return;
      top.push([x(row.gen), y(c + w)]);
      bot.push([x(row.gen), y(c - w)]);
    });
    if (top.length) {
      d = 'M' + top.map(p => p[0] + ',' + p[1]).join(' L');
      d += ' L' + bot.reverse().map(p => p[0] + ',' + p[1]).join(' L') + ' Z';
      svg.appendChild(svgEl('path', { d, fill: opts.band.color, 'fill-opacity': '0.12', stroke: 'none' }));
    }
  }

  if (opts.marker != null) {
    const yy = y(opts.marker);
    svg.appendChild(svgEl('line', { class: 'gridline marker', x1: PAD.l, y1: yy, x2: W - PAD.r, y2: yy, stroke: opts.markerColor || getColor('--win'), 'stroke-width': '2' }));
  }

  for (const s of series) {
    let d = '', pen = false;
    data.forEach((row, i) => {
      const v = s.values[i];
      if (v == null) { pen = false; return; }
      const px = x(row.gen), py = y(v);
      d += (pen ? ' L' : ' M') + px + ',' + py;
      pen = true;
    });
    if (d) svg.appendChild(svgEl('path', { d, fill: 'none', stroke: s.color, 'stroke-width': s.thick ? '2.5' : '2', 'stroke-linejoin': 'round', 'stroke-dasharray': s.dashed ? '4 3' : '' }));
    // dots only when sparse, so dense per-gen curves stay clean
    if (data.length <= 60 || s.dots) {
      data.forEach((row, i) => {
        const v = s.values[i];
        if (v == null) return;
        svg.appendChild(svgEl('circle', { cx: x(row.gen), cy: y(v), r: s.dots ? '2.5' : '1.8', fill: s.color }));
      });
    }
  }
  card.appendChild(svg);

  if (series.length > 1 || (series[0] && series[0].label)) {
    const leg = document.createElement('div');
    leg.className = 'legend';
    for (const s of series) {
      const span = document.createElement('span');
      const sw = document.createElement('span');
      sw.className = 'swatch'; sw.style.background = s.color;
      span.appendChild(sw);
      span.appendChild(document.createTextNode(s.label));
      leg.appendChild(span);
    }
    card.appendChild(leg);
  }
  if (opts.note) {
    const n = document.createElement('div');
    n.className = 'note';
    n.textContent = opts.note;
    card.appendChild(n);
  }
  return card;
}

// --- §10 outcome-cause breakdown (who won + HOW), intent + rounds bars ------
// The §10 win-cause taxonomy. Order = stack order; colors are stable so the
// reader learns "purple = Device" etc. across panels.
const CAUSE_META = [
  ['device', 'Strange Device', '#c792ea'],
  ['domination', 'Aluevaltaus ≥70%', '#5aa9ff'],
  ['conquest', 'Valloitus (0 ruutua)', '#ff9e64'],
  ['bankruptcy', 'Vastustaja konkurssiin', '#ff6b6b'],
  ['tiebreak', 'Ruutuenemmistö (cap)', '#64ffda'],
];
// One labelled horizontal stacked bar. Segments are scaled by count / nGames, so
// the bar LENGTH equals that side's win-rate and the two rows are directly
// comparable on the same games axis; colour composition = how those wins came.
function causeBar(rowLabel, wins, nGames, totalWins) {
  const wrap = document.createElement('div');
  wrap.style.cssText = 'margin:12px 0';
  const head = document.createElement('div');
  head.style.cssText = 'display:flex;justify-content:space-between;font-size:12px;margin-bottom:4px';
  const pct = nGames > 0 ? (100 * totalWins / nGames).toFixed(1) + '%' : '—';
  head.innerHTML = '<span style="color:var(--ink);font-weight:600">' + escapeHtml(rowLabel) + '</span>'
    + '<span style="color:var(--muted)">' + totalWins + '/' + nGames + ' · ' + pct + '</span>';
  wrap.appendChild(head);
  const track = document.createElement('div');
  track.style.cssText = 'display:flex;height:28px;border-radius:5px;overflow:hidden;background:#222b35;border:1px solid var(--grid)';
  for (const [key, label, color] of CAUSE_META) {
    const c = (wins && num(wins[key])) || 0;
    if (c <= 0) continue;
    const seg = document.createElement('div');
    const w = nGames > 0 ? (100 * c / nGames) : 0;
    seg.style.cssText = 'width:' + w + '%;background:' + color + ';min-width:3px;'
      + 'display:flex;align-items:center;justify-content:center;font-size:10px;color:#0f1419;font-weight:700;overflow:hidden';
    seg.title = label + ': ' + c + (nGames > 0 ? (' (' + (100 * c / nGames).toFixed(1) + '%)') : '');
    if (w > 6) seg.textContent = String(c);
    track.appendChild(seg);
  }
  wrap.appendChild(track);
  return wrap;
}
function sumCauses(o) { return CAUSE_META.reduce((s, m) => s + (o ? (num(o[m[0]]) || 0) : 0), 0); }
// The headline panel the user asked for: how OUR AI's wins split by cause, and
// how the HARD CPU's wins split — side by side, latest benchmark.
function causeCard(b) {
  const card = document.createElement('div');
  card.className = 'chart wide';
  const h = document.createElement('h2');
  h.innerHTML = 'Voittotavat — kuka voitti ja miten <span class="hint">· uusin benchmark</span>';
  card.appendChild(h);
  if (!b) { const e = document.createElement('div'); e.className = 'empty'; e.textContent = 'ei benchmark-dataa vielä'; card.appendChild(e); return card; }
  const n = num(b.nGames) || 0;
  card.appendChild(causeBar('Meidän AI (neural)', b.champWins, n, sumCauses(b.champWins)));
  card.appendChild(causeBar('Hard CPU (heuristiikka)', b.hardWins, n, sumCauses(b.hardWins)));
  const tie = num(b.trueTie) || 0;
  const note = document.createElement('div');
  note.className = 'note';
  note.innerHTML = 'Ratkeamattomat (aito tasapeli): <strong>' + tie + '</strong>'
    + (n > 0 ? ' (' + (100 * tie / n).toFixed(1) + '%)' : '') + ' — tavoite ≈ 0.';
  card.appendChild(note);
  const leg = document.createElement('div'); leg.className = 'legend';
  for (const [, label, color] of CAUSE_META) {
    const span = document.createElement('span');
    const sw = document.createElement('span'); sw.className = 'swatch';
    sw.style.background = color; sw.style.width = '11px'; sw.style.height = '11px'; sw.style.borderRadius = '2px';
    span.appendChild(sw); span.appendChild(document.createTextNode(label));
    leg.appendChild(span);
  }
  card.appendChild(leg);
  return card;
}
// PILLAR-6 ACTIVITY / PASSIVITY panel. A compact stat grid answering "is the net
// passively turtling or intelligently aggressive?" from fields actually present in
// log.jsonl / benchmark-history.jsonl. b = latest bench row, latest = latest log
// row (per-iter self-play), data = windowed log rows (for Pass% derivation).
function activityCard(b, latest, data) {
  const card = document.createElement('div');
  card.className = 'chart wide';
  const h = document.createElement('h2');
  h.innerHTML = 'Aktiivisuus / passiivisuus <span class="hint">· armeija · marssi · kontakti · crack</span>';
  const q = document.createElement('span'); q.className = 'tipq'; q.textContent = ' ⓘ';
  q.title = 'Onko verkko passiivinen turtle vai älykkäästi aggressiivinen? maxSoldiers = armeijan huippukoko (bench). Pass% = passiivisten siirtojen osuus uusimmasta self-play-intent-histogrammista. Contact% = self-play-pelit joissa ≥1 hyökkäys/etenevä yksikkö. MarchSoldier = armeijan marssitus kohti vihollista (uusin self-play). crackDevice/HQ = yritykset+onnistumiset (bench). Bridges/peli = sillanrakennus (liikkuvuus).';
  h.appendChild(q);
  card.appendChild(h);

  // Pass% + MarchSoldier usage from the per-iter self-play intent histogram (preferred,
  // updates every iter), else the bench intents.
  const ints = (latest && latest.iterIntents) ? latest.iterIntents : (b && b.intents) ? b.intents : null;
  let passPct = null, marchCount = null, attackCount = null, intTotal = 0;
  if (ints) {
    for (const k in ints) { const v = num(ints[k]); if (v != null && k !== 'HireWorker' && k !== 'HireExpert') intTotal += v; }
    if (intTotal > 0) {
      const p = num(ints.Pass); passPct = p != null ? p / intTotal : null;
    }
    marchCount = num(ints.MarchSoldier);
    attackCount = num(ints.Attack);
  }
  const contact = latest ? num(latest.spContactRate) : null;
  const maxSol = b ? num(b.maxSoldiersPerGame) : null;
  const bridges = b ? num(b.bridgesPerGame) : null;
  const cdA = b ? num(b.crackDeviceAttempts) : null, cdS = b ? num(b.crackDeviceSuccesses) : null;
  const chA = b ? num(b.crackHQAttempts) : null, chS = b ? num(b.crackHQSuccesses) : null;

  if (!b && !latest) {
    const e = document.createElement('div'); e.className = 'empty'; e.textContent = 'ei dataa vielä'; card.appendChild(e); return card;
  }

  // Stat tiles. Each: [label, value-string, tone] where tone colours the value.
  const fmtN = (v, d) => v == null ? '—' : (d != null ? v.toFixed(d) : String(v));
  const tiles = [
    ['Armeijan huippu', maxSol != null ? maxSol.toFixed(2) : '—', '/peli', 'maxSoldiersPerGame (bench): keskim. korkein samaan aikaan kentällä ollut sotilasmäärä. Yli 1 = oikea armeija, ei pelkkä HQ-vartija.'],
    ['Pass-osuus', passPct != null ? (passPct * 100).toFixed(1) + '%' : '—', 'self-play intent', 'Pass-intentien osuus uusimmasta self-play-histogrammista. Korkea = passiivinen ohitus. Tavoite matala.'],
    ['Kontaktiaste', contact != null ? (contact * 100).toFixed(1) + '%' : '—', 'self-play / iter', 'spContactRate: self-play-pelit joissa ≥1 hyökkäys tai etenevä yksikkö / kaikki. Matala = pelit jäätyvät ilman taistelua.'],
    ['MarchSoldier', marchCount != null ? String(marchCount) : '—', 'self-play count', 'Armeijan marssitus kohti vihollisen Devicea/HQ:ta uusimmassa self-play-iteraatiossa. 0 = ei vie armeijaa hyökkäykseen.'],
    ['Attack', attackCount != null ? String(attackCount) : '—', 'self-play count', 'Hyökkäysintentit uusimmassa self-play-iteraatiossa.'],
    ['Sillat', bridges != null ? bridges.toFixed(2) : '—', '/peli (bench)', 'bridgesPerGame: keskim. rakennetut sillat. Liikkuvuus jokien yli = pääsy hyökkäämään.'],
    ['CrackDevice', (cdA != null ? cdA : '—') + ' → ' + (cdS != null ? cdS : '—'), 'yrit. → onn. (bench)', 'crackDeviceAttempts → Successes: vihollisen Strange Devicen murtaminen.'],
    ['CrackHQ', (chA != null ? chA : '—') + ' → ' + (chS != null ? chS : '—'), 'yrit. → onn. (bench)', 'crackHQAttempts → Successes: vihollisen HQ:n murtaminen/valloitus.'],
  ];
  const grid = document.createElement('div');
  grid.style.cssText = 'display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:6px';
  for (const [lbl, val, sub, tip] of tiles) {
    const t = document.createElement('div');
    t.style.cssText = 'background:#1a212a;border:1px solid var(--grid);border-radius:7px;padding:10px 12px';
    t.title = tip;
    t.innerHTML = '<div style="font-size:11px;color:var(--muted);margin-bottom:3px">' + escapeHtml(lbl) + '</div>'
      + '<div style="font-size:19px;font-weight:700;color:var(--ink);font-variant-numeric:tabular-nums">' + escapeHtml(val) + '</div>'
      + '<div style="font-size:10px;color:var(--muted);margin-top:2px">' + escapeHtml(sub) + '</div>';
    grid.appendChild(t);
  }
  card.appendChild(grid);
  const note = document.createElement('div');
  note.className = 'note';
  note.textContent = 'Tulkinta: korkea Pass% + matala kontakti + March 0 = passiivinen turtle. Kasvava armeija + March/Attack + crack-yritykset = älykäs aggressio.';
  card.appendChild(note);
  return card;
}
// Generic labelled horizontal bar list (intent histogram, rounds-per-cause).
function barListCard(title, hint, rows, opts) {
  opts = opts || {};
  const card = document.createElement('div'); card.className = 'chart';
  const h = document.createElement('h2');
  h.innerHTML = escapeHtml(title) + (hint ? ' <span class="hint">· ' + escapeHtml(hint) + '</span>' : '');
  card.appendChild(h);
  if (!rows || !rows.length) { const e = document.createElement('div'); e.className = 'empty'; e.textContent = 'ei dataa'; card.appendChild(e); return card; }
  const max = rows.reduce((m, r) => Math.max(m, r.value), 0) || 1;
  for (const r of rows) {
    const row = document.createElement('div');
    row.style.cssText = 'display:flex;align-items:center;gap:8px;margin:3px 0;font-size:11px';
    const lab = document.createElement('span');
    lab.style.cssText = 'width:' + (opts.labelWidth || 120) + 'px;color:var(--muted);text-align:right;white-space:nowrap;overflow:hidden;text-overflow:ellipsis';
    lab.textContent = r.label;
    const barwrap = document.createElement('div');
    barwrap.style.cssText = 'flex:1;background:#222b35;border-radius:3px;height:14px;overflow:hidden';
    const bar = document.createElement('div');
    bar.style.cssText = 'height:100%;border-radius:3px;width:' + (100 * r.value / max) + '%;background:' + (r.color || 'var(--mean)');
    barwrap.appendChild(bar);
    const val = document.createElement('span');
    val.style.cssText = 'width:' + (opts.valWidth || 82) + 'px;color:var(--ink);white-space:nowrap';
    val.textContent = r.text;
    row.appendChild(lab); row.appendChild(barwrap); row.appendChild(val);
    card.appendChild(row);
  }
  if (opts.note) { const nn = document.createElement('div'); nn.className = 'note'; nn.textContent = opts.note; card.appendChild(nn); }
  return card;
}

// Intent histogram (latest benchmark) with a per-intent HISTORY sparkline that
// pops up on hover/focus. The key set is derived from the LATEST bench's intents
// object (so new keys like HireWorker/HireExpert/StackProducer render
// automatically), ordered by a preferred list with any unknown keys appended.
// The sparkline reads the already-loaded bench-history array (winHistFull) — no
// extra fetch — and maps each bench to that single intent count over gen.
//
// Categories (workforce / construction / military / other) drive both bar color
// and the small swatch on each label so the chart is readable without relying on
// color alone (every bar is also labelled).
var INTENT_ORDER = [
  // construction
  'BuildFarm', 'BuildMine', 'BuildVillage', 'BuildOutpost', 'BuildHydro',
  'BuildNuclear', 'BuildStrangeDevice', 'BuildBridge',
  // workforce
  'Expand', 'HireWorker', 'HireExpert', 'StackProducer',
  // military
  'HireSoldier', 'Attack', 'MarchSoldier', 'CrackDevice', 'CrackHQ',
  // other
  'Pass',
];
function intentCategory(key) {
  if (key === 'Expand' || key === 'HireWorker' || key === 'HireExpert' || key === 'StackProducer') return 'workforce';
  if (key === 'HireSoldier' || key === 'Attack' || key === 'MarchSoldier' || key === 'CrackDevice' || key === 'CrackHQ') return 'military';
  if (key.indexOf('Build') === 0) return 'construction';
  return 'other';
}
function intentCatColor(cat) {
  // Distinct, theme-aligned hues per category (also shown as a label swatch).
  if (cat === 'workforce') return getColor('--div');       // cyan
  if (cat === 'construction') return getColor('--best');   // green
  if (cat === 'military') return getColor('--bank');       // red
  return getColor('--muted');                              // grey (Pass/other)
}
// Build an ordered list of intent keys present in the latest hist (preferred
// order first, then any extras the data introduced, alpha-sorted for stability).
function orderedIntentKeys(hist) {
  const present = Object.keys(hist);
  const set = {}; for (const k of present) set[k] = true;
  const out = [];
  for (const k of INTENT_ORDER) if (set[k]) { out.push(k); delete set[k]; }
  const extras = Object.keys(set).sort();
  for (const k of extras) out.push(k);
  return out;
}
// Render the per-intent history sparkline (inline SVG) into box. history is an
// array of rows each carrying a .gen and an intent object under histField
// ('iterIntents' for per-iteration self-play, or 'intents' for the benchmark);
// key the intent; each point = (gen, count for key). Rows lacking that object
// are skipped so older log lines degrade gracefully.
function renderIntentSparkline(box, history, key, color, histField) {
  box.textContent = '';
  const field = histField || 'intents';
  const pts = [];
  for (const h of history) {
    const g = num(h.gen);
    const ints = h[field];
    if (g == null || !ints) continue;
    pts.push({ gen: g, v: num(ints[key]) || 0 });
  }
  const title = document.createElement('div');
  title.style.cssText = 'font-size:11px;color:var(--ink);margin-bottom:4px;font-weight:600';
  title.textContent = key + ' — historia';
  box.appendChild(title);
  if (pts.length < 2) {
    const e = document.createElement('div');
    e.style.cssText = 'font-size:11px;color:var(--muted)';
    e.textContent = pts.length === 1 ? ('vain 1 benchmark (arvo ' + pts[0].v + ')') : 'ei historiaa';
    box.appendChild(e);
    return;
  }
  const W = 240, H = 70, PADl = 30, PADr = 8, PADt = 8, PADb = 16;
  let lo = Infinity, hi = -Infinity, gmin = Infinity, gmax = -Infinity;
  for (const p of pts) {
    if (p.v < lo) lo = p.v; if (p.v > hi) hi = p.v;
    if (p.gen < gmin) gmin = p.gen; if (p.gen > gmax) gmax = p.gen;
  }
  if (lo === hi) { lo = Math.max(0, lo - 1); hi = hi + 1; }
  if (gmin === gmax) { gmin -= 1; gmax += 1; }
  const sx = (g) => PADl + (W - PADl - PADr) * (g - gmin) / (gmax - gmin);
  const sy = (v) => PADt + (H - PADt - PADb) * (1 - (v - lo) / (hi - lo));
  const svg = svgEl('svg', { width: String(W), height: String(H), viewBox: '0 0 ' + W + ' ' + H, style: 'display:block' });
  // baseline grid (min & max ticks)
  svg.appendChild(svgEl('line', { x1: PADl, y1: sy(hi), x2: W - PADr, y2: sy(hi), stroke: getColor('--grid'), 'stroke-width': '1' }));
  svg.appendChild(svgEl('line', { x1: PADl, y1: sy(lo), x2: W - PADr, y2: sy(lo), stroke: getColor('--grid'), 'stroke-width': '1' }));
  const tickMax = svgEl('text', { x: PADl - 4, y: sy(hi) + 3, 'text-anchor': 'end', fill: getColor('--muted'), 'font-size': '9' });
  tickMax.textContent = String(hi); svg.appendChild(tickMax);
  const tickMin = svgEl('text', { x: PADl - 4, y: sy(lo) + 3, 'text-anchor': 'end', fill: getColor('--muted'), 'font-size': '9' });
  tickMin.textContent = String(lo); svg.appendChild(tickMin);
  // gen axis labels
  const gA = svgEl('text', { x: PADl, y: H - 3, 'text-anchor': 'start', fill: getColor('--muted'), 'font-size': '9' });
  gA.textContent = 'g' + gmin; svg.appendChild(gA);
  const gB = svgEl('text', { x: W - PADr, y: H - 3, 'text-anchor': 'end', fill: getColor('--muted'), 'font-size': '9' });
  gB.textContent = 'g' + gmax; svg.appendChild(gB);
  // line
  let d = '';
  for (let i = 0; i < pts.length; i++) d += (i ? ' L' : 'M') + sx(pts[i].gen).toFixed(1) + ' ' + sy(pts[i].v).toFixed(1);
  svg.appendChild(svgEl('path', { d: d, fill: 'none', stroke: color || getColor('--mean'), 'stroke-width': '1.6' }));
  // dots
  for (const p of pts) svg.appendChild(svgEl('circle', { cx: sx(p.gen).toFixed(1), cy: sy(p.v).toFixed(1), r: '1.8', fill: color || getColor('--mean') }));
  // highlight + label the current (latest) value
  const cur = pts[pts.length - 1];
  svg.appendChild(svgEl('circle', { cx: sx(cur.gen).toFixed(1), cy: sy(cur.v).toFixed(1), r: '3', fill: getColor('--win') }));
  box.appendChild(svg);
  const cap = document.createElement('div');
  cap.style.cssText = 'font-size:10px;color:var(--muted);margin-top:2px';
  const unit = field === 'iterIntents' ? ' iteraatiota' : ' benchmarkia';
  cap.textContent = 'nyt: ' + cur.v + ' (g' + cur.gen + ') · min ' + lo + ' · max ' + hi + ' · ' + pts.length + unit;
  box.appendChild(cap);
}
// histSource is the object whose keys are the histogram (latest bench .intents
// OR latest log .iterIntents); history + histField drive the hover sparkline
// series (per-iteration iterIntents when available, else the bench intents).
// This lets the card update EVERY iteration from the per-gen log.
function intentHistogramCard(histSource, history, histField, hint) {
  const card = document.createElement('div'); card.className = 'chart';
  // Full-width row in the .charts grid so all bars, labels and the sparkline fit
  // (one ~380px column truncates the labels and squeezes the history panel).
  card.style.gridColumn = '1 / -1';
  const h = document.createElement('h2');
  h.innerHTML = escapeHtml('Intent-histogrammi') + (hint ? ' <span class="hint">· ' + escapeHtml(hint) + '</span>' : '');
  card.appendChild(h);
  const hist = histSource || null;
  if (!hist) { const e = document.createElement('div'); e.className = 'empty'; e.textContent = 'ei dataa'; card.appendChild(e); return card; }
  const keys = orderedIntentKeys(hist);
  const tot = keys.reduce((s, k) => s + (num(hist[k]) || 0), 0) || 1;
  let maxV = 0; for (const k of keys) maxV = Math.max(maxV, num(hist[k]) || 0);
  if (!maxV) maxV = 1;
  // layout: bars on the left, a sticky sparkline popover area on the right.
  const wrap = document.createElement('div');
  wrap.style.cssText = 'display:flex;gap:12px;align-items:flex-start';
  const bars = document.createElement('div');
  bars.style.cssText = 'flex:1;min-width:0';
  const spark = document.createElement('div');
  spark.style.cssText = 'flex:0 0 256px;background:#161c23;border:1px solid var(--grid);border-radius:6px;padding:8px;min-height:96px;align-self:stretch';
  const sparkHint = document.createElement('div');
  sparkHint.style.cssText = 'font-size:11px;color:var(--muted)';
  sparkHint.textContent = 'osoita palkkia → tämän intentin historiakäyrä';
  spark.appendChild(sparkHint);
  for (const k of keys) {
    const v = num(hist[k]) || 0;
    const cat = intentCategory(k);
    const color = intentCatColor(cat);
    const row = document.createElement('div');
    row.style.cssText = 'display:flex;align-items:center;gap:8px;margin:3px 0;font-size:11px;border-radius:3px;cursor:pointer;outline:none';
    row.tabIndex = 0;
    row.setAttribute('role', 'button');
    row.setAttribute('aria-label', k + ': ' + v + ' (' + (100 * v / tot).toFixed(1) + '%) — näytä historia');
    const lab = document.createElement('span');
    lab.style.cssText = 'width:140px;color:var(--muted);text-align:right;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;display:flex;align-items:center;justify-content:flex-end;gap:5px';
    const sw = document.createElement('span');
    sw.style.cssText = 'width:8px;height:8px;border-radius:2px;flex:0 0 8px;background:' + color;
    const lt = document.createElement('span'); lt.textContent = k;
    lab.appendChild(lt); lab.appendChild(sw);
    const barwrap = document.createElement('div');
    barwrap.style.cssText = 'flex:1;background:#222b35;border-radius:3px;height:14px;overflow:hidden';
    const bar = document.createElement('div');
    bar.style.cssText = 'height:100%;border-radius:3px;width:' + (100 * v / maxV) + '%;background:' + color;
    barwrap.appendChild(bar);
    const val = document.createElement('span');
    val.style.cssText = 'width:92px;color:var(--ink);white-space:nowrap';
    val.textContent = v + ' (' + (100 * v / tot).toFixed(1) + '%)';
    row.appendChild(lab); row.appendChild(barwrap); row.appendChild(val);
    const show = () => { row.style.background = '#222b35'; renderIntentSparkline(spark, history, k, color, histField); };
    const hide = () => { row.style.background = ''; };
    row.addEventListener('mouseenter', show);
    row.addEventListener('mouseleave', hide);
    row.addEventListener('focus', show);
    row.addEventListener('blur', hide);
    bars.appendChild(row);
  }
  wrap.appendChild(bars); wrap.appendChild(spark);
  card.appendChild(wrap);
  // legend (category → color), plus total.
  const leg = document.createElement('div');
  leg.style.cssText = 'margin-top:8px;font-size:10px;color:var(--muted);display:flex;flex-wrap:wrap;gap:10px;align-items:center';
  const cats = [['construction', 'rakennus'], ['workforce', 'työvoima'], ['military', 'sotilas'], ['other', 'muu']];
  for (const [cat, label] of cats) {
    const s = document.createElement('span');
    s.style.cssText = 'display:inline-flex;align-items:center;gap:4px';
    const sw = document.createElement('span');
    sw.style.cssText = 'width:8px;height:8px;border-radius:2px;background:' + intentCatColor(cat);
    const t = document.createElement('span'); t.textContent = label;
    s.appendChild(sw); s.appendChild(t); leg.appendChild(s);
  }
  const totSpan = document.createElement('span');
  totSpan.style.cssText = 'margin-left:auto;color:var(--muted)';
  totSpan.textContent = tot + ' päätöstä';
  leg.appendChild(totSpan);
  card.appendChild(leg);
  return card;
}

// --- helpers ---------------------------------------------------------------
function num(v) { return typeof v === 'number' && isFinite(v) ? v : null; }
function fmt(v, digits) { v = num(v); return v == null ? '—' : v.toFixed(digits == null ? 3 : digits); }
function pct(v) { v = num(v); return v == null ? '—' : (v * 100).toFixed(1) + '%'; }
function col(data, k) { return data.map(r => num(r[k])); }
function fmtDur(s) { s = num(s); if (s == null) return '—';
  s = Math.round(s); const h = Math.floor(s/3600), m = Math.floor((s%3600)/60), sec = s%60;
  return (h ? h+'h ' : '') + (h||m ? m+'m ' : '') + sec + 's'; }

// Window: keep the most-recent N gens ('all' = everything). Works on any
// array whose items carry a numeric .gen field (log rows OR win-history points).
function windowed(rows) {
  if (CTRL.window === 'all' || !rows.length) return rows;
  const n = CTRL.window;
  const maxGen = rows[rows.length - 1].gen;
  return rows.filter(r => r.gen > maxGen - n);
}
// Rolling-mean smoothing (radius r, null-aware). Returns a new value array.
function smooth(values) {
  const r = CTRL.smooth;
  if (!r) return values;
  const out = new Array(values.length).fill(null);
  for (let i = 0; i < values.length; i++) {
    let sum = 0, cnt = 0;
    for (let j = Math.max(0, i - r); j <= Math.min(values.length - 1, i + r); j++) {
      if (values[j] != null) { sum += values[j]; cnt++; }
    }
    out[i] = cnt ? sum / cnt : null;
  }
  return out;
}
function scol(data, k) { return smooth(col(data, k)); }

// --- rendering -------------------------------------------------------------
function statCard(k, v, cls, tip) {
  const c = document.createElement('div');
  c.className = 'stat' + (cls ? ' ' + cls : '');
  if (tip) c.title = tip;
  const kd = document.createElement('div'); kd.className = 'k'; kd.textContent = k;
  if (tip) { const q = document.createElement('span'); q.className = 'tipq'; q.textContent = ' ⓘ'; q.title = tip; kd.appendChild(q); }
  const vd = document.createElement('div'); vd.className = 'v'; vd.innerHTML = v;
  c.appendChild(kd); c.appendChild(vd);
  return c;
}

function renderControls() {
  const wb = document.getElementById('winBtns'); wb.textContent = '';
  for (const [val, lab] of WINDOWS) {
    const b = document.createElement('button');
    b.className = 'btn' + (CTRL.window === val ? ' on' : '');
    b.textContent = lab;
    b.onclick = () => { CTRL.window = val; renderControls(); render(); };
    wb.appendChild(b);
  }
  const sb = document.getElementById('smoothBtns'); sb.textContent = '';
  for (const [val, lab] of SMOOTHS) {
    const b = document.createElement('button');
    b.className = 'btn' + (CTRL.smooth === val ? ' on' : '');
    b.textContent = lab;
    b.onclick = () => { CTRL.smooth = val; renderControls(); render(); };
    sb.appendChild(b);
  }
}

// --- tabs + build/models/research panels -----------------------------------
function escapeHtml(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
// NOTE: this whole script lives inside the PAGE template literal, so every
// regex backslash must be DOUBLED here to survive template-literal escaping
// (\\s -> \s in the emitted page). Escaped backticks (\`) stay single.
function inlineMd(s) {
  // escape, then apply inline code / bold / links (order matters)
  s = escapeHtml(s);
  s = s.replace(/\`([^\`]+)\`/g, '<code>$1</code>');
  s = s.replace(/\\*\\*([^*]+)\\*\\*/g, '<strong>$1</strong>');
  s = s.replace(/\\[([^\\]]+)\\]\\(([^)]+)\\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  return s;
}
// Minimal, dependency-free markdown → HTML for the research writeup.
function renderMarkdown(md) {
  const lines = md.split('\\n');
  let html = '', i = 0, inCode = false, listOpen = false;
  const closeList = () => { if (listOpen) { html += '</ul>'; listOpen = false; } };
  while (i < lines.length) {
    const line = lines[i];
    if (/^\`\`\`/.test(line)) {
      if (!inCode) { closeList(); html += '<pre><code>'; inCode = true; }
      else { html += '</code></pre>'; inCode = false; }
      i++; continue;
    }
    if (inCode) { html += escapeHtml(line) + '\\n'; i++; continue; }
    // tables: a header row followed by a |---| separator
    if (/^\\s*\\|/.test(line) && i + 1 < lines.length && /^\\s*\\|[\\s:|-]+\\|?\\s*$/.test(lines[i + 1])) {
      closeList();
      const cells = (r) => r.trim().replace(/^\\||\\|$/g, '').split('|').map(c => c.trim());
      html += '<table><thead><tr>' + cells(line).map(c => '<th>' + inlineMd(c) + '</th>').join('') + '</tr></thead><tbody>';
      i += 2;
      while (i < lines.length && /^\\s*\\|/.test(lines[i])) {
        html += '<tr>' + cells(lines[i]).map(c => '<td>' + inlineMd(c) + '</td>').join('') + '</tr>';
        i++;
      }
      html += '</tbody></table>';
      continue;
    }
    const h = line.match(/^(#{1,4})\\s+(.*)$/);
    if (h) { closeList(); html += '<h' + h[1].length + '>' + inlineMd(h[2]) + '</h' + h[1].length + '>'; i++; continue; }
    if (/^---+\\s*$/.test(line)) { closeList(); html += '<hr>'; i++; continue; }
    if (/^>\\s?/.test(line)) { closeList(); html += '<blockquote>' + inlineMd(line.replace(/^>\\s?/, '')) + '</blockquote>'; i++; continue; }
    const li = line.match(/^\\s*[-*]\\s+(.*)$/);
    if (li) { if (!listOpen) { html += '<ul>'; listOpen = true; } html += '<li>' + inlineMd(li[1]) + '</li>'; i++; continue; }
    if (!line.trim()) { closeList(); i++; continue; }
    closeList(); html += '<p>' + inlineMd(line) + '</p>'; i++;
  }
  closeList();
  return html;
}

function renderTabsBar() {
  const bar = document.getElementById('tabs');
  bar.textContent = '';
  for (const [id, label] of TABS) {
    const b = document.createElement('button');
    b.className = 'tab' + (TAB === id ? ' on' : '');
    b.textContent = label;
    b.onclick = () => { TAB = id; renderTabsBar(); showTab(); };
    bar.appendChild(b);
  }
}
function showTab() {
  for (const [id] of TABS) {
    const el = document.getElementById('tab-' + id);
    if (el) el.className = (id === TAB ? '' : 'panel-hidden');
  }
}

function timeAgo(ts) {
  const ms = Date.now() - new Date(ts).getTime();
  if (!isFinite(ms)) return ts;
  const s = Math.round(ms / 1000);
  if (s < 90) return s + ' s sitten';
  const m = Math.round(s / 60); if (m < 90) return m + ' min sitten';
  const h = Math.round(m / 60); if (h < 36) return h + ' h sitten';
  return Math.round(h / 24) + ' pv sitten';
}

function renderBuild() {
  const bs = STATE.buildStatus;
  const overall = document.getElementById('overall');
  const phasesEl = document.getElementById('phases');
  const feed = document.getElementById('logfeed');
  const STATUS_FI = { pending: 'odottaa', active: 'käynnissä', done: 'valmis' };
  if (!bs || !bs.phases) {
    overall.innerHTML = '<div class="empty">ei build-status.json-tiedostoa vielä</div>';
    phasesEl.textContent = ''; feed.textContent = '';
    return;
  }
  const phases = bs.phases;
  const done = phases.filter(p => p.status === 'done').length;
  // overall progress: done phases + fractional credit for the active phase
  let frac = done;
  const active = phases.find(p => p.status === 'active');
  if (active) frac += (typeof active.progress === 'number' ? active.progress : 0.5);
  const ovPct = Math.round(100 * frac / phases.length);
  overall.innerHTML = '<div style="display:flex;justify-content:space-between;align-items:baseline">'
    + '<span style="font-size:15px;font-weight:600">' + escapeHtml(bs.title || 'AI build') + '</span>'
    + '<span style="color:var(--muted);font-size:12px">' + done + '/' + phases.length + ' vaihetta · ' + ovPct + '%'
    + (active ? ' · nyt: ' + escapeHtml(active.title) : '') + '</span></div>'
    + '<div class="track"><div class="fill" style="width:' + ovPct + '%"></div></div>';

  phasesEl.textContent = '';
  for (const p of phases) {
    const card = document.createElement('div');
    card.className = 'phase ' + (p.status || 'pending');
    let inner = '<div class="top"><span class="badge">' + (STATUS_FI[p.status] || p.status || 'odottaa') + '</span>'
      + '<span class="title">' + escapeHtml(p.title || p.id) + '</span></div>';
    if (p.detail) inner += '<div class="detail">' + escapeHtml(p.detail) + '</div>';
    if (p.status === 'active' && typeof p.progress === 'number') {
      inner += '<div class="track"><div class="fill" style="width:' + Math.round(p.progress * 100) + '%"></div></div>';
    }
    card.innerHTML = inner;
    phasesEl.appendChild(card);
  }

  const log = (STATE.buildLog || []).slice().reverse(); // newest first
  feed.textContent = '';
  if (!log.length) { feed.innerHTML = '<div class="empty">ei lokimerkintöjä vielä</div>'; return; }
  for (const e of log) {
    const row = document.createElement('div');
    row.className = 'logrow ' + (e.level || 'info');
    row.innerHTML = '<span class="when" title="' + escapeHtml(e.ts) + '">' + timeAgo(e.ts) + '</span>'
      + '<span class="ph">' + escapeHtml(e.phase || '') + '</span>'
      + '<span class="msg">' + escapeHtml(e.msg || '') + '</span>';
    feed.appendChild(row);
  }
}

function renderModels() {
  const root = document.getElementById('registry');
  const reg = STATE.registry || [];
  if (!reg.length) { root.innerHTML = '<div class="empty">ei malleja rekisterissä vielä</div>'; return; }
  // newest first
  const rows = reg.slice().reverse();
  let html = '<table><thead><tr><th>Malli</th><th>Tyyppi</th><th>Arch</th><th>Params</th>'
    + '<th>vs hard</th><th>Ruudut</th><th>Leaf</th><th>Luotu</th><th>Muistiinpanot</th></tr></thead><tbody>';
  for (const m of rows) {
    const wr = (m.winRateVsHard != null) ? (m.winRateVsHard * 100).toFixed(1) + '%' : '—';
    const tf = (m.tileFrac != null) ? (m.tileFrac * 100).toFixed(0) + '%' : '—';
    html += '<tr><td>' + escapeHtml(m.name)
      + (m.deployed ? ' <span class="pill live">käytössä</span>' : '')
      + '<div style="color:var(--muted);font-size:10px">' + escapeHtml(m.id) + '</div></td>'
      + '<td><span class="pill">' + escapeHtml(m.kind || '?') + '</span></td>'
      + '<td>' + escapeHtml((m.arch || []).join('×') || '—') + '</td>'
      + '<td>' + (m.params != null ? m.params : '—') + '</td>'
      + '<td class="wr">' + wr + '</td>'
      + '<td>' + tf + '</td>'
      + '<td>' + escapeHtml(m.leaf || '—') + '</td>'
      + '<td>' + timeAgo(m.created) + '</td>'
      + '<td style="color:var(--muted);max-width:320px">' + escapeHtml(m.notes || '') + '</td></tr>';
  }
  html += '</tbody></table>';
  root.innerHTML = html;
}

let RESEARCH_CACHE = '';
function renderResearch() {
  const docs = STATE.research || [];
  const nav = document.getElementById('resnav');
  const root = document.getElementById('research');
  if (!docs.length) { nav.textContent = ''; root.innerHTML = '<div class="empty">ei dokumentteja</div>'; return; }
  if (!docs.find(d => d.id === RES_TAB)) RES_TAB = docs[0].id;
  // sub-nav across the available docs
  nav.textContent = '';
  for (const d of docs) {
    const b = document.createElement('button');
    b.className = 'tab' + (RES_TAB === d.id ? ' on' : '');
    b.textContent = d.title;
    b.onclick = () => { RES_TAB = d.id; renderResearch(); };
    nav.appendChild(b);
  }
  const doc = docs.find(d => d.id === RES_TAB);
  const key = RES_TAB + ':' + (doc.md ? doc.md.length : 0);
  if (RESEARCH_CACHE === key) return; // skip markdown re-parse when unchanged
  RESEARCH_CACHE = key;
  root.innerHTML = renderMarkdown(doc.md);
}

// --- CNN spatial heatmap ("what the net sees / where it wants to act") -------
// Renders <dir>/spatial.json: { iter, width, height, frames:[ {label,round,curSeat,
// value,terrain,owner,building,soldiers,myHq,enemyHq,policy,valueMap,topMoves}, ... ] }.
// Each frame is ROW-MAJOR (index = y*W + x). Three overlays: policy (where the net
// wants to act), Δ-value (valueAfter − root value; the KEY view) and raw valueMap.
// Backward-compatible: a legacy single-frame object (no .frames) is wrapped as one
// frame; hidden entirely when STATE.spatial is null (older runs / non-CNN arcs).
const STCOLOR = { r: '#14506a', m: '#454b54', f: '#1d3a28', a: '#15301e', g: '#23301a' };
// Building glyphs/colors for the spatial board — reuse the replay codes (defined
// below as BGLYPH/BCOLOR) but the spatial data uses 'HQ' (two chars) so map here.
const SP_BGLYPH = { F: 'F', M: 'M', V: 'V', O: 'O', H: 'H', N: 'N', B: 'B', D: '◆', HQ: '★', K: 'K' };
const SP_BCOLOR = { D: '#c792ea', HQ: '#ffcb6b' };
// Intent → chip color (Attack reddish, Build* greenish, Expand bluish, Pass grey,
// BuildStrangeDevice purple, Hire* amber).
function intentChip(intent) {
  const s = String(intent || '');
  let bg = 'rgba(139,151,163,.22)', fg = '#c2cbd4'; // Pass / default grey
  if (/StrangeDevice/i.test(s)) { bg = 'rgba(199,146,234,.24)'; fg = '#d8b6f2'; }
  else if (/^Attack/i.test(s)) { bg = 'rgba(255,107,107,.22)'; fg = '#ff9d9d'; }
  else if (/^Build/i.test(s)) { bg = 'rgba(53,210,107,.20)'; fg = '#7fe3a3'; }
  else if (/^Expand/i.test(s)) { bg = 'rgba(90,169,255,.22)'; fg = '#9fcbff'; }
  else if (/Hire|Soldier/i.test(s)) { bg = 'rgba(255,203,107,.20)'; fg = '#ffd98a'; }
  return '<span class="ichip" style="background:' + bg + ';color:' + fg + '">' + escapeHtml(s) + '</span>';
}

// Warm sequential colormap for policy (faint amber → saturated red), t in [0,1].
function warmRGBA(t) {
  t = Math.max(0, Math.min(1, t));
  // yellow (255,203,107) → red (255,107,107); alpha ramps so weak tiles stay clear.
  const g = Math.round(203 - 96 * t), b = 107;
  const a = 0.10 + 0.80 * t;
  return 'rgba(255,' + g + ',' + b + ',' + a.toFixed(3) + ')';
}
// Diverging colormap: red(neg) → grey(0) → green(pos), s already normalised to [-1,1].
function divRGBA(s) {
  s = Math.max(-1, Math.min(1, s));
  const a = 0.18 + 0.70 * Math.min(1, Math.abs(s));
  if (s >= 0) { // grey → green
    const t = s, r = Math.round(139 - 86 * t), g = Math.round(151 + 59 * t), b = Math.round(163 - 56 * t);
    return 'rgba(' + r + ',' + g + ',' + b + ',' + a.toFixed(3) + ')';
  }
  const t = -s, r = Math.round(139 + 116 * t), g = Math.round(151 - 44 * t), b = Math.round(163 - 56 * t);
  return 'rgba(' + r + ',' + g + ',' + b + ',' + a.toFixed(3) + ')';
}

// Normalise the legacy single-frame shape into the new { width,height,frames } form.
function spatialFrames(sp) {
  if (!sp) return null;
  if (Array.isArray(sp.frames) && sp.frames.length) return sp;
  if (sp.width) return { iter: sp.iter, width: sp.width, height: sp.height, frames: [sp] };
  return null;
}
// The frame the user is viewing (default = middle frame so it opens on content).
function activeSpatialFrame(spn) {
  const n = spn.frames.length;
  let idx = SPATIAL_FRAME == null ? Math.floor((n - 1) / 2) : SPATIAL_FRAME;
  idx = Math.max(0, Math.min(n - 1, idx));
  return { idx: idx, frame: spn.frames[idx] };
}

// Build the per-tile heat value array + its scale meta for the active overlay.
function spatialHeat(f) {
  const n = f.terrain.length;
  const pol = f.policy || [], vm = f.valueMap || [], root = Number(f.value) || 0;
  if (SPATIAL_MAP === 'policy') {
    let mx = 0; for (let i = 0; i < pol.length; i++) if (pol[i] > mx) mx = pol[i];
    return { kind: 'seq', vals: pol, max: mx || 1 };
  }
  // delta or valueMap: both diverging, symmetric around 0 over their own non-nulls.
  const vals = new Array(n).fill(null);
  let absMax = 1e-6;
  for (let i = 0; i < n; i++) {
    const v = vm[i];
    if (v == null) continue;
    const x = SPATIAL_MAP === 'delta' ? (v - root) : v;
    vals[i] = x;
    if (Math.abs(x) > absMax) absMax = Math.abs(x);
  }
  return { kind: 'div', vals: vals, max: absMax };
}

function drawSpatial(canvas, spn) {
  const af = activeSpatialFrame(spn);
  const f = af.frame;
  const W = spn.width, H = spn.height;
  const terr = f.terrain || '';
  const own = f.owner || [], bld = f.building || [], sol = f.soldiers || [];
  const cell = Math.max(24, Math.min(40, Math.floor(540 / W)));
  if (canvas.width !== cell * W) { canvas.width = cell * W; canvas.height = cell * H; }
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const heat = spatialHeat(f);
  const chosen = (f.topMoves && f.topMoves.length && f.topMoves[0].idx >= 0) ? f.topMoves[0].idx : -1;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = y * W + x; // ROW-MAJOR
    const px = x * cell, py = y * cell;
    // 1. terrain base
    ctx.fillStyle = STCOLOR[terr[i]] || '#161c24';
    ctx.fillRect(px, py, cell, cell);
    // 2. heat overlay (selected layer only)
    const hv = heat.vals[i];
    if (hv != null) {
      if (heat.kind === 'seq') { if (hv > 0) { ctx.fillStyle = warmRGBA(hv / heat.max); ctx.fillRect(px, py, cell, cell); } }
      else { ctx.fillStyle = divRGBA(hv / heat.max); ctx.fillRect(px, py, cell, cell); }
    }
    // 3. ownership: translucent tint + 2px inset border in the seat colour (0=blue, 1=red)
    const o = own[i];
    if (o === 0 || o === 1) {
      ctx.fillStyle = o === 0 ? 'rgba(90,169,255,0.16)' : 'rgba(255,107,107,0.16)';
      ctx.fillRect(px, py, cell, cell);
      ctx.strokeStyle = o === 0 ? '#5aa9ff' : '#ff6b6b'; ctx.lineWidth = 2;
      ctx.strokeRect(px + 1.5, py + 1.5, cell - 3, cell - 3);
    }
    // grid hairline
    ctx.strokeStyle = '#0b0f14'; ctx.lineWidth = 1; ctx.strokeRect(px + 0.5, py + 0.5, cell - 1, cell - 1);
    // 4. building glyph (with dark outline so it reads over heat)
    const b = bld[i];
    if (b) {
      const isHq = (i === f.myHq || i === f.enemyHq);
      if (!isHq) { // HQ ring+star drawn in step 6
        ctx.font = '700 ' + Math.floor(cell * 0.5) + 'px ui-monospace,monospace';
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.lineWidth = 3; ctx.strokeStyle = 'rgba(11,15,20,0.85)';
        ctx.strokeText(SP_BGLYPH[b] || b, px + cell / 2, py + cell / 2 + 1);
        ctx.fillStyle = SP_BCOLOR[b] || '#e6edf3';
        ctx.fillText(SP_BGLYPH[b] || b, px + cell / 2, py + cell / 2 + 1);
      }
    }
    // 5. soldiers — small pill badge with |count|, blue (mine,+) or red (enemy,−)
    const s = sol[i];
    if (s) {
      const cnt = Math.abs(s), bw = Math.max(13, cell * 0.42), bh = Math.floor(cell * 0.34);
      const bx = px + cell - bw - 1.5, by = py + cell - bh - 1.5;
      ctx.fillStyle = s > 0 ? '#5aa9ff' : '#ff6b6b';
      ctx.beginPath();
      if (ctx.roundRect) { ctx.roundRect(bx, by, bw, bh, 3); } else { ctx.rect(bx, by, bw, bh); }
      ctx.fill();
      ctx.fillStyle = '#0b0f14';
      ctx.font = '700 ' + Math.floor(bh * 0.85) + 'px ui-monospace,monospace';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(String(cnt), bx + bw / 2, by + bh / 2 + 0.5);
    }
    // 6. HQ — bright ring + ★ for my / enemy HQ
    if (i === f.myHq || i === f.enemyHq) {
      const mine = i === f.myHq;
      ctx.strokeStyle = mine ? '#5aa9ff' : '#ff6b6b'; ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.arc(px + cell / 2, py + cell / 2, cell * 0.34, 0, Math.PI * 2); ctx.stroke();
      ctx.font = '700 ' + Math.floor(cell * 0.46) + 'px ui-monospace,monospace';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.lineWidth = 3; ctx.strokeStyle = 'rgba(11,15,20,0.85)';
      ctx.strokeText('★', px + cell / 2, py + cell / 2 + 1);
      ctx.fillStyle = '#ffcb6b'; ctx.fillText('★', px + cell / 2, py + cell / 2 + 1);
    }
    // 7. chosen move — thick accent-yellow ring so the eye lands on what the net picked
    if (i === chosen) {
      ctx.strokeStyle = '#ffcb6b'; ctx.lineWidth = 3;
      ctx.strokeRect(px + 2.5, py + 2.5, cell - 5, cell - 5);
    }
    // hover highlight from the top-moves table
    if (i === SPATIAL_HOVER && i !== chosen) {
      ctx.strokeStyle = '#e6edf3'; ctx.lineWidth = 2.5;
      ctx.strokeRect(px + 2.5, py + 2.5, cell - 5, cell - 5);
    }
  }
}

// Frame selector + overlay selector (two segmented groups).
function spatialControlsHtml(spn) {
  const af = activeSpatialFrame(spn);
  let fb = '';
  spn.frames.forEach((f, i) => {
    const lab = (f.label ? f.label.charAt(0).toUpperCase() + f.label.slice(1) : 'F' + i) + ' · r' + (f.round != null ? f.round : '?');
    fb += '<button class="btn sframe' + (i === af.idx ? ' on' : '') + '" data-frame="' + i + '">' + escapeHtml(lab) + '</button>';
  });
  const OV = [['policy', 'Policy'], ['delta', 'Δ-arvo'], ['valueMap', 'Arvo jälkeen']];
  let ob = '';
  for (const [val, lab] of OV) ob += '<button class="btn smap' + (val === SPATIAL_MAP ? ' on' : '') + '" data-map="' + val + '">' + lab + '</button>';
  return '<div class="ctl">'
    + '<div class="ctlgrp"><span class="lab">Vaihe</span>' + fb + '</div>'
    + '<div class="ctlgrp"><span class="lab">Kerros</span>' + ob + '</div>'
    + '</div>';
}

// Side panel: value gauge + ranked top-moves table.
function spatialSideHtml(spn) {
  const af = activeSpatialFrame(spn);
  const f = af.frame, W = spn.width;
  const v = Number(f.value) || 0, vs = (v >= 0 ? '+' : '') + v.toFixed(3);
  const markPct = (50 + 50 * Math.max(-1, Math.min(1, v))).toFixed(1);
  const seatChip = f.curSeat === 0
    ? '<span class="chip s0">seat 0 · sininen</span>'
    : '<span class="chip s1">seat 1 · punainen</span>';
  // value gauge
  let html = '<div class="big">Voittoarvio</div>'
    + '<div class="valbig" style="color:' + (v >= 0 ? '#7fe3a3' : '#ff9d9d') + '">' + vs + '</div>'
    + '<div style="margin:2px 0 4px">vuorossa: ' + seatChip + '</div>'
    + '<div class="gauge"><div class="bar"><div class="zero"></div><div class="mark" style="left:' + markPct + '%"></div></div>'
    + '<div class="ends"><span>−1 häviö</span><span>0</span><span>+1 voitto</span></div></div>';
  // top moves
  html += '<div class="big" style="font-size:13px;margin:8px 0 2px">Parhaat siirrot</div>';
  const tm = f.topMoves || [];
  if (!tm.length) {
    html += '<div style="color:var(--muted)">ei siirtoja</div>';
  } else {
    let mx = 0; for (const m of tm) if (Number(m.prob) > mx) mx = Number(m.prob);
    if (mx <= 0) mx = 1;
    html += '<table class="moves">';
    tm.forEach((m, i) => {
      const idx = (typeof m.idx === 'number') ? m.idx : -1;
      const coord = idx >= 0 ? '(' + (idx % W) + ',' + Math.floor(idx / W) + ')' : '—';
      const va = (m.valueAfter == null) ? '—' : ((m.valueAfter >= 0 ? '+' : '') + Number(m.valueAfter).toFixed(3));
      const pp = (100 * Number(m.prob)).toFixed(1) + '%';
      const bw = (100 * Number(m.prob) / mx).toFixed(0);
      html += '<tr class="' + (i === 0 ? 'top1' : '') + '" data-tile="' + idx + '">'
        + '<td class="rk">' + (i + 1) + '</td>'
        + '<td>' + intentChip(m.intent) + '</td>'
        + '<td class="num" style="color:var(--muted)">' + coord + '</td>'
        + '<td class="pbar"><div class="track"><div class="fill" style="width:' + bw + '%"></div></div></td>'
        + '<td class="num" style="color:var(--muted);width:38px">' + pp + '</td>'
        + '<td class="num">' + va + '</td>'
        + '</tr>';
    });
    html += '</table>';
  }
  return html;
}

// Legend incl. the heat-scale gradient bar with NUMERIC ticks for the current overlay.
function spatialLegendHtml(spn) {
  const af = activeSpatialFrame(spn);
  const f = af.frame;
  const heat = spatialHeat(f);
  const terrLeg = '<span class="sw" style="background:#23301a"></span>nurmi'
    + ' <span class="sw" style="background:#1d3a28"></span>metsä'
    + ' <span class="sw" style="background:#15301e"></span>rehevä metsä'
    + ' <span class="sw" style="background:#454b54"></span>vuori'
    + ' <span class="sw" style="background:#14506a"></span>joki';
  let scale;
  if (SPATIAL_MAP === 'policy') {
    scale = '<span class="grad" style="background:linear-gradient(90deg,rgba(255,203,107,.10),rgba(255,107,107,.95))"></span>'
      + '<span style="font-variant-numeric:tabular-nums">0 — ' + heat.max.toFixed(3) + '</span> (policy-todennäköisyys)';
  } else {
    const m = heat.max;
    scale = '<span class="grad" style="background:linear-gradient(90deg,#d35d5d,#3a414b 50%,#3fb56b)"></span>'
      + '<span style="font-variant-numeric:tabular-nums">−' + m.toFixed(3) + ' · 0 · +' + m.toFixed(3) + '</span>'
      + (SPATIAL_MAP === 'delta' ? ' (arvon muutos vs. Pass)' : ' (arvo siirron jälkeen)');
  }
  const bcodes = 'F=farm M=mine V=village O=outpost H=hydro N=nuclear B=bridge <span style="color:#c792ea">◆=device</span> <span style="color:#ffcb6b">★=HQ</span>';
  return '<div class="leg">'
    + 'Maasto: ' + terrLeg + '<br>'
    + 'Omistus: <span class="sw" style="background:#5aa9ff"></span>seat 0 &nbsp; <span class="sw" style="background:#ff6b6b"></span>seat 1 &nbsp; (★ = HQ)<br>'
    + 'Rakennukset: ' + bcodes + ' &nbsp;·&nbsp; sotilaat: <span class="sw" style="background:#5aa9ff"></span>omat / <span class="sw" style="background:#ff6b6b"></span>vihollinen<br>'
    + 'Lämpöasteikko: ' + scale + ' &nbsp;·&nbsp; <span style="color:#ffcb6b">keltainen kehys</span> = verkon valitsema siirto'
    + '</div>';
}

// Wire the frame + overlay buttons and the move-row hover → tile-ring.
function wireSpatial(spn) {
  for (const b of document.querySelectorAll('.sframe')) {
    b.onclick = () => {
      const i = Number(b.getAttribute('data-frame'));
      if (i !== activeSpatialFrame(spn).idx) { SPATIAL_FRAME = i; SPATIAL_HOVER = -1; renderSpatial(true); }
    };
  }
  for (const b of document.querySelectorAll('.smap')) {
    b.onclick = () => {
      const m = b.getAttribute('data-map');
      if (m !== SPATIAL_MAP) { SPATIAL_MAP = m; renderSpatial(true); }
    };
  }
  const canvas = document.getElementById('spatialCanvas');
  for (const tr of document.querySelectorAll('.spatial .moves tr')) {
    tr.onmouseenter = () => { const t = Number(tr.getAttribute('data-tile')); if (t >= 0 && canvas) { SPATIAL_HOVER = t; drawSpatial(canvas, spn); } };
    tr.onmouseleave = () => { if (SPATIAL_HOVER !== -1 && canvas) { SPATIAL_HOVER = -1; drawSpatial(canvas, spn); } };
  }
}

function renderSpatial(animate) {
  const panel = document.getElementById('spatialPanel');
  if (!panel) return;
  const spn = spatialFrames(STATE.spatial);
  if (!spn || !spn.width || !spn.height || !spn.frames.length) { panel.innerHTML = ''; return; } // hidden when absent
  // A new data batch (different iter / frame count) → reset frame to the middle.
  const key = spn.iter + ':' + spn.frames.length;
  if (key !== SPATIAL_KEY) { SPATIAL_KEY = key; SPATIAL_FRAME = null; }
  const af = activeSpatialFrame(spn);
  const reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const title = '<h2>CNN — mitä verkko näkee <span class="hint">· iteraatio ' + spn.iter
    + ' · ' + escapeHtml(af.frame.label || '') + ' (kierros ' + (af.frame.round != null ? af.frame.round : '?') + ')</span></h2>';
  panel.innerHTML = '<div class="spatial">' + title + spatialControlsHtml(spn)
    + '<div class="stage">'
    + '<canvas id="spatialCanvas"></canvas>'
    + '<div class="side" id="spatialSide">' + spatialSideHtml(spn) + '</div>'
    + '</div>'
    + spatialLegendHtml(spn)
    + '</div>';
  wireSpatial(spn);
  const canvas = document.getElementById('spatialCanvas');
  drawSpatial(canvas, spn);
  // short fade on frame/overlay change (respect reduced-motion)
  if (animate && !reduce && canvas) {
    canvas.style.opacity = '0';
    requestAnimationFrame(() => { canvas.style.opacity = '1'; });
  }
}

// --- live game-replay viewer ----------------------------------------------
// Building glyphs + per-glyph color overrides (Device/HQ stand out).
const BGLYPH = { F: 'F', M: 'M', V: 'V', O: 'O', H: 'H', N: 'N', B: 'B', D: '◆', Q: '★', K: 'K' };
const BCOLOR = { D: '#c792ea', Q: '#ffcb6b' };
// Terrain base colors — river is unmistakably water-blue so it reads under owner tint.
const TCOLOR = { r: '#14506a', m: '#454b54', f: '#1d3a28', a: '#15301e', g: '#23301a' };

function drawReplayFrame(canvas, r, fi) {
  const f = r.frames[fi];
  if (!f) return;
  const W = r.width, H = r.height;
  const terr = r.terrain || '';
  const cell = Math.max(10, Math.min(34, Math.floor(520 / W)));
  if (canvas.width !== cell * W) { canvas.width = cell * W; canvas.height = cell * H; }
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const own = f.own, bld = f.bld, sol = f.sol;
  for (let i = 0; i < own.length; i++) {
    const x = Math.floor(i / H), y = i % H; // index = x*height + y (column-major)
    const px = x * cell, py = y * cell;
    // 1. terrain base (rivers/mountains/forest stay visible all game).
    ctx.fillStyle = TCOLOR[terr[i]] || '#161c24';
    ctx.fillRect(px, py, cell, cell);
    // 2. ownership: a translucent tint + a 2px inset border in the owner's color.
    const o = own[i];
    if (o === '1' || o === '2') {
      ctx.fillStyle = o === '1' ? 'rgba(90,169,255,0.30)' : 'rgba(255,107,107,0.30)';
      ctx.fillRect(px, py, cell, cell);
      ctx.strokeStyle = o === '1' ? '#5aa9ff' : '#ff6b6b'; ctx.lineWidth = 2;
      ctx.strokeRect(px + 1.5, py + 1.5, cell - 3, cell - 3);
    }
    ctx.strokeStyle = '#0b0f14'; ctx.lineWidth = 1; ctx.strokeRect(px + 0.5, py + 0.5, cell - 1, cell - 1);
    const b = bld[i];
    if (b && b !== '.') {
      ctx.fillStyle = BCOLOR[b] || '#e6edf3';
      ctx.font = '700 ' + Math.floor(cell * 0.56) + 'px ui-monospace,monospace';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(BGLYPH[b] || b, px + cell / 2, py + cell / 2 + 1);
    }
    const s = sol[i];
    if (s && s !== '.' && s !== '0') {
      ctx.fillStyle = '#ffcb6b';
      ctx.font = '700 ' + Math.floor(cell * 0.4) + 'px ui-monospace,monospace';
      ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
      ctx.fillText(s, px + cell - 1, py + cell - 0);
    }
  }
}

function replaySideHtml(r, fi) {
  const f = r.frames[fi];
  // Side label resolution: prefer the replay's own \`mode\` tag (set by the trainer per
  // source) over the dashboard's current source toggle, so a stale \`STATE.replayVsX\`
  // file labels correctly even if the toggle changed before the next poll.
  const meta = replaySrcMeta(REPLAY_SRC);
  const self = r.mode === 'self';
  const blueLbl = self ? 'AI #1' : 'Meidän AI';
  const redLbl = self ? 'AI #2' : (r.mode === 'hard' ? 'Hard CPU' : meta[3]);
  const blue = blueLbl + ' (sininen)';
  const red = redLbl + ' (punainen)';
  const turn = f.p === 0 ? '<span class="blue">' + blue + '</span>'
    : f.p === 1 ? '<span class="red">' + red + '</span>' : 'asetelma';
  let res;
  if (r.result.winnerSeat === 0) res = '<span class="blue">' + blue + '</span> voitti — ' + escapeHtml(r.result.cause);
  else if (r.result.winnerSeat === 1) res = '<span class="red">' + red + '</span> voitti — ' + escapeHtml(r.result.cause);
  else res = 'ratkeamaton';
  const modeStr = self ? ' · self-play' : (r.mode === 'hard' ? ' · vs hard' : (' · vs ' + redLbl));
  return '<div class="big">Iteraatio ' + r.iter + modeStr + '</div>'
    + 'Kierros <b style="color:var(--ink)">' + f.r + '</b> · vuoro: ' + turn + '<br>'
    + 'Ruutu ' + (fi + 1) + '/' + r.frames.length + '<br><br>'
    + '<b style="color:var(--ink)">Lopputulos:</b><br>' + res + ' (' + r.result.rounds + ' kierrosta)';
}

// Source toggle — one button per entry in REPLAY_SOURCES (hard, self, + 5 scripted
// opponents). Shown even when a source is empty so the user can see which replays
// are ABOUT to appear after the next replay-tick.
function replayToggleHtml() {
  let html = '<div class="ctl" style="margin:0 0 10px;flex-wrap:wrap">'
    + '<span style="color:var(--muted);text-transform:uppercase;letter-spacing:.06em;font-size:11px">Liiga</span>';
  let inLegacy = false;
  for (const row of REPLAY_SOURCES) {
    const legacy = row[4] === 'legacy';
    // Insert a divider + label before the first legacy button so the SD3-league
    // opponents (the curriculum's actual training set) read as the primary group.
    if (legacy && !inLegacy) {
      inLegacy = true;
      html += '<span style="width:1px;height:18px;background:var(--grid);margin:0 4px"></span>'
        + '<span style="color:var(--muted);text-transform:uppercase;letter-spacing:.06em;font-size:11px">Vanhat</span>';
    }
    const cls = 'btn rtoggle' + (legacy ? ' rlegacy' : '');
    const style = legacy ? ' style="opacity:.7"' : '';
    html += '<button class="' + cls + '" data-src="' + row[0] + '"' + style + '>' + escapeHtml(row[1]) + '</button>';
  }
  return html + '</div>';
}
function wireReplayToggle() {
  for (const b of document.querySelectorAll('.rtoggle')) {
    b.classList.toggle('on', b.getAttribute('data-src') === REPLAY_SRC);
    b.onclick = () => {
      const src = b.getAttribute('data-src');
      if (src !== REPLAY_SRC) { REPLAY_SRC = src; REPLAY_IDX = 0; REPLAY_KEY = ''; renderReplay(); }
    };
  }
}

// The game currently shown: the REPLAY_IDX-th of the five fresh games for the
// active source (clamped if a future batch has fewer).
function activeReplay() {
  const gs = gamesFor(REPLAY_SRC);
  if (!gs.length) return null;
  return gs[Math.min(REPLAY_IDX, gs.length - 1)];
}

function ensureReplayTimer() {
  if (REPLAY_TIMER) return;
  REPLAY_TIMER = setInterval(() => {
    const r = activeReplay();
    if (!r || !r.frames || !r.frames.length) return;
    const canvas = document.getElementById('replayCanvas');
    if (!canvas) return;
    if (REPLAY_PLAYING) REPLAY_FRAME = (REPLAY_FRAME + 1) % r.frames.length;
    drawReplayFrame(canvas, r, REPLAY_FRAME);
    const sb = document.getElementById('replayScrub'); if (sb && document.activeElement !== sb) sb.value = String(REPLAY_FRAME);
    const side = document.getElementById('replaySide'); if (side) side.innerHTML = replaySideHtml(r, REPLAY_FRAME);
  }, Math.round(1000 / REPLAY_FPS));
}
function restartReplayTimer() { if (REPLAY_TIMER) { clearInterval(REPLAY_TIMER); REPLAY_TIMER = null; } ensureReplayTimer(); }

function renderReplay() {
  const panel = document.getElementById('replayPanel');
  if (!panel) return;
  const self = REPLAY_SRC === 'self';
  const meta = replaySrcMeta(REPLAY_SRC);
  const isScripted = REPLAY_SRC !== 'hard' && REPLAY_SRC !== 'self';
  // A new 5-game batch (next iteration) arrived → snap to the first FRESH game so
  // the viewer is never left on a stale one.
  const batch = batchKeyOf(REPLAY_SRC);
  if (batch !== REPLAY_BATCH) { REPLAY_BATCH = batch; REPLAY_IDX = 0; }
  const r = activeReplay();
  const nGames = gamesFor(REPLAY_SRC).length;
  // Every source (vs-hard / self-play / each scripted opponent) writes replay_games
  // fresh games per replay tick (default 5). The fallback "5" is only used as a hint
  // while the file is still missing.
  const expectedN = 5;
  const title = '<h2>Live-peli — ' + (self ? 'AI vs AI (self-play)' : (isScripted ? ('AI vs ' + meta[3]) : 'AI vs Hard CPU'))
    + ' <span class="hint">· ' + (nGames || expectedN) + ' tuoretta peliä / iteraatio · selaa “Seuraava peli”</span></h2>';
  // Key includes the game INDEX so "Seuraava peli" re-renders, and the iteration so
  // a fresh batch re-renders.
  const key = REPLAY_SRC + ':' + REPLAY_IDX + ':' + (r && r.frames ? (r.iter + ':' + r.seed + ':' + r.frames.length) : 'none');
  if (key === REPLAY_KEY) return; // same game still playing — let the animation continue
  REPLAY_KEY = key;
  REPLAY_FRAME = 0;
  if (!r || !r.frames || !r.frames.length) {
    const legacy = meta[4] === 'legacy';
    const fileName = 'replay' + (REPLAY_SRC === 'hard' ? '' : REPLAY_SRC === 'self' ? '_selfplay' : '_vs_' + REPLAY_SRC) + '.json';
    const emptyMsg = self
      ? 'Ei self-play-replayta vielä'
      : (isScripted ? ('Ei replayta vastustajalle ' + meta[3]) : 'Ei replay.jsonia vielä');
    panel.innerHTML = '<div class="replay">' + title + replayToggleHtml()
      + '<div class="empty">' + escapeHtml(emptyMsg)
      + ' — odotetaan tiedostoa <code style="color:var(--muted)">' + escapeHtml(fileName) + '</code> (kirjoitetaan joka --replay-every iteraatio).'
      + (legacy ? ' Tämä on VANHA liigan ulkopuolinen vastustaja (curriculum ei enää samplaa).' : '')
      + '</div></div>';
    wireReplayToggle();
    return;
  }
  // Staleness: the replay's iteration vs the latest training iteration. A replay is
  // stale if training has advanced ≥2 replay-cycles past the captured game (the user
  // should know they're watching an old game, not a silently blank/wrong one).
  const latestIter = STATE.latest && typeof STATE.latest.gen === 'number' ? STATE.latest.gen : null;
  const replayIter = (typeof r.iter === 'number') ? r.iter : null;
  const staleBy = (latestIter != null && replayIter != null) ? (latestIter - replayIter) : null;
  const blueLbl = self ? 'AI #1' : 'meidän AI';
  const redLbl = self ? 'AI #2' : (isScripted ? meta[3] : 'Hard CPU');
  const staleBanner = (staleBy != null && staleBy >= 25)
    ? '<div class="note" style="color:var(--win);margin:0 0 8px">⚠ Tämä replay on iteraatiosta ' + replayIter
        + ', koulutus on jo iteraatiossa ' + latestIter + ' (' + staleBy + ' jäljessä). Uusi replay kirjoitetaan seuraavalla --replay-every-syklillä.</div>'
    : '';
  panel.innerHTML =
    '<div class="replay">' + title + replayToggleHtml() + staleBanner
    + '<div class="stage">'
    + '<canvas id="replayCanvas"></canvas>'
    + '<div class="side" id="replaySide"></div>'
    + '</div>'
    + '<div class="ctl">'
    + '<button class="btn" id="replayPlay"></button>'
    + '<input type="range" id="replayScrub" min="0" max="' + (r.frames.length - 1) + '" value="0">'
    + '<span id="replaySpeed"></span>'
    + '<button class="btn" id="replayNext" title="Selaa tämän iteraation viittä tuoretta peliä">Seuraava peli ⏭</button>'
    + '<span id="replayGamePos" style="color:var(--muted);font-size:11px"></span>'
    + '</div>'
    + '<div class="leg">Reuna/sävy: <span style="color:#5aa9ff">sininen = ' + blueLbl + '</span> · <span style="color:#ff6b6b">punainen = ' + redLbl + '</span>'
    + ' · maasto: <span style="color:#3a9fd0">joki</span>, <span style="color:#8a929c">vuori</span>, <span style="color:#3f8a5c">metsä</span>, ruoho '
    + '· kirjaimet = rakennukset (F farm, M mine, V village, O outpost, H hydro, N nuclear, <b>B bridge</b>, ★ HQ, <span style="color:#c792ea">◆ Strange Device</span>) · keltainen numero = sotilaat (MarchSoldier näkyy sotilaiden siirtymisenä ruudusta toiseen)</div>'
    + '</div>';
  wireReplayToggle();
  const playBtn = document.getElementById('replayPlay');
  const scrub = document.getElementById('replayScrub');
  const speed = document.getElementById('replaySpeed');
  const syncPlay = () => { playBtn.textContent = REPLAY_PLAYING ? '⏸ tauko' : '▶ toista'; playBtn.className = 'btn' + (REPLAY_PLAYING ? ' on' : ''); };
  const syncSpeed = () => { speed.textContent = (REPLAY_FPS / 6).toFixed(1).replace(/\\.0$/, '') + '×'; };
  playBtn.onclick = () => { REPLAY_PLAYING = !REPLAY_PLAYING; syncPlay(); };
  scrub.oninput = () => { REPLAY_PLAYING = false; syncPlay(); REPLAY_FRAME = Number(scrub.value); const a = activeReplay(); drawReplayFrame(document.getElementById('replayCanvas'), a, REPLAY_FRAME); document.getElementById('replaySide').innerHTML = replaySideHtml(a, REPLAY_FRAME); };
  speed.style.cssText = 'cursor:pointer;color:var(--accent);font-weight:600;user-select:none';
  speed.onclick = () => { const steps = [3, 6, 12, 24]; REPLAY_FPS = steps[(steps.indexOf(REPLAY_FPS) + 1) % steps.length]; syncSpeed(); restartReplayTimer(); };
  // "Seuraava peli" — cycle through the five FRESH games of this iteration (wraps).
  const nextBtn = document.getElementById('replayNext');
  const gamePos = document.getElementById('replayGamePos');
  const syncGamePos = () => {
    const n = gamesFor(REPLAY_SRC).length || 1;
    gamePos.textContent = 'peli ' + (REPLAY_IDX + 1) + '/' + n + ' · iter ' + (r.iter ?? '?');
  };
  nextBtn.onclick = () => {
    const n = gamesFor(REPLAY_SRC).length;
    if (n < 2) { syncGamePos(); return; }
    REPLAY_IDX = (REPLAY_IDX + 1) % n;
    REPLAY_KEY = ''; REPLAY_FRAME = 0; REPLAY_PLAYING = true;
    renderReplay();
  };
  syncPlay(); syncSpeed(); syncGamePos();
  drawReplayFrame(document.getElementById('replayCanvas'), r, 0);
  document.getElementById('replaySide').innerHTML = replaySideHtml(r, 0);
  ensureReplayTimer();
}

function renderActive() {
  // Render all panels (cheap) so a tab switch is instant; visibility is CSS.
  renderSpatial();
  renderReplay();
  renderBuild();
  renderModels();
  renderResearch();
  render();
}

function render() {
  const fullLog = STATE.log || [];
  const data = windowed(fullLog);
  const latest = STATE.latest;
  const bench = STATE.benchmark;
  const benchWin = bench ? num(bench.winRate) : null;
  const winHistFull = (STATE.winHistory || []).filter(h => num(h.winRate) != null);
  // AlphaZero runs write policyLoss/valueLoss/bufferSize; GA runs write fitness.
  const isAz = !!latest && latest.policyLoss != null;

  document.getElementById('sub').textContent =
    'Colonizing Pirkanmaa · dir: ' + STATE.dir + (STATE.logMtime ? ' · loki päivitetty ' + STATE.logMtime : '');

  // window info
  const wi = document.getElementById('windowInfo');
  wi.textContent = fullLog.length
    ? (CTRL.window === 'all' ? 'näytetään kaikki ' + fullLog.length + ' sukupolvea' : 'näytetään viimeiset ' + data.length + ' / ' + fullLog.length + ' sukupolvea')
      + (CTRL.smooth ? ' · tasoitettu ±' + CTRL.smooth : '')
    : '';

  // --- summary cards -------------------------------------------------------
  const summary = document.getElementById('summary');
  summary.textContent = '';
  if (!latest) {
    summary.appendChild(statCard('Tila', 'odotetaan dataa…'));
  } else {
    // latest + best win-rate from the benchmark time series (fall back to benchmark.json / per-gen field)
    const lastWin = winHistFull.length ? num(winHistFull[winHistFull.length - 1].winRate)
      : (benchWin != null ? benchWin : num(latest.winRateVsHeur));
    let bestWin = null, bestGen = null;
    for (const h of winHistFull) { const w = num(h.winRate); if (w != null && (bestWin == null || w > bestWin)) { bestWin = w; bestGen = num(h.gen); } }
    let trend = '';
    if (winHistFull.length >= 2) {
      const a = num(winHistFull[winHistFull.length - 2].winRate), b = num(winHistFull[winHistFull.length - 1].winRate);
      if (a != null && b != null) trend = b >= a ? ' <span class="up">▲</span>' : ' <span class="down">▼</span>';
    }
    const winHero = (lastWin == null ? '—' : pct(lastWin)) + trend
      + (bestWin != null ? ' <small>best ' + pct(bestWin) + (bestGen != null ? ' @' + bestGen : '') + '</small>' : '');

    summary.appendChild(statCard(isAz ? 'Iteraatio' : 'Sukupolvi', String(latest.gen),
      null, isAz ? 'Koulutuskierros: self-play → gradient-treenaus → benchmark. Kasvaa ajan myötä.' : 'Evoluution sukupolvi.'));
    summary.appendChild(statCard('Win-rate vs hard', winHero, 'hero',
      'PÄÄTAVOITE (70 %): kuinka usein AI voittaa kovan CPU-botin benchmark-peleissä. "best N% @M" = paras tulos tähän asti (iteraatiossa M). ▲/▼ = suunta edellisestä mittauksesta.'));
    if (isAz) {
      // latest tile-frac vs hard from the benchmark series
      const lastTf = winHistFull.length ? num(winHistFull[winHistFull.length - 1].tileFrac) : null;
      summary.appendChild(statCard('Tiles vs hard', lastTf != null ? pct(lastTf) : '—',
        null, 'Kuinka suuren osan kartasta AI omistaa pelin lopussa kovaa bottia vastaan. Korkeampi = laajentaa ja dominoi paremmin (ei jää kotinurkkaan).'));
      summary.appendChild(statCard('Policy loss', fmt(latest.policyLoss),
        null, 'Kuinka hyvin policy-verkko ennustaa MCTS-haun suosimat siirrot (cross-entropy). Pienempi = parempi. Voi nousta hetkellisesti kun haku löytää uusia hyviä siirtoja.'));
      summary.appendChild(statCard('Value loss', fmt(latest.valueLoss),
        null, 'Kuinka hyvin value-verkko ennustaa pelin lopputuloksen asemasta (MSE). Pienempi = tarkempi voitto/tappio-arvio.'));
      summary.appendChild(statCard('Replay buffer', num(latest.bufferSize) != null ? num(latest.bufferSize).toLocaleString() : '—',
        null, 'Self-play-esimerkkien määrä koulutusmuistissa (rengaspuskuri; täyttyy kattoon ja vanhimmat poistuvat).'));
    } else {
      summary.appendChild(statCard('Best fitness', fmt(latest.bestFit), null, 'Populaation paras sopivuusarvo (GA).'));
      summary.appendChild(statCard('Mean fitness', fmt(latest.meanFit), null, 'Populaation keskimääräinen sopivuus (GA).'));
      summary.appendChild(statCard('Champion tiles', latest.championTileFrac != null ? pct(latest.championTileFrac) : '—',
        null, 'Mestarin keskimääräinen ruutuosuus self-play-peleissä.'));
    }
    const gps = num(latest.gamesPerSec);
    // Show decimals so a rate around 1 is legible (0.73 vs 1.42); drop them when large.
    const gpsStr = gps == null ? '—' : gps < 10 ? gps.toFixed(2) : gps < 100 ? gps.toFixed(1) : gps.toFixed(0);
    summary.appendChild(statCard('Throughput', gpsStr + ' <small>games/s</small>',
      null, 'Self-play-pelejä sekunnissa (koulutuksen läpimenonopeus). Desimaalit näkyvät kun nopeus on pieni.'));
    summary.appendChild(statCard('Ajoaika', fmtDur(latest.elapsedSec), null, 'Koulutuksen kokonaiskesto käynnistyksestä.'));
  }

  // --- charts --------------------------------------------------------------
  const root = document.getElementById('charts');
  root.textContent = '';
  if (!data.length) {
    const e = document.createElement('div');
    e.className = 'empty';
    e.textContent = 'odotetaan dataa…';
    root.appendChild(e);
    return;
  }

  // 1. Win-rate vs hard AI — a real CURVE from the benchmark time series (wide).
  const winHist = windowed(winHistFull);
  if (winHist.length) {
    const wseries = [
      { label: 'win-rate', color: getColor('--win'), values: smooth(winHist.map(h => num(h.winRate))), thick: true, dots: true },
    ];
    if (winHist.some(h => num(h.timeoutRate) != null))
      wseries.push({ label: 'timeout', color: getColor('--timeout'), values: smooth(winHist.map(h => num(h.timeoutRate))), dashed: true });
    if (winHist.some(h => num(h.lossRate) != null))
      wseries.push({ label: 'loss', color: getColor('--loss'), values: smooth(winHist.map(h => num(h.lossRate))), dashed: true });
    const last = num(winHist[winHist.length - 1].winRate);
    // 95% Wald CI half-width per point (1.96·√(p(1−p)/n)); a 40-game bench is ±~15pp,
    // so the band shows when an apparent rise is real vs benchmark noise.
    const ciW = winHist.map(h => { const p = num(h.winRate), nn = num(h.nGames); return (p != null && nn) ? 1.96 * Math.sqrt(Math.max(p * (1 - p), 0) / nn) : null; });
    const lastN = num(winHist[winHist.length - 1].nGames);
    const lastCi = (last != null && lastN) ? 1.96 * Math.sqrt(Math.max(last * (1 - last), 0) / lastN) : null;
    root.appendChild(chart('Win-rate vs hard AI', winHist, wseries, {
      pct: true, range: [0, 1], wide: true, dots: true,
      hint: 'real win-rate vs hard heuristic — seat-averaged, 95% CI band',
      band: CTRL.smooth ? null : { center: winHist.map(h => num(h.winRate)), width: ciW, color: getColor('--win') },
      tip: 'Oikea win-rate kovaa CPU-bottia vastaan per benchmark, KESKIARVOISTETTU molempien aloituspaikkojen yli. Keltainen = voitot (tavoite kohti 70 %), harmaa = aidot tasapelit, punainen = häviöt. Varjostettu vyö = 95 % luottamusväli (±1.96·√(p(1−p)/n)) — kertoo milloin nousu on aitoa vs. benchmark-kohinaa.',
      note: 'latest ' + pct(last) + (lastCi != null ? ' ±' + (lastCi * 100).toFixed(1) + 'pp' : '') + ' · ' + winHist.length + ' benchmark point(s)',
    }));
  } else {
    const card = document.createElement('div');
    card.className = 'chart wide';
    const h = document.createElement('h2'); h.textContent = 'Win-rate vs hard AI'; card.appendChild(h);
    const e = document.createElement('div'); e.className = 'empty';
    e.textContent = benchWin != null ? 'latest benchmark: ' + pct(benchWin) + ' (history file not written yet)' : 'no benchmark yet — the benchmark sidecar appends benchmark-history.jsonl';
    card.appendChild(e);
    root.appendChild(card);
  }

  if (isAz) {
    // --- AlphaZero charts --------------------------------------------------
    const latestBench = winHistFull.length ? winHistFull[winHistFull.length - 1] : null;

    // ★★ PILLAR-6 HEADLINE: per-opponent win-rate over training time. ONE labeled
    // series per rebuilt SD3 league opponent (Rusher / Fortress / Device / Strong /
    // HARD) from the benchVs* bench fields. This is the key "win-rate vs each scripted
    // opponent at a glance" view. Distinct colour AND line style per series (a11y:
    // chart guidance says don't rely on colour alone). Guarded: pre-Pillar-6 history
    // lacks benchVs* → empty-state with a clear "next run only" hint instead of blank.
    {
      const LEAGUE = [
        ['benchVsRusher',     'vs Rusher',      getColor('--loss'),    false],
        ['benchVsFortress',   'vs Fortress',    getColor('--div'),     true ],
        ['benchVsDeviceRush', 'vs Device Rush', getColor('--bank'),    true ],
        ['benchVsStrongArmy', 'vs Strong Army', getColor('--median'),  false],
        ['benchVsHard',       'vs HARD',        getColor('--win'),     false],
      ];
      const hasLeague = winHist.length && LEAGUE.some(([k]) => winHist.some(h => num(h[k]) != null));
      if (hasLeague) {
        const lseries = LEAGUE.map(([k, lbl, c, dash]) => ({
          label: lbl, color: c, values: smooth(winHist.map(h => num(h[k]))),
          dashed: dash, thick: k === 'benchVsHard', dots: true,
        }));
        // legacy winRateVsHeur reference removed (it is always null in AZ logs and
        // benchVsHard is the same measurement on the per-opponent budget) — kept the
        // full-budget win-rate as a faint reference instead.
        lseries.push({ label: 'win-rate (full bench)', color: getColor('--muted') || '#8b97a3',
          values: smooth(winHist.map(h => num(h.winRate))), dashed: true });
        const lastTxt = LEAGUE.map(([k, lbl]) => {
          const v = num(winHist[winHist.length - 1][k]); return v == null ? null : lbl.replace('vs ', '') + ' ' + pct(v);
        }).filter(Boolean).join(' · ');
        const perN = num(winHist[winHist.length - 1].benchPerOpp);
        root.appendChild(chart('Win-rate vs jokainen liigavastustaja (Pillar 6)', winHist, lseries, {
          pct: true, range: [0, 1], wide: true, dots: true,
          hint: 'per-opponent bench · ' + (perN ? perN + ' games/opp' : 'league'),
          note: lastTxt || 'latest league bench',
          tip: 'PÄÄNÄKYMÄ (Pillar 6): oppijan voittoprosentti JOKAISTA uudelleenrakennettua SD3-liigan vastustajaa vastaan per benchmark — Rusher / Fortress / Device Rush / Strong Army + HARD-mittatikku. Jokainen sarja eri väri JA viivatyyli (saavutettavuus). Katkoviiva harmaa = koko-budjetin win-rate vs HARD (vertailu). Näkyy vain kun ajo käyttää Pillar-6-binääriä (benchVs*-kentät); vanhat ajot näyttävät tyhjän.',
        }));
      } else {
        const card = document.createElement('div');
        card.className = 'chart wide';
        const h = document.createElement('h2');
        h.textContent = 'Win-rate vs jokainen liigavastustaja (Pillar 6)';
        const sp = document.createElement('span'); sp.className = 'hint';
        sp.textContent = ' · per-opponent bench'; h.appendChild(sp);
        card.appendChild(h);
        const e = document.createElement('div'); e.className = 'empty';
        e.textContent = 'Ei benchVs*-kenttiä vielä — nämä ilmestyvät vasta kun ajo käyttää Pillar-6-binääriä (per-opponent benchmark). Nykyinen ajo ennustaa tätä pillaria; seuraava ajo täyttää käyrät.';
        card.appendChild(e);
        root.appendChild(card);
      }
    }

    // ★ §10 HEADLINE: who won, and HOW — our AI vs the hard CPU, split by cause.
    root.appendChild(causeCard(latestBench));

    // ★★ PILLAR-6 ACTIVITY / PASSIVITY panel — is the net passively turtling or
    // intelligently aggressive? Reads the bench + per-iter self-play fields actually
    // present in the log. One glance: army size, pass%, contact, march usage, crack
    // attempts/successes, bridges.
    root.appendChild(activityCard(latestBench, latest, data));

    // ★0 HONEST WIN-RATE (Step 0): trueWinVsHard (bankruptcy-propped wins removed)
    // vs the raw win-rate. The GAP between the two curves = the bankruptcy MIRAGE
    // (~30% of "wins" are free enemy self-bankruptcy). Judge progress by the honest
    // (solid) line, not the raw (dashed) one. Guarded: old history lines lack the
    // field → that series is null and the panel still renders the raw curve.
    if (winHist.length && winHist.some(h => num(h.trueWinVsHard) != null)) {
      const lastTrue = num(winHist[winHist.length - 1].trueWinVsHard);
      const lastRaw = num(winHist[winHist.length - 1].winRate);
      root.appendChild(chart('Honest win-rate vs hard AI (Step 0)', winHist, [
        { label: 'trueWinVsHard (honest)', color: getColor('--best'), values: smooth(winHist.map(h => num(h.trueWinVsHard))), thick: true, dots: true },
        { label: 'winRate (raw — incl. mirage)', color: getColor('--win'), values: smooth(winHist.map(h => num(h.winRate))), dashed: true },
      ], { pct: true, range: [0, 1], wide: true, dots: true,
        hint: 'gap between curves = bankruptcy mirage',
        note: 'latest honest ' + pct(lastTrue) + ' vs raw ' + pct(lastRaw)
          + (lastTrue != null && lastRaw != null ? ' · mirage ' + ((lastRaw - lastTrue) * 100).toFixed(1) + 'pp' : ''),
        tip: 'HONEST headline (Step 0). trueWinVsHard = champ wins EXCLUDING bankruptcy-propped wins, jaettuna pelimäärällä = (device+domination+conquest+tiebreak)/nGames. Raw winRate laskee vastustajan oman konkurssin "voitoksi" (~30 % voitoista on tätä illuusiota). Käyrien VÄLI = konkurssi-illuusio. Arvioi edistystä VAIN vihreästä (aito) käyrästä.' }));
    }

    // ★0b PER-SKILL behavioral counters (Step 0): standing Villages / Outposts (econ
    // + army-prerequisite) and PEAK soldiers per game (the "fields an army" signal,
    // currently 0–3). Tighter than the ±12.6% win-rate. Soldiers on its own axis
    // (counts), villages/outposts share the same small-count axis.
    if (winHist.length && winHist.some(h => num(h.maxSoldiersPerGame) != null)) {
      root.appendChild(chart('Per-skill: standing Villages / Outposts (Step 0)', winHist, [
        { label: 'villages / game', color: getColor('--tile'), values: smooth(winHist.map(h => num(h.villagesPerGame))), dots: true },
        { label: 'outposts / game', color: getColor('--sigma'), values: smooth(winHist.map(h => num(h.outpostsPerGame))), dots: true },
      ], { range: [0, 4],
        hint: 'built-and-survived at game end · avg per game',
        tip: 'Per-skill BEHAVIORAL signaalit (Step 0). Standing Village/Outpost = rakennettu JA selvinnyt pelin loppuun (vain pelaajan omistamilta ruuduilta). Village = talous + työvoimakatto; Outpost = armeijan edellytys (sotilaskatto) + Device-linjan vaatimus. >0 ja PYSYVÄ = skill alkaa emergoitua. Tiukempi mittari kuin ±12.6 % win-rate.' }));
      root.appendChild(chart('Per-skill: peak army size (Step 0)', winHist, [
        { label: 'max soldiers / game', color: getColor('--bank'), values: smooth(winHist.map(h => num(h.maxSoldiersPerGame))), thick: true, dots: true },
      ], { range: [0, 6],
        hint: 'peak fielded soldiers, avg per game · target > 3',
        tip: 'Suurin kentällä ollut sotilasmäärä per peli, keskiarvoistettu. "Fields an army" -signaali — tällä hetkellä 0–3 (sotilaskatto = HQ+1 ilman Outpostia). Tavoite: rutiininomaisesti > 3 (dokumentoitu epäonnistumiskynnys).' }));

      // ★0b-dist Peak-soldier DISTRIBUTION (latest bench). The avg-per-game line
      // above hides bimodal behaviour — a flat 1.0 mean can be "always 1 soldier"
      // OR "0 or 3, never 1/2". This panel makes the shape explicit: x-axis bins
      // [0,1,2,3,4+], y-axis = number of bench games. Reads champSoldierBins
      // emitted by the Rust trainer; older history lines lacking the field render
      // the empty state (the surrounding winHist.some(...) gate hides the whole
      // pair when neither metric is present).
      var lbDist = winHist.length ? winHist[winHist.length - 1] : null;
      var dist = lbDist && lbDist.champSoldierBins ? lbDist.champSoldierBins : null;
      if (dist) {
        var binOrder = ['0', '1', '2', '3', '4+'];
        var totalGames = binOrder.reduce(function (s, k) { return s + (num(dist[k]) || 0); }, 0);
        var rows = binOrder.map(function (k) {
          var c = num(dist[k]) || 0;
          var sharePct = totalGames > 0 ? (100 * c / totalGames).toFixed(1) + '%' : '—';
          return {
            label: k + (k === '1' ? ' soldier' : ' soldiers'),
            value: c,
            text: c + ' / ' + totalGames + ' · ' + sharePct,
            // Mirror the army-size line colour so the visual link is obvious.
            color: getColor('--bank'),
          };
        });
        root.appendChild(barListCard(
          'Per-skill: peak-soldiers distribution (Step 0)',
          'latest bench · ' + totalGames + ' games',
          rows,
          {
            labelWidth: 110,
            valWidth: 140,
            note: 'Bins 0/1/2/3/4+ = number of bench games whose CHAMPION reached that PEAK soldier count. Makes a flat ~1.0 avg distinct from a bimodal one (e.g. "0 or 3, never 1/2"). 4+ stays at 0 until the soldier cap is raised (current cap = HQ+1 without Outpost = 3).',
          }
        ));
      }
    }

    // ★PB — PLAN-B INTENT ACTIVITY (BuildBridge, CrackDevice, CrackHQ).
    // The 2026-06-05 redesign added three new first-class intents to break out
    // of the passivity equilibrium: bridge-builds (cross rivers), device-cracks
    // (counter HARD's device line), HQ-cracks (directed offence). All three
    // panels are GATED on field presence so older runs render nothing.

    // PB1 — time-series: bridges / device-crack-successes / HQ-crack-successes
    // per game, across benches. Memo gate: bridgesPerGame >= 0.3 means the
    // policy is actually exercising the new BuildBridge action.
    if (winHist.length && winHist.some(h => num(h.bridgesPerGame) != null || num(h.crackHQSuccesses) != null)) {
      root.appendChild(chart('Plan-B: uudet intentit per peli (bridges / HQ-crack / device-crack)', winHist, [
        { label: 'bridgesPerGame', color: getColor('--tile'),
          values: smooth(winHist.map(h => num(h.bridgesPerGame))), dots: true, thick: true },
        { label: 'crackHQ-success / game', color: getColor('--bank'),
          values: smooth(winHist.map(h => {
            var s = num(h.crackHQSuccesses); var n = num(h.nGames);
            return (s != null && n) ? s / n : null;
          })), dots: true, thick: true },
        { label: 'crackDevice-success / game', color: getColor('--sigma'),
          values: smooth(winHist.map(h => {
            var s = num(h.crackDeviceSuccesses); var n = num(h.nGames);
            return (s != null && n) ? s / n : null;
          })), dots: true },
      ], { range: [0, 1.5],
        hint: 'memo gate: bridgesPerGame >= 0.3 · crackHQ kasvava trend = HQ-suunnattu offence',
        tip: 'Plan-B uudet first-class -toiminnot. bridgesPerGame = silloja rakennettu / pelimäärä per bench. crackHQ-success / game = pelien osuus joissa champ valloitti vihollisen HQ:n. crackDevice-success / game = pelien osuus joissa champ tuhosi vihollisen Strange Devicen ennen countdownia. Gate: bridgesPerGame >= 0.3 viimeisen 6 benssin keskiarvossa.' }));
    }

    // PB2 — latest-bench Plan-B intent counts (raw attempts vs successes).
    // Two-bar comparison per intent (attempts vs successes) to see whether
    // the policy is using each action AND whether it succeeds when it tries.
    if (latestBench
        && (num(latestBench.crackHQAttempts) != null
            || num(latestBench.crackDeviceAttempts) != null
            || num(latestBench.bridgesPerGame) != null)) {
      const ng = num(latestBench.nGames) || 60;
      const bridges = Math.round((num(latestBench.bridgesPerGame) || 0) * ng);
      const rows = [
        { label: 'BuildBridge (count)', value: bridges,
          text: bridges + ' / ' + ng + ' games',
          color: getColor('--tile') },
        { label: 'CrackHQ attempts', value: num(latestBench.crackHQAttempts) || 0,
          text: (num(latestBench.crackHQAttempts) || 0) + ' attempts',
          color: getColor('--bank') },
        { label: 'CrackHQ successes', value: num(latestBench.crackHQSuccesses) || 0,
          text: (num(latestBench.crackHQSuccesses) || 0) + ' successes',
          color: getColor('--best') },
        { label: 'CrackDevice attempts', value: num(latestBench.crackDeviceAttempts) || 0,
          text: (num(latestBench.crackDeviceAttempts) || 0) + ' attempts',
          color: getColor('--sigma') },
        { label: 'CrackDevice successes', value: num(latestBench.crackDeviceSuccesses) || 0,
          text: (num(latestBench.crackDeviceSuccesses) || 0) + ' successes',
          color: getColor('--best') },
      ];
      root.appendChild(barListCard(
        'Plan-B: uudet intentit · viimeisin bench (60 peliä)',
        ng + ' games',
        rows,
        {
          labelWidth: 170,
          valWidth: 160,
          note: 'BuildBridge / CrackHQ / CrackDevice toimivat first-class intentteinä Plan-B:n jälkeen. attempts = pelit joissa champ valitsi tämän toiminnon vähintään kerran. successes = sama + vastapuolen kohde menetetty pelin aikana. CrackHQ success-aste on lähellä 100% kun candidate emit:tää (HQ-laatat usein puolustamattomia per §5). gate: kaikki kolme >0 ja kasvavat.',
        }
      ));
    }

    // ★M — BEHAVIORAL DIAGNOSTIC PANELS (M1–M9). Each panel is GATED on field
    // presence so older history lines (cnn-bc2 et al, predating these fields)
    // render no panel at all. Computed in the Rust bench loop (bench_vs_hard)
    // and emitted into benchmark-history.jsonl as additive keys. The user wants
    // these as the next-generation passivity diagnosis tightening: each picks at
    // a specific failure mode (idle workers, idle soldiers, never-Outpost, etc.).

    // M1 — unit efficiency: % of worker/expert ROUNDS spent on a producer
    // building (Farm/Mine/Village/Hydro/Nuclear). Farms count even during the
    // 4-round growth warmup per the user-stated rule. A passive AI that hires
    // workers and parks them on grassland shows near-0% here.
    if (winHist.length && winHist.some(h => num(h.unitEfficiency) != null)) {
      root.appendChild(chart('M1 — yksikön tehokkuus (worker/expert PRODUCING) [legacy]', winHist, [
        { label: 'unitEfficiency', color: getColor('--best'),
          values: smooth(winHist.map(h => num(h.unitEfficiency))), thick: true, dots: true },
      ], { pct: true, range: [0, 1],
        hint: 'producing_rounds / (producing + idle) · per bench (60 games)',
        tip: 'M1 (vanhentunut, suppea luokittelija). Osuus kierroksista jonka työläiset/asiantuntijat seisovat tuottavalla RAKENNUKSELLA (Farm / Mine / Village / Hydro / Nuclear). Farm-ruutu lasketaan TUOTTAVAKSI myös 4-kierroksen lämpenemisen aikana. Korvattu alapuolella olevalla USEFUL-vs-USELESS -pylväskuviolla (2026-06-05 käyttäjäpalaute), joka huomioi myös luontaisesti tuottavat maastoruudut ja Expand-tapahtumat. Säilytetty vertailua varten.' }));
    }

    // M1 (Correction 1, 2026-06-05) — NEW broader USEFUL classifier as a two-bar
    // raw-count comparison (per user feedback): USEFUL = worker/expert rounds on a
    // producer building OR on a champ-owned natural-producing tile (Forest with
    // wood_left > 0 / AbundantForest — Mountain & River need a building so they're
    // NOT credited) OR a champion Expand event (the worker actively claimed/moved
    // this round). USELESS = the inverse.
    if (latestBench
        && num(latestBench.unitUsefulRounds) != null
        && num(latestBench.unitUselessRounds) != null) {
      const useful = num(latestBench.unitUsefulRounds) || 0;
      const useless = num(latestBench.unitUselessRounds) || 0;
      const total = useful + useless;
      const rows = [
        { label: 'USEFUL', value: useful,
          text: useful + (total > 0 ? ' · ' + (100 * useful / total).toFixed(0) + '%' : ''),
          color: getColor('--best') },
        { label: 'USELESS', value: useless,
          text: useless + (total > 0 ? ' · ' + (100 * useless / total).toFixed(0) + '%' : ''),
          color: getColor('--muted') },
      ];
      root.appendChild(barListCard(
        'M1 — yksikön tehokkuus · USEFUL vs USELESS (laaja luokittelija)',
        'uusin benchmark · raw unit-round counts',
        rows,
        { labelWidth: 110, valWidth: 130,
          note: 'USEFUL = worker/expert tuottavalla rakennuksella (Farm/Mine/Village/Hydro/Nuclear) TAI omalla Forest-ruudulla (wood_left > 0) TAI AbundantForest-ruudulla TAI champion käytti Expand-intentin tällä kierroksella. USELESS = muut. Pylväät ovat raw counts (ei suhde) joten lyhyet pelit eivät peitä trendiä — laske summa molemmista ymmärtääksesi mittakaavan.' }));
    }

    // M2 — soldier-position split: % of soldier-rounds in each role.
    // ATTACKING (conquering), DEFENDING (own tile next to enemy), IDLE (interior).
    if (winHist.length && winHist.some(h => num(h.soldierAttack) != null || num(h.soldierDefend) != null)) {
      root.appendChild(chart('M2 — sotilaan rooli (attack / defend / idle) [3-luokka]', winHist, [
        { label: 'attack (staged on enemy)', color: getColor('--bank'),
          values: smooth(winHist.map(h => num(h.soldierAttack))), dots: true },
        { label: 'defend (frontier-owned)', color: getColor('--best'),
          values: smooth(winHist.map(h => num(h.soldierDefend))), dots: true },
        { label: 'idle (interior)', color: getColor('--muted'),
          values: smooth(winHist.map(h => num(h.soldierIdle))), dashed: true, dots: true },
      ], { pct: true, range: [0, 1],
        hint: 'osuus soldier-rounds per bench',
        tip: 'M2 (kolmen luokan jakauma, säilytetty yksityiskohtia varten). Sotilaiden roolijakauma (osuus soldier-roundeista per benchmark). ATTACK = conquering_units-listalla (vihollisruudulla staged, §2). DEFEND = omalla ruudulla joka rajoittuu (orth-4) vihollisruutuun (rintama, §4 — siellä sotilaat oikeasti merkitsevät). IDLE = sisämaa-ruudulla. Korkea IDLE = sotilaat eivät pääse / mene rintamaan. Otsikkomittari on alapuolella oleva USEFUL-vs-USELESS -pylväskuvio (2026-06-05 käyttäjäpalaute).' }));
    }

    // M2 (Correction 2, 2026-06-05) — USEFUL vs USELESS two-bar headline panel
    // (per user feedback): USEFUL = ATTACK + DEFEND combined; USELESS = IDLE.
    // The 3-bucket chart above stays for drill-down.
    if (latestBench
        && num(latestBench.soldierUsefulRounds) != null
        && num(latestBench.soldierUselessRounds) != null) {
      const useful = num(latestBench.soldierUsefulRounds) || 0;
      const useless = num(latestBench.soldierUselessRounds) || 0;
      const total = useful + useless;
      const rows = [
        { label: 'USEFUL', value: useful,
          text: useful + (total > 0 ? ' · ' + (100 * useful / total).toFixed(0) + '%' : ''),
          color: getColor('--best') },
        { label: 'USELESS', value: useless,
          text: useless + (total > 0 ? ' · ' + (100 * useless / total).toFixed(0) + '%' : ''),
          color: getColor('--muted') },
      ];
      root.appendChild(barListCard(
        'M2 — sotilaan rooli · USEFUL vs USELESS',
        'uusin benchmark · raw soldier-round counts',
        rows,
        { labelWidth: 110, valWidth: 130,
          note: 'USEFUL = ATTACK + DEFEND (sotilaat tekevät jotain). USELESS = IDLE (sisämaa-ruudulla, ei rintamaa). Pylväät raw counts. Tarkempi kolmen luokan erittely yllä olevassa kaaviossa.' }));
    }

    // M3 / M4 — win-rate split by per-game villages-built / outposts-built.
    // Each bar is a build-count bin (0 / 1 / 2 / 3+); height = champ win-rate
    // within that bin. Label includes raw game count so a sparse bin doesn't
    // mislead. Per the user's spec the bins are 0,1,2,3+.
    function winByBuildsCard(title, hint, src, color) {
      if (!src) return null;
      const bins = ['0', '1', '2', '3+'];
      const rows = bins.map(k => {
        const slot = src[k] || { games: 0, wins: 0 };
        const g = num(slot.games) || 0;
        const w = num(slot.wins) || 0;
        const rate = g > 0 ? w / g : null;
        return {
          label: k + (k === '1' ? ' rakennettu' : k === '0' ? ' rakennettu' : ' rakennettu'),
          value: rate == null ? 0 : rate,
          text: g > 0 ? (100 * w / g).toFixed(0) + '% · ' + w + '/' + g
                     : '— · ' + g + ' peliä',
          color: color,
        };
      });
      // The bar list uses the raw value; we pre-scaled rate to [0,1].
      return barListCard(title, hint, rows, {
        labelWidth: 140, valWidth: 110,
        note: 'Pylvään korkeus = champion-voittoprosentti tässä bin:issä. "n / N" = voitot / pelit tässä bin:issä uusimmasta benchistä.',
      });
    }
    if (latestBench && latestBench.winByVillagesBuilt) {
      const card = winByBuildsCard(
        'M3 — voittoprosentti vs. rakennettuja Villageja',
        'uusin benchmark · per peli',
        latestBench.winByVillagesBuilt, getColor('--tile'));
      if (card) root.appendChild(card);
    }
    if (latestBench && latestBench.winByOutpostsBuilt) {
      const card = winByBuildsCard(
        'M4 — voittoprosentti vs. rakennettuja Outposteja',
        'uusin benchmark · per peli · Outpost = soldier-cap unlock (§5)',
        latestBench.winByOutpostsBuilt, getColor('--sigma'));
      if (card) root.appendChild(card);
    }

    // M5 — AI-vs-AI contact rate (PURE self-play games). A game "made contact"
    // iff ≥1 Attack intent OR any tile carried ≥1 conquering unit at some point.
    // From the per-iter log line (updates every iteration, unlike the every-5
    // benchmark) → faster signal than bench-derived metrics.
    if (data.length && data.some(r => num(r.spContactRate) != null)) {
      root.appendChild(chart('M5 — kontaktiprosentti (self-play AI-vs-AI) [trendi]', data, [
        { label: 'spContactRate', color: getColor('--bank'),
          values: scol(data, 'spContactRate'), thick: true },
      ], { pct: true, range: [0, 1],
        hint: 'self-play · games with ≥1 Attack OR staged conqueror · per iteration',
        tip: 'M5 (suhde-trendi, säilytetty trendiä varten). Osuus pure-self-play-peleistä joissa AI:t törmäsivät. Lähellä 0 = molemmat pelit kasvavat rinnakkain ilman koskaan kohtaamista (passivity). Otsikkomittari on alapuolella oleva CONTACT-vs-NO-CONTACT pylväskuvio (2026-06-05 käyttäjäpalaute).' }));
    }

    // M5 (Correction 3, 2026-06-05) — raw counts side-by-side as a two-bar
    // comparison (per user feedback). spContact + spContactN already exist on the
    // iter log line. spNoContact = spContactN - spContact.
    {
      const latestIter = data.length ? data[data.length - 1] : null;
      if (latestIter
          && num(latestIter.spContact) != null
          && num(latestIter.spContactN) != null) {
        const contact = num(latestIter.spContact) || 0;
        const total = num(latestIter.spContactN) || 0;
        const noContact = Math.max(0, total - contact);
        const rows = [
          { label: 'CONTACT', value: contact,
            text: contact + (total > 0 ? ' · ' + (100 * contact / total).toFixed(0) + '%' : ''),
            color: getColor('--bank') },
          { label: 'NO CONTACT', value: noContact,
            text: noContact + (total > 0 ? ' · ' + (100 * noContact / total).toFixed(0) + '%' : ''),
            color: getColor('--muted') },
        ];
        root.appendChild(barListCard(
          'M5 — self-play kontakti · CONTACT vs NO-CONTACT',
          'uusin iter · raw game counts',
          rows,
          { labelWidth: 110, valWidth: 140,
            note: 'CONTACT = pelissä oli ≥1 Attack-intentti TAI ≥1 staged conquering -yksikkö (kummalla tahansa puolella). NO-CONTACT = pelit joissa ei kohdattu lainkaan. Yhteensä = spContactN. Raw counts viimeisestä iteraatiosta.' }));
      }
    }

    // M6 — soldier STACKING: per-game peak stack-size on any single tile,
    // bucketed 1/2/3 (the §2 cap is 3). Bin 0 (= never had a soldier) is already
    // covered by champSoldierBins. % of bench games per bin.
    if (latestBench && latestBench.stackBins) {
      const sb = latestBench.stackBins;
      const bins = ['1', '2', '3'];
      const total = bins.reduce((s, k) => s + (num(sb[k]) || 0), 0);
      const rows = bins.map(k => {
        const c = num(sb[k]) || 0;
        const share = total > 0 ? (100 * c / total).toFixed(1) + '%' : '—';
        return { label: 'max-stack ' + k, value: c,
                 text: c + ' / ' + total + ' · ' + share,
                 color: getColor('--bank') };
      });
      root.appendChild(barListCard(
        'M6 — sotilaiden pinoutuminen (peak-stack per peli)',
        'uusin benchmark · ' + total + ' peliä joissa champ kentällä',
        rows,
        { labelWidth: 110, valWidth: 140,
          note: 'Bin = pelin suurin SAMAN ruudun sotilaspino (oma + valloitus, §2 max 3). Korkea 3:n bin = AI keskittää voimat; pelkkä 1:n bin = sotilaat ovat sirpaloituneet.',
        }));
    }

    // M7 — experts hired per game (champion side). User said the AI should learn
    // experts boost production; currently this is 0 in many runs. Per-bench rate.
    if (winHist.length && winHist.some(h => num(h.expertsHiredPerGame) != null)) {
      root.appendChild(chart('M7 — palkattuja Experteja / peli (champ)', winHist, [
        { label: 'expertsHiredPerGame', color: getColor('--div'),
          values: smooth(winHist.map(h => num(h.expertsHiredPerGame))), thick: true, dots: true },
      ], { hint: 'champ side · avg per bench game',
        tip: 'M7. Kuinka monta Expertia champion palkkasi keskimäärin per benchmark-peli. Expert kaksinkertaistaa tuotannon — käyttäjä haluaa nähdä että AI oppii tämän. 0 = ei koskaan palkkaa.' }));
    }

    // M8 — frontier ratio: fraction of champ-owned tiles bordering ≥1 enemy
    // tile, averaged across rounds, averaged across bench games. Proxy for
    // aggression posture vs turtling.
    if (winHist.length && winHist.some(h => num(h.frontierRatio) != null)) {
      root.appendChild(chart('M8 — rintamasuhde (frontier ratio)', winHist, [
        { label: 'frontierRatio', color: getColor('--accent'),
          values: smooth(winHist.map(h => num(h.frontierRatio))), thick: true, dots: true },
      ], { pct: true, range: [0, 1],
        hint: 'osuus omista ruuduista jotka rajoittuvat viholliseen · avg per peli, avg per bench',
        tip: 'M8. Keskimäärin per kierros: osuus champion-omistamista ruuduista jotka ovat orthog-4 viholliseen rajoittuvia. Korkea = aggressiivinen ekspansio (rintamaa joka puolella). Matala = kotiintunut puolustus (turtling). Tunnistaa passivity-mallin pelityylin tasolla.' }));
    }

    // M9 — average game length by champion outcome (win vs loss). A different
    // lens than per-cause: fast wins = decisive play, long losses = slow attrition.
    if (winHist.length && winHist.some(h => h.roundsByOutcome
        && (num(h.roundsByOutcome.win) != null || num(h.roundsByOutcome.loss) != null))) {
      const winRds = winHist.map(h => num((h.roundsByOutcome || {}).win));
      const lossRds = winHist.map(h => num((h.roundsByOutcome || {}).loss));
      root.appendChild(chart('M9 — pelin pituus voitossa vs. tappiossa', winHist, [
        { label: 'rounds when champ WON', color: getColor('--win'),
          values: smooth(winRds), thick: true, dots: true },
        { label: 'rounds when champ LOST', color: getColor('--loss'),
          values: smooth(lossRds), dashed: true, dots: true },
      ], { hint: 'avg rounds per bench, split by champ outcome',
        tip: 'M9. Pelin keskimääräinen pituus eriteltynä champion lopputuloksen mukaan: voitto vs. tappio. Nopea voitto = ratkaisevaa pelaamista; pitkä tappio = kulutussotaa. Eri linssi kuin per-syy (Device/Conquest/...): kertoo pelin DYNAMIIKASTA voiton/tappion takana.' }));
    }

    // ★0c device-DENIAL + bankruptcy-share (Step 0). Denial = HARD built a Strange
    // Device but did NOT win by it (cracked/prevented) — a defense/reaction signal.
    // bankruptcy-share = champ bankruptcy wins / total champ wins (the mirage, as a
    // fraction of OUR wins).
    if (winHist.length && winHist.some(h => num(h.deviceDenialRate) != null || num(h.bankruptcyWinShare) != null)) {
      root.appendChild(chart('Per-skill: device-denial & bankruptcy-share (Step 0)', winHist, [
        { label: 'device-denial rate', color: getColor('--div'), values: smooth(winHist.map(h => num(h.deviceDenialRate))), dots: true },
        { label: 'bankruptcy share of wins', color: getColor('--loss'), values: smooth(winHist.map(h => num(h.bankruptcyWinShare))), dashed: true, dots: true },
      ], { pct: true, range: [0, 1],
        hint: 'denial = enemy device cracked/prevented · share = mirage fraction',
        tip: 'device-denial rate = osuus HARD:n rakentamista Strange Deviceista, joilla HARD EI voittanut (= halpa counter: yksi sotilas puolustamattomalle device-ruudulle, tai HARD hävisi ensin). hardDeviceDenied / hardDeviceBuilt. bankruptcy share = champ-konkurssivoitot / kaikki champ-voitot = illuusion osuus OMISTA voitoistamme (tee illuusio näkyväksi). Tavoite: denial ↑, bankruptcy-share ↓.' }));
    }

    // ★1 Policy entropy (training health — collapse = policy freeze).
    root.appendChild(chart('Policy entropy', data, [
      { label: 'policyEntropy', color: getColor('--accent'), values: scol(data, 'policyEntropy') },
    ], { range: [0, 1], hint: 'normalised · 1=exploratory 0=frozen',
      tip: 'Policy-jakauman normalisoitu entropia (÷ln K) keskimäärin self-play-tiloissa. 1 ≈ tasajakauma/tutkiva, →0 ≈ romahtanut/ylivarma → policy-jäätyminen. Aikainen romahdus = juuttuminen yhteen siirtoon.' }));

    // ★2 Value calibration — predicted value bucketed by the TRUE outcome.
    root.appendChild(chart('Value calibration (pred vs outcome)', data, [
      { label: 'win→+1', color: getColor('--best'), values: scol(data, 'valPredWin') },
      { label: 'draw→0', color: getColor('--muted'), values: scol(data, 'valPredDraw') },
      { label: 'loss→−1', color: getColor('--loss'), values: scol(data, 'valPredLoss') },
    ], { range: [-1, 1], marker: 0, markerColor: getColor('--grid'),
      tip: 'Value-verkon ENNUSTAMA arvo ryhmiteltynä todellisen lopputuloksen mukaan. Terve verkko ajaa voitot →+1, häviöt →−1, tasapelit →0. Jos kaikki kolme romahtavat lähelle nollaa → draw-collapse (verkko ennustaa kaiken tasapeliksi).' }));

    // ★4 Strange Device — build rate & survival (mechanic balance).
    if (winHist.length && winHist.some(h => num(h.deviceBuildRate) != null)) {
      root.appendChild(chart('Strange Device — build & survival', winHist, [
        { label: 'build rate', color: getColor('--div'), values: smooth(winHist.map(h => num(h.deviceBuildRate))), dots: true },
        { label: 'survival', color: getColor('--median'), values: smooth(winHist.map(h => num(h.deviceSurvival))), dots: true },
      ], { pct: true, range: [0, 1],
        tip: 'Build rate = osuus peleistä, joissa Strange Device rakennettiin. Survival = rakennetuista laitteista se osuus, joka selvisi laskuriin (= voitto laitteella). build rate ≈0 → mekaniikka kuollut; survival kertoo onko X/hinta tasapainossa.' }));
    }

    // ★5 Win-rate by seat (first-mover advantage — surface it to correct for it).
    if (winHist.length && winHist.some(h => num(h.winSeat0) != null)) {
      root.appendChild(chart('Win-rate by seat', winHist, [
        { label: 'seat 0 (first)', color: getColor('--win'), values: smooth(winHist.map(h => num(h.winSeat0))), dots: true },
        { label: 'seat 1 (second)', color: getColor('--len'), values: smooth(winHist.map(h => num(h.winSeat1))), dots: true },
      ], { pct: true, range: [0, 1],
        tip: 'Meidän AI:n win-rate sen mukaan, kummalla aloituspaikalla se pelasi. Ero = ensisiirtäjän etu. Hyödyllinen sekä korjaamiseen että sen tarkistamiseen, ettei voitto johdu pelkästä paikkaedusta.' }));
    }

    // Tile fraction vs hard (expansion / domination), from the benchmark series.
    if (winHist.length) {
      root.appendChild(chart('Tiles vs hard AI', winHist, [
        { label: 'tileFrac', color: getColor('--tile'), values: smooth(winHist.map(h => num(h.tileFrac))), thick: true, dots: true },
      ], { pct: true, range: [0, 1], hint: 'champ tile fraction at game end', dots: true,
        tip: 'AI:n keskimääräinen ruutuosuus pelin lopussa kovaa bottia vastaan. Nousu = AI oppii laajentamaan ja dominoimaan kartan (ei jää passiiviseksi). Liittyy suoraan win-rateen.' }));
    }
    // Policy loss (cross-entropy vs MCTS visit counts).
    root.appendChild(chart('Policy loss (CE vs MCTS π)', data, [
      { label: 'policyLoss', color: getColor('--mean'), values: scol(data, 'policyLoss') },
    ], { tip: 'Cross-entropy policy-verkon ennusteen ja MCTS-haun käyntimäärien (π) välillä. Lasku = policy oppii jäljittelemään hakua → vahvemmat priorit → vahvempi peli. Lievä nousu on OK kun haku löytää uusia siirtoja.' }));
    // Value loss (MSE vs outcome / shaped target).
    root.appendChild(chart('Value loss (MSE)', data, [
      { label: 'valueLoss', color: getColor('--median'), values: scol(data, 'valueLoss') },
    ], { tip: 'MSE value-verkon asema-arvion ja todellisen lopputuloksen (voitto/tappio) välillä. Lasku = value-verkko ennustaa voittajan tarkemmin → parempi MCTS-leaf-arvio.' }));
    // Replay buffer size.
    root.appendChild(chart('Replay buffer size', data, [
      { label: 'bufferSize', color: getColor('--div'), values: col(data, 'bufferSize') },
    ], { tip: 'Koulutusmuistin koko (self-play-esimerkkejä). Täyttyy kattoon ja pysyy siellä; vanhimmat poistuvat sitä mukaa kun uusia tulee.' }));
    // New self-play examples per iteration.
    root.appendChild(chart('New examples / iter', data, [
      { label: 'newExamples', color: getColor('--wt'), values: col(data, 'newExamples') },
    ], { tip: 'Uusia self-play-päätöksiä kerätty per iteraatio. Vaihtelee pelien pituuden mukaan (pidemmät pelit → enemmän päätöksiä).' }));

    // Self-play tie rate per iteration (spTie / (spTie + spDecisive)). Per-gen,
    // from log.jsonl — updates EVERY iteration (unlike the every-5 benchmark).
    // Older log lines lack spTie/spDecisive → those rows map to null (gap).
    const tieRate = data.map(r => {
      const t = num(r.spTie), d = num(r.spDecisive);
      if (t == null || d == null) return null;
      const tot = t + d;
      return tot > 0 ? t / tot : 0;
    });
    root.appendChild(chart('Tasapelit / iteraatio', data, [
      { label: 'spTie-rate', color: getColor('--bank'), values: smooth(tieRate), thick: true },
    ], { pct: true, range: [0, 1], hint: 'self-play · per iteraatio',
      tip: 'Self-play-pelien tasapeli-osuus per iteraatio: spTie / (spTie + spDecisive). Jokainen iteraatio mitataan (toisin kuin win-rate joka mitataan joka 5.). Korkea osuus = pelit jäätyvät no-progress-katkaisuun ratkaisematta — passiivisuuden merkki. Vanhat lokirivit ilman kenttiä jätetään tyhjäksi.' }));
    // Self-play average game length per iteration (spAvgRounds).
    root.appendChild(chart('Self-play kierrokset / iteraatio', data, [
      { label: 'spAvgRounds', color: getColor('--div'), values: scol(data, 'spAvgRounds') },
    ], { hint: 'self-play · per iteraatio',
      tip: 'Self-play-pelien keskimääräinen pituus (kierroksia) per iteraatio. Lyhyemmät pelit = ratkaisevampi peli. Vanhat lokirivit ilman kenttää jätetään tyhjäksi.' }));

    // PILLAR-6 curriculum gate — per-league-opponent learner win-rate from SELF-PLAY
    // (spVs* fields, per iteration — denser than the every-5 bench). The curriculum now
    // samples Rusher / Fortress / Device / StrongArmy; each maps to its sp counter
    // (null/gap on iters where that bot wasn't drawn). Distinct colour + line style.
    {
      const cseries = [
        { label: 'vsRusher',     color: getColor('--loss'),   values: scol(data, 'spVsRusher'),     thick: true },
        { label: 'vsFortress',   color: getColor('--div'),    values: scol(data, 'spVsFortress'),   dashed: true },
        { label: 'vsDeviceRush', color: getColor('--bank'),   values: scol(data, 'spVsDeviceRush'), dashed: true },
        { label: 'vsStrongArmy', color: getColor('--median'), values: scol(data, 'spVsStrongArmy') },
      ];
      const anyLeague = cseries.some(s => s.values.some(v => v != null));
      // Fall back to showing the legacy curriculum series only if NO new-league data
      // exists yet (pre-Pillar-6 logs), so old runs still render something useful.
      if (!anyLeague) {
        cseries.length = 0;
        cseries.push({ label: 'vsArmyRush (old)', color: getColor('--loss'), values: scol(data, 'spVsArmyRush'), thick: true });
        cseries.push({ label: 'vsDeviceRush', color: getColor('--bank'), values: scol(data, 'spVsDeviceRush') });
      }
      root.appendChild(chart('Curriculum win-rate (self-play vs liiga)', data, cseries,
        { pct: true, range: [0, 1], hint: 'self-play · per iteraatio · vain kun --script-opponents päällä',
        tip: 'Oppijan voittoprosentti uudelleenrakennettuja SD3-liigan opettaja-botteja vastaan SELF-PLAY-peleissä (spVs*-kentät, per iteraatio — tiheämpi kuin joka-5. benchmark). Curriculum samplaa Rusher / Fortress / Device / StrongArmy; aukko = bottia ei nostettu sillä iteraatiolla. Vanhat ajot ilman liigakenttiä putoavat takaisin vanhaan ArmyRush-sarjaan.' }));
    }

    // STEP-2 §1.5 DEFENSE gate — mean tiles the learner LOST to the army-rusher per
    // game. Defined only for army-rush games (else null). The gate wants this TRENDING
    // DOWN as the net learns to garrison the frontier / hold chokepoints.
    root.appendChild(chart('Tiles lost to army-rusher / peli', data, [
      { label: 'tilesLostToRusher', color: getColor('--loss'), values: scol(data, 'tilesLostToRusher'), thick: true },
    ], { hint: 'self-play · per iteraatio · vain army-rush-pelit',
      tip: 'Keskimäärin montako omaa ruutua oppija menetti army-rusherille per peli (valtaukset + HQ-katkaisut). STEP-2 §1.5 puolustusportti: pitää TRENDATA ALAS kun verkko oppii puolustamaan rintamaa ja pitämään kapeikot. Määritelty vain army-rush-peleille; muut rivit tyhjäksi.' }));

    // ★3 Intent histogram (what the AI actually DID). PREFER the latest log
    // line's per-iteration self-play iterIntents (updates EVERY iteration);
    // fall back to the benchmark's intents (every 5 gens) for older data.
    // Bars derive their key set dynamically (new keys render automatically);
    // hovering a bar pops up that intent's history sparkline — sourced from the
    // per-gen iterIntents series when available, else the bench-history.
    const iterIntentsLatest = latest && latest.iterIntents ? latest.iterIntents : null;
    if (iterIntentsLatest) {
      root.appendChild(intentHistogramCard(iterIntentsLatest, fullLog, 'iterIntents',
        'self-play · per iteraatio · osoita palkkia → historia'));
    } else if (latestBench && latestBench.intents) {
      root.appendChild(intentHistogramCard(latestBench.intents, winHistFull, 'intents',
        'uusin benchmark · osoita palkkia → historia'));
    }
    // 5b Rounds-to-resolution per cause — latest benchmark.
    if (latestBench && latestBench.roundsByCause) {
      const rb = latestBench.roundsByCause;
      const rrows = CAUSE_META.map(m => ({ label: m[1], color: m[2], value: num(rb[m[0]]) }))
        .filter(r => r.value != null).map(r => ({ label: r.label, color: r.color, value: r.value, text: r.value.toFixed(1) }));
      root.appendChild(barListCard('Kierroksia ratkaisuun voittotavoittain', 'uusin benchmark', rrows,
        { labelWidth: 150, valWidth: 52, note: 'Device-voitot ovat määritelmällisesti pisimpiä (laskuri); valloitus/konkurssi nopeampia.' }));
    }
  } else {
    // --- GA / neuroevolution charts ----------------------------------------
    root.appendChild(chart('Fitness', data, [
      { label: 'bestFit', color: getColor('--best'), values: scol(data, 'bestFit') },
      { label: 'meanFit', color: getColor('--mean'), values: scol(data, 'meanFit') },
      { label: 'medianFit', color: getColor('--median'), values: scol(data, 'medianFit') },
    ], { band: CTRL.smooth ? null : { center: col(data, 'meanFit'), width: col(data, 'fitStd'), color: getColor('--mean') } }));

    root.appendChild(chart('Champion tile fraction', data, [
      { label: 'championTileFrac', color: getColor('--tile'), values: scol(data, 'championTileFrac') },
    ], { pct: true, range: [0, 1] }));

    root.appendChild(chart('Avg game length (rounds)', data, [
      { label: 'avgGameLen', color: getColor('--len'), values: scol(data, 'avgGameLen') },
    ]));

    root.appendChild(chart('Bankrupt rate', data, [
      { label: 'bankruptRate', color: getColor('--bank'), values: scol(data, 'bankruptRate') },
    ], { pct: true, range: [0, 1] }));

    root.appendChild(chart('Population diversity', data, [
      { label: 'populationDiversity', color: getColor('--div'), values: scol(data, 'populationDiversity') },
    ]));

    root.appendChild(chart('Sigma (annealing)', data, [
      { label: 'sigma', color: getColor('--sigma'), values: col(data, 'sigma') },
      { label: 'wT', color: getColor('--wt'), values: col(data, 'wT') },
    ]));
  }
}

function updateStatus() {
  const el = document.getElementById('status');
  if (!STATE.updated) { el.innerHTML = '<span class="dot">●</span> yhdistetään…'; return; }
  const ageMs = Date.now() - new Date(STATE.updated).getTime();
  const ageS = Math.max(0, Math.round(ageMs / 1000));
  const stale = ageMs > POLL_MS * 3;
  el.className = 'status' + (stale ? ' stale' : '');
  el.innerHTML = '<span class="dot">●</span> ' + (stale ? 'vanhentunut' : 'live') +
    ' — päivitys 5 s välein · päivitetty ' + ageS + ' s sitten';
}

async function poll() {
  try {
    const r = await fetch('/data.json', { cache: 'no-store' });
    STATE = await r.json();
    renderActive();
  } catch (e) {
    // leave last good state; status will go stale
  }
  updateStatus();
}

renderControls();
renderTabsBar();
showTab();
poll();
setInterval(poll, POLL_MS);
setInterval(updateStatus, 1000); // keep "Ns ago" ticking between polls
</script>
</body>
</html>
`;
