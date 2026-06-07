// Live, auto-refreshing AlphaZero training dashboard.
//
// A DEPENDENCY-FREE server (node:http/fs/path/url only) that serves a single
// self-contained HTML page (inline CSS/JS, vanilla, no build, no CDN) polling
// /data.json every 5s and re-rendering in place. Everything is read FRESH per
// request and tolerant of partial/missing files (training writes mid-poll).
//
// Run:
//   npx vite-node training/serve-dashboard.ts -- --dir <checkpoints-dir> --port <n>
//
// Data sources assembled into /data.json (see DATA CATALOGUE):
//   <dir>/benchmark-history.jsonl  authoritative champion-vs-HARD + league record
//   <dir>/log.jsonl                per-iteration training/self-play telemetry
//   <dir>/replay*.json             game replays for the playback viewer
//   <dir>/spatial.json             CNN policy/value introspection snapshots
//   <dir>/champion.json            current champion sidecar
//   models/registry.jsonl          model registry (repo root)
//   models/CHAMPION.json           champion/deploy pointers (repo root)
//   build-status.json/build-log.jsonl  build-process feed (repo root)

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
function readJsonl(path: string): { rows: Record<string, unknown>[]; mtime: string | null } {
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
  const rows: Record<string, unknown>[] = [];
  for (const line of raw.split('\n')) {
    const s = line.trim();
    if (!s) continue;
    try {
      rows.push(JSON.parse(s));
    } catch {
      // skip malformed line (e.g. partial/incomplete final line mid-write)
    }
  }
  return { rows, mtime };
}

function readJsonSafe(path: string): unknown | null {
  if (!existsSync(path)) return null;
  try {
    return JSON.parse(readFileSync(path, 'utf8'));
  } catch {
    return null;
  }
}
function readJsonlSafe(path: string): Record<string, unknown>[] {
  return readJsonl(path).rows;
}
function readText(path: string): string | null {
  if (!existsSync(path)) return null;
  try {
    return readFileSync(path, 'utf8');
  } catch {
    return null;
  }
}

// Dedupe rows by `gen` (keeping the LAST occurrence — the most recent run's value)
// and sort ascending, so a checkpoint dir that accidentally holds lines from two
// runs still renders a clean, monotonic series.
function dedupeByGen(rows: Record<string, unknown>[]): Record<string, unknown>[] {
  const byGen = new Map<number, Record<string, unknown>>();
  for (const r of rows) {
    const g = Number((r as { gen?: unknown }).gen);
    if (Number.isFinite(g)) byGen.set(g, r);
  }
  return [...byGen.keys()].sort((a, b) => a - b).map((g) => byGen.get(g) as Record<string, unknown>);
}

// Opponent label table shared by replay sources + league panels. Each entry:
//   [srcKey, replayDataKey, label, isLegacy]
// srcKey doubles as the file-suffix (replay_vs_<srcKey>.json) for non-legacy.
const OPPONENTS = [
  ['hard', 'replay', 'Hard CPU', false],
  ['self', 'replaySelf', 'Self-play', false],
  ['rusher', 'replayVsRusher', 'Rusher', false],
  ['strongarmy', 'replayVsStrongArmy', 'StrongArmy', false],
  ['fortress', 'replayVsFortress', 'Fortress', false],
  ['devicerush', 'replayVsDeviceRush', 'DeviceRush', false],
  ['armyrush', 'replayVsArmyRush', 'ArmyRush', true],
  ['garrison', 'replayVsGarrison', 'Garrison', true],
  ['hqrush', 'replayVsHqRush', 'HqRush', true],
  ['marcher', 'replayVsMarcher', 'Marcher', true],
  ['expert', 'replayVsExpert', 'Expert', true],
] as const;

function buildData(dir: string): Record<string, unknown> {
  const lg = readJsonl(join(dir, 'log.jsonl'));
  const rows = dedupeByGen(lg.rows) as LogRow[];
  const winHistory = dedupeByGen(readJsonlSafe(join(dir, 'benchmark-history.jsonl')));
  const replays: Record<string, unknown> = {};
  for (const [srcKey, dataKey] of OPPONENTS) {
    if (srcKey === 'hard') replays[dataKey] = readJsonSafe(join(dir, 'replay.json'));
    else if (srcKey === 'self') replays[dataKey] = readJsonSafe(join(dir, 'replay_selfplay.json'));
    else replays[dataKey] = readJsonSafe(join(dir, `replay_vs_${srcKey}.json`));
  }
  return {
    dir,
    updated: new Date().toISOString(),
    log: rows,
    winHistory,
    benchmark: readJsonSafe(join(dir, 'benchmark.json')), // legacy single-latest sidecar
    champion: readJsonSafe(join(dir, 'champion.json')),
    ...replays,
    // CNN spatial heatmap (policy/value introspection). null for non-CNN arcs.
    spatial: readJsonSafe(join(dir, 'spatial.json')),
    latest: rows.length ? rows[rows.length - 1] : null,
    benchLatest: winHistory.length ? winHistory[winHistory.length - 1] : null,
    logMtime: lg.mtime,
    // Repo-root artifacts.
    buildStatus: readJsonSafe(join(REPO_ROOT, 'build-status.json')),
    buildLog: readJsonlSafe(join(REPO_ROOT, 'build-log.jsonl')),
    registry: readJsonlSafe(join(REPO_ROOT, 'models', 'registry.jsonl')),
    championPtr: readJsonSafe(join(REPO_ROOT, 'models', 'CHAMPION.json')),
    research: [
      { id: 'research', title: 'Tutkimus', md: readText(join(REPO_ROOT, 'rust-trainer', 'TRAINING-RESEARCH.md')) },
      { id: 'design', title: 'AlphaZero-suunnitelma', md: readText(join(REPO_ROOT, 'rust-trainer', 'ALPHAZERO-DESIGN.md')) },
      { id: 'reward', title: 'Palkkiosignaalit', md: readText(join(REPO_ROOT, 'rust-trainer', 'REWARD-DESIGN.md')) },
    ].filter((d) => d.md != null),
  };
}

function emptyData(dir: string): Record<string, unknown> {
  const d: Record<string, unknown> = {
    dir,
    updated: new Date().toISOString(),
    log: [],
    winHistory: [],
    benchmark: null,
    champion: null,
    spatial: null,
    latest: null,
    benchLatest: null,
    logMtime: null,
    buildStatus: null,
    buildLog: [],
    registry: [],
    championPtr: null,
    research: [],
  };
  for (const [, dataKey] of OPPONENTS) d[dataKey] = null;
  return d;
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
      body = JSON.stringify(emptyData(rawDir));
    }
    res.writeHead(200, { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store' });
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
  console.log(`AlphaZero training dashboard serving ${DIR}`);
  console.log(`  http://127.0.0.1:${port}/`);
});

// ===========================================================================
// The page. Self-contained: inline CSS + vanilla JS, no build step, no CDN.
// Polls /data.json every 5s and re-renders the active tab in place.
// ===========================================================================
const PAGE = /* html */ `<!doctype html>
<html lang="fi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AZ training</title>
<style>
:root{
  --bg:#0b0f14; --bg2:#0e131a; --panel:#141b22; --panel2:#0f151b;
  --grid:#222b35; --line:#2a3540;
  --ink:#d7e0e8; --muted:#7a8794; --faint:#4a5560;
  --good:#4dd2a0; --raw:#5aa9ff; --illusion:#ff9e64; --bad:#ff6b6b;
  --tie:#8b97a3; --mil:#c792ea; --econ:#7fdbff; --accent:#4dd2a0;
}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--ink);
  font-family:ui-monospace,"JetBrains Mono",Menlo,Consolas,monospace;
  font-size:13px;line-height:1.45;font-variant-numeric:tabular-nums}
a{color:var(--raw);text-decoration:none}
a:hover{text-decoration:underline}
code{color:var(--muted)}
.wrap{max-width:1480px;margin:0 auto;padding:0 16px 60px}

/* ---- sticky header ---- */
header{position:sticky;top:0;z-index:30;background:var(--bg);
  border-bottom:1px solid var(--line);backdrop-filter:blur(2px)}
.hdr{max-width:1480px;margin:0 auto;padding:8px 16px}
.hrow{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.dot{width:9px;height:9px;border-radius:50%;background:var(--good);flex:0 0 auto}
.dot.stale{background:var(--illusion)} .dot.dead{background:var(--bad)}
.brand{font-weight:700;letter-spacing:.02em}
.brand .sub{color:var(--muted);font-weight:400}
.hmeta{color:var(--muted);font-size:11px;margin-left:auto;text-align:right}
.hmeta b{color:var(--ink)}
.kstrip{display:flex;gap:10px;flex-wrap:wrap;margin-top:6px;padding-top:6px;border-top:1px solid var(--line)}
.kpi{min-width:130px;background:var(--panel2);border:1px solid var(--line);border-radius:6px;
  padding:6px 10px;display:flex;flex-direction:column;gap:1px;transition:border-color .12s}
.kpi.flash-up{border-color:var(--good)} .kpi.flash-dn{border-color:var(--bad)}
.kpi .lbl{font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}
.kpi .val{font-size:20px;font-weight:700;display:flex;align-items:baseline;gap:6px}
.kpi .val .d{font-size:11px;font-weight:600}
.kpi .spark{height:16px;margin-top:1px}
.kpi.honest .val{color:var(--good)} .kpi.raw .val{color:var(--raw)}
.kpi.tie .val{color:var(--tie)} .kpi.thru .val{color:var(--ink)}
.up{color:var(--good)} .dn{color:var(--bad)}
.gapnote{font-size:10px;color:var(--illusion);align-self:center}

/* ---- tabs (single source of truth) ---- */
.tabs{display:flex;gap:2px;align-items:center;margin:14px 0 6px;
  border-bottom:1px solid var(--line);overflow-x:auto;white-space:nowrap}
.tabs .tab{background:none;border:none;color:var(--muted);font:inherit;
  padding:7px 12px;cursor:pointer;border-bottom:2px solid transparent;min-height:32px}
.tabs .tab:hover{color:var(--ink)}
.tabs .tab.on{color:var(--accent);border-bottom-color:var(--accent);font-weight:600}
.tabs .sp{margin-left:auto}
.tabs .div{color:var(--faint);padding:0 4px}

/* ---- segmented control (NEVER a .tab) ---- */
.segmented{display:inline-flex;background:var(--panel2);border:1px solid var(--line);
  border-radius:6px;padding:2px;gap:2px;flex-wrap:wrap}
.segmented .seg{background:none;border:none;color:var(--muted);font:inherit;
  padding:4px 9px;border-radius:4px;cursor:pointer;min-height:30px;font-size:12px}
.segmented .seg:hover{color:var(--ink)}
.segmented .seg.sel{background:var(--panel);color:var(--ink);font-weight:600}
.segmented .seg.legacy{opacity:.6}
.seglbl{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.06em;align-self:center;padding:0 6px}

/* ---- toolbar ---- */
.toolbar{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin:0 0 12px;
  color:var(--muted);font-size:11px}
.toolbar .segmented .seg{padding:3px 8px;min-height:26px}

/* ---- panels & cards ---- */
[hidden]{display:none!important}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:12px}
.card.wide{grid-column:1 / -1}
@media(min-width:1200px){.card.span2{grid-column:span 2}}
.card h3{margin:0 0 8px;font-size:11px;text-transform:uppercase;letter-spacing:.08em;
  color:var(--muted);font-weight:600;display:flex;align-items:center;gap:6px}
.card h3 .tip{color:var(--faint);cursor:help;font-weight:400}
.well{background:var(--panel2);border:1px solid var(--line);border-radius:6px;
  box-shadow:inset 0 1px 0 #0008;padding:10px}
.empty{color:var(--faint);font-size:12px;padding:18px;text-align:center}
.leg{margin-top:6px;color:var(--muted);font-size:10.5px;display:flex;gap:10px;flex-wrap:wrap}
.leg i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:4px;vertical-align:-1px}
svg{display:block;width:100%}
.dim{color:var(--muted)} .mono{font-variant-numeric:tabular-nums}

/* ---- KPI tile mini ---- */
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px}
.tile{background:var(--panel2);border:1px solid var(--line);border-radius:6px;padding:8px 10px}
.tile .lbl{font-size:10px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
.tile .v{font-size:18px;font-weight:700}
.tile .v.zero{color:var(--bad)}
.tile .sub{font-size:10px;color:var(--faint)}

/* ---- funnel ---- */
.funnel{display:flex;align-items:stretch;gap:0;flex-wrap:wrap}
.fstage{flex:1;min-width:120px;display:flex;flex-direction:column;align-items:center;gap:4px;padding:0 4px}
.fstage .bar{width:100%;background:var(--panel2);border:1px solid var(--line);border-radius:4px;
  display:flex;align-items:flex-end;height:120px;overflow:hidden}
.fstage .fill{width:100%;background:var(--mil);border-radius:0 0 3px 3px;min-height:2px;transition:height .2s}
.fstage.bottleneck .bar{border-color:var(--illusion);box-shadow:0 0 0 1px var(--illusion)}
.fstage .nm{font-size:10px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
.fstage .ct{font-size:15px;font-weight:700;color:var(--mil)}
.fstage .ct.z{color:var(--bad)}
.farrow{display:flex;align-items:center;color:var(--faint);font-size:18px;padding:0 2px}
.fdrop{font-size:10px;color:var(--illusion);text-align:center}

/* ---- bars ---- */
.hbar{display:grid;grid-template-columns:120px 1fr 56px;gap:8px;align-items:center;margin:3px 0;font-size:12px}
.hbar .track{display:block;background:var(--panel2);border:1px solid var(--line);border-radius:3px;height:16px;position:relative;overflow:hidden}
.hbar .fill{display:block;height:100%;border-radius:0}
.hbar .ref{position:absolute;top:0;bottom:0;width:1px;background:var(--faint)}
.hbar .n{text-align:right;color:var(--muted);font-size:11px}
.hbar .nm{color:var(--ink);overflow:hidden;text-overflow:ellipsis}

/* ---- stacked bar ---- */
.sbar{display:flex;height:22px;border:1px solid var(--line);border-radius:4px;overflow:hidden;background:var(--panel2)}
.sbar div{display:flex;align-items:center;justify-content:center;font-size:10px;color:#0b0f14;font-weight:700;min-width:0}

/* ---- table ---- */
table{width:100%;border-collapse:collapse;font-size:12px}
th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.06em;font-weight:600}
tbody tr:nth-child(odd){background:var(--bg2)}
tr.champ{box-shadow:inset 3px 0 0 var(--good)}
tr.oldarc td{color:var(--faint)}
.arch{display:inline-block;padding:1px 6px;border:1px solid var(--line);border-radius:10px;font-size:10px;color:var(--muted)}
.pill{display:inline-block;padding:1px 6px;border-radius:10px;font-size:10px;font-weight:700}
.pill.dep{background:rgba(77,210,160,.18);color:var(--good)}
.pill.exp{background:rgba(122,135,148,.18);color:var(--muted)}
.tblwrap{overflow-x:auto}

/* ---- replay / spatial ---- */
.stage{display:flex;gap:14px;flex-wrap:wrap;align-items:flex-start}
.stage canvas{background:var(--panel2);border:1px solid var(--line);border-radius:6px;max-width:100%;height:auto;image-rendering:pixelated}
.side{flex:1;min-width:220px;font-size:12px;color:var(--muted);line-height:1.6}
.side .big{color:var(--ink);font-size:13px;font-weight:700;margin-bottom:6px}
.ctl{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:10px}
.ctl input[type=range]{flex:1;min-width:140px;accent-color:var(--accent)}
.btn{background:var(--panel2);border:1px solid var(--line);color:var(--ink);font:inherit;
  padding:4px 10px;border-radius:4px;cursor:pointer;min-height:30px}
.btn:hover{border-color:var(--accent)} .btn.on{border-color:var(--accent);color:var(--accent)}
.blue{color:var(--raw)} .red{color:var(--bad)}
.ichip{display:inline-block;padding:0 5px;border-radius:3px;font-size:10px;margin:1px}
.note{font-size:11px;padding:6px 8px;border:1px solid var(--line);border-radius:4px;background:var(--panel2)}
.topmoves{margin-top:8px;font-size:11px}
.topmoves .tm{display:grid;grid-template-columns:1fr auto auto;gap:8px;padding:2px 0;border-bottom:1px solid var(--line)}

/* ---- build / research ---- */
.phase{display:flex;align-items:center;gap:8px;padding:5px 0;font-size:12px}
.phase .st{width:10px;height:10px;border-radius:50%;flex:0 0 auto;background:var(--faint)}
.phase .st.active{background:var(--raw)} .phase .st.done{background:var(--good)}
.feed{max-height:380px;overflow:auto;font-size:11.5px}
.feed .row{padding:3px 0;border-bottom:1px solid var(--line)}
.feed .ts{color:var(--faint);font-size:10px}
.md{font-size:13px;line-height:1.6}
.md h1,.md h2,.md h3{color:var(--ink);margin:14px 0 6px}
.md h1{font-size:18px} .md h2{font-size:15px} .md h3{font-size:13px}
.md code{background:var(--panel2);padding:1px 4px;border-radius:3px}
.md pre{background:var(--panel2);border:1px solid var(--line);border-radius:4px;padding:10px;overflow:auto}
.md ul{padding-left:18px}
.banner{display:none;position:sticky;top:0;z-index:40;background:var(--bad);color:#0b0f14;
  text-align:center;font-size:11px;font-weight:700;padding:3px}
.banner.show{display:block}
@media(prefers-reduced-motion:reduce){.kpi,.fstage .fill{transition:none}}
@media(max-width:900px){.kstrip{overflow-x:auto;flex-wrap:nowrap}.kpi{min-width:120px}}
</style>
</head>
<body>
<div class="banner" id="banner">yhteys katkesi — yritetään uudelleen…</div>
<header>
  <div class="hdr">
    <div class="hrow">
      <span class="dot" id="statusDot" title="last /data.json OK"></span>
      <span class="brand">colonizing-pirkanmaa <span class="sub">· AZ TRAINING</span></span>
      <span id="runInfo" class="dim" style="font-size:12px"></span>
      <span class="hmeta" id="hmeta"></span>
    </div>
    <div class="kstrip" id="kstrip"></div>
  </div>
</header>
<div class="wrap">
  <nav class="tabs" id="tabs"></nav>
  <div class="toolbar" id="toolbar"></div>
  <section data-panel="overview"></section>
  <section data-panel="economy" hidden></section>
  <section data-panel="military" hidden></section>
  <section data-panel="opponents" hidden></section>
  <section data-panel="replay" hidden></section>
  <section data-panel="spatial" hidden></section>
  <section data-panel="models" hidden></section>
  <section data-panel="build" hidden></section>
  <section data-panel="research" hidden></section>
</div>
<script>
"use strict";
/* ===================== state ===================== */
var STATE = null, PREV_KPI = {}, LAST_OK = 0;
var TABS = [
  ['overview','Overview'],['economy','Economy'],['military','Military'],
  ['opponents','Opponents'],['replay','Replay'],['spatial','Spatial'],['models','Models'],
  ['__div__',''],['build','Build'],['research','Research'],
];
var TAB = (location.hash || '').replace('#','') || localStorage.getItem('cp.dash.tab') || 'overview';
if(!TABS.some(function(t){return t[0]===TAB;})) TAB='overview';
var CTRL = { win: 0, smooth: false }; // win 0 = all
var RES_TAB = 0;
var POLL_MS = 5000;

/* ===================== helpers ===================== */
function num(x){ var n = (x===null||x===undefined||x==='') ? NaN : Number(x); return isFinite(n)?n:null; }
function pct(x,d){ var n=num(x); return n==null?'—':(100*n).toFixed(d==null?1:d)+'%'; }
function f2(x,d){ var n=num(x); return n==null?'—':n.toFixed(d==null?2:d); }
function esc(s){ return String(s==null?'':s).replace(/[&<>"]/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c];}); }
function timeAgo(iso){ if(!iso) return '—'; var t=Date.parse(iso); if(isNaN(t)) return '—';
  var s=Math.max(0,Math.round((Date.now()-t)/1000));
  if(s<60) return s+' s sitten'; if(s<3600) return Math.round(s/60)+' min sitten';
  if(s<86400) return Math.round(s/3600)+' h sitten'; return Math.round(s/86400)+' pv sitten'; }
function lastN(arr){ if(CTRL.win<=0||!arr) return arr; return arr.slice(Math.max(0,arr.length-CTRL.win)); }
function smooth(vals){ if(!CTRL.smooth) return vals; var w=Math.max(2,Math.round(vals.length/24)); var out=[];
  for(var i=0;i<vals.length;i++){ var a=Math.max(0,i-w),s=0,n=0; for(var j=a;j<=i;j++){ if(vals[j]!=null){s+=vals[j];n++;} } out.push(n?s/n:null);} return out; }
function field(rows,key){ return rows.map(function(r){ return num(r[key]); }); }
function has(rows,key){ return rows.some(function(r){ return num(r[key])!=null; }); }
function tip(t){ return '<span class="tip" title="'+esc(t)+'">ⓘ</span>'; }

var C = { good:'#4dd2a0', raw:'#5aa9ff', illusion:'#ff9e64', bad:'#ff6b6b',
  tie:'#8b97a3', mil:'#c792ea', econ:'#7fdbff', grid:'#222b35', muted:'#7a8794', faint:'#4a5560' };

/* ---- SVG line/band chart ----
   series: [{vals,color,label,axis?,dash?,dots?}], opts: {h, band:{lo,hi,color}, y0,y1, pctY, rightLabel} */
function chart(xs, series, opts){
  opts = opts||{};
  var H = opts.h||190, W=600, padL=40, padR=opts.rightAxis?40:10, padT=8, padB=18;
  var n = xs.length;
  if(!n){ return '<div class="empty">ei dataa</div>'; }
  var allv=[]; series.forEach(function(s){ if(s.axis!=='r') s.vals.forEach(function(v){ if(v!=null) allv.push(v);}); });
  if(opts.band){ opts.band.lo.concat(opts.band.hi).forEach(function(v){ if(v!=null) allv.push(v); }); }
  var y0 = opts.y0!=null?opts.y0:(allv.length?Math.min.apply(null,allv):0);
  var y1 = opts.y1!=null?opts.y1:(allv.length?Math.max.apply(null,allv):1);
  if(y0===y1){ y1=y0+1; }
  var rv=[]; series.forEach(function(s){ if(s.axis==='r') s.vals.forEach(function(v){ if(v!=null) rv.push(v);});});
  var r0=rv.length?Math.min.apply(null,rv):0, r1=rv.length?Math.max.apply(null,rv):1; if(r0===r1) r1=r0+1;
  var x0=xs[0], x1=xs[n-1]; if(x0===x1) x1=x0+1;
  function X(v){ return padL + (v-x0)/(x1-x0)*(W-padL-padR); }
  function Y(v){ return padT + (1-(v-y0)/(y1-y0))*(H-padT-padB); }
  function YR(v){ return padT + (1-(v-r0)/(r1-r0))*(H-padT-padB); }
  var s='<svg viewBox="0 0 '+W+' '+H+'" preserveAspectRatio="none">';
  for(var g=0;g<=4;g++){ var gy=padT+g/4*(H-padT-padB); var gv=y1-(g/4)*(y1-y0);
    s+='<line x1="'+padL+'" y1="'+gy+'" x2="'+(W-padR)+'" y2="'+gy+'" stroke="'+C.grid+'" stroke-width="1"/>';
    s+='<text x="2" y="'+(gy+3)+'" fill="'+C.muted+'" font-size="10">'+(opts.pctY?Math.round(gv*100)+'%':(Math.abs(gv)<10?gv.toFixed(2):Math.round(gv)))+'</text>'; }
  if(opts.band){ var pts=''; for(var i=0;i<n;i++){ if(opts.band.hi[i]!=null) pts+=X(xs[i])+','+Y(opts.band.hi[i])+' '; }
    for(var k=n-1;k>=0;k--){ if(opts.band.lo[k]!=null) pts+=X(xs[k])+','+Y(opts.band.lo[k])+' '; }
    if(pts) s+='<polygon points="'+pts+'" fill="'+opts.band.color+'" opacity="0.18"/>'; }
  series.forEach(function(ser){
    var d='', open=false, YY = ser.axis==='r'?YR:Y;
    for(var i=0;i<n;i++){ var v=ser.vals[i]; if(v==null){ open=false; continue; }
      d += (open?'L':'M')+X(xs[i]).toFixed(1)+' '+YY(v).toFixed(1)+' '; open=true; }
    if(d) s+='<path d="'+d+'" fill="none" stroke="'+ser.color+'" stroke-width="'+(ser.w||1.6)+'"'+(ser.dash?' stroke-dasharray="4 3"':'')+'/>';
    if(ser.dots){ for(var j=0;j<n;j++){ if(ser.vals[j]!=null) s+='<circle cx="'+X(xs[j]).toFixed(1)+'" cy="'+YY(ser.vals[j]).toFixed(1)+'" r="2" fill="'+ser.color+'"/>'; } }
  });
  s+='</svg>';
  var lg = series.filter(function(s){return s.label;}).map(function(s){ return '<span><i style="background:'+s.color+'"></i>'+esc(s.label)+'</span>'; }).join('');
  return s + (lg?'<div class="leg">'+lg+'</div>':'');
}

/* ---- normalized stacked area ----
   layers: [{key|vals, color, label}], built from rows. */
function stackedArea(xs, layers, opts){
  opts=opts||{}; var H=opts.h||190, W=600, padL=40, padR=10, padT=8, padB=18, n=xs.length;
  if(!n) return '<div class="empty">ei dataa</div>';
  var x0=xs[0],x1=xs[n-1]; if(x0===x1)x1=x0+1;
  function X(v){ return padL+(v-x0)/(x1-x0)*(W-padL-padR); }
  function Y(v){ return padT+(1-v)*(H-padT-padB); }
  // normalize per-x
  var tot=new Array(n).fill(0);
  for(var i=0;i<n;i++){ layers.forEach(function(l){ var v=l.vals[i]; if(v!=null&&v>0) tot[i]+=v; }); }
  var s='<svg viewBox="0 0 '+W+' '+H+'" preserveAspectRatio="none">';
  for(var g=0;g<=4;g++){ var gy=padT+g/4*(H-padT-padB);
    s+='<line x1="'+padL+'" y1="'+gy+'" x2="'+(W-padR)+'" y2="'+gy+'" stroke="'+C.grid+'"/>';
    s+='<text x="2" y="'+(gy+3)+'" fill="'+C.muted+'" font-size="10">'+(100-g*25)+'%</text>'; }
  var base=new Array(n).fill(0);
  layers.forEach(function(l){
    var top=[]; for(var i=0;i<n;i++){ var v=(l.vals[i]!=null&&tot[i]>0)?l.vals[i]/tot[i]:0; top.push(base[i]+v); }
    var pts=''; for(var i2=0;i2<n;i2++) pts+=X(xs[i2])+','+Y(top[i2])+' ';
    for(var k=n-1;k>=0;k--) pts+=X(xs[k])+','+Y(base[k])+' ';
    s+='<polygon points="'+pts+'" fill="'+l.color+'" opacity="0.78"/>';
    base=top;
  });
  s+='</svg>';
  var lg=layers.map(function(l){return '<span><i style="background:'+l.color+'"></i>'+esc(l.label)+'</span>';}).join('');
  return s+'<div class="leg">'+lg+'</div>';
}

/* ---- horizontal bars with optional 50% ref tick ---- */
function hbars(items, opts){
  opts=opts||{}; var max=opts.max!=null?opts.max:1;
  if(!items.length) return '<div class="empty">ei dataa</div>';
  return items.map(function(it){
    var v=it.val==null?0:it.val, w=Math.max(0,Math.min(1,v/max))*100;
    var col=it.color||(it.warn?C.illusion:C.good);
    var ref=opts.ref!=null?'<span class="ref" style="left:'+(opts.ref/max*100)+'%"></span>':'';
    return '<div class="hbar"><span class="nm" title="'+esc(it.name)+'">'+esc(it.name)+'</span>'
      +'<span class="track"><span class="fill" style="width:'+w.toFixed(1)+'%;background:'+col+'"></span>'+ref+'</span>'
      +'<span class="n">'+esc(it.right!=null?it.right:(opts.pctRight?pct(v):f2(v)))+'</span></div>';
  }).join('');
}

/* ---- stacked single bar (W/L/T etc.) ---- */
function stackBar(parts){
  var tot=parts.reduce(function(a,p){return a+(p.v>0?p.v:0);},0)||1;
  return '<div class="sbar">'+parts.map(function(p){ var w=p.v>0?p.v/tot*100:0;
    return w<0.5?'':'<div style="width:'+w.toFixed(1)+'%;background:'+p.color+'" title="'+esc(p.label)+' '+f2(p.v)+'">'+(w>9?Math.round(w)+'%':'')+'</div>';
  }).join('')+'</div>';
}

/* ---- mini sparkline ---- */
function spark(vals, color, h){
  h=h||16; var W=120, n=vals.length; var clean=vals.filter(function(v){return v!=null;});
  if(clean.length<2) return '';
  var mn=Math.min.apply(null,clean), mx=Math.max.apply(null,clean); if(mn===mx)mx=mn+1;
  var d='',open=false;
  for(var i=0;i<n;i++){ var v=vals[i]; if(v==null){open=false;continue;}
    var x=n>1?i/(n-1)*W:0, y=h-2-((v-mn)/(mx-mn))*(h-4);
    d+=(open?'L':'M')+x.toFixed(1)+' '+y.toFixed(1)+' '; open=true; }
  return '<svg class="spark" viewBox="0 0 '+W+' '+h+'" preserveAspectRatio="none"><path d="'+d+'" fill="none" stroke="'+(color||C.good)+'" stroke-width="1.4"/></svg>';
}

/* ---- markdown (dependency-free, minimal) ---- */
function md(src){
  if(!src) return '<div class="empty">ei dokumenttia</div>';
  var lines=src.split('\\n'), out=[], inCode=false, inUl=false;
  function closeUl(){ if(inUl){ out.push('</ul>'); inUl=false; } }
  for(var i=0;i<lines.length;i++){ var ln=lines[i];
    if(/^\`\`\`/.test(ln)){ if(inCode){ out.push('</pre>'); inCode=false; } else { closeUl(); out.push('<pre>'); inCode=true; } continue; }
    if(inCode){ out.push(esc(ln)); continue; }
    var m;
    if((m=ln.match(/^(#{1,3})\\s+(.*)/))){ closeUl(); out.push('<h'+m[1].length+'>'+inline(m[2])+'</h'+m[1].length+'>'); continue; }
    if((m=ln.match(/^\\s*[-*]\\s+(.*)/))){ if(!inUl){out.push('<ul>');inUl=true;} out.push('<li>'+inline(m[1])+'</li>'); continue; }
    if(/^\\s*$/.test(ln)){ closeUl(); continue; }
    closeUl(); out.push('<p>'+inline(ln)+'</p>');
  }
  if(inCode) out.push('</pre>'); closeUl();
  return out.join('');
  function inline(t){ return esc(t).replace(/\`([^\`]+)\`/g,'<code>$1</code>').replace(/\\*\\*([^*]+)\\*\\*/g,'<b>$1</b>'); }
}

/* ===================== tab system (single source of truth) ===================== */
function renderTabs(){
  var html=TABS.map(function(t){
    if(t[0]==='__div__') return '<span class="div sp">·</span>';
    return '<button class="tab" data-tab="'+t[0]+'">'+esc(t[1])+'</button>';
  }).join('');
  document.getElementById('tabs').innerHTML=html;
  document.querySelectorAll('#tabs .tab').forEach(function(b){
    b.onclick=function(){ setTab(b.dataset.tab); };
  });
}
function setTab(id){
  TAB=id; localStorage.setItem('cp.dash.tab', id); location.hash=id;
  document.querySelectorAll('#tabs .tab').forEach(function(b){ b.classList.toggle('on', b.dataset.tab===TAB); });
  document.querySelectorAll('[data-panel]').forEach(function(p){ p.hidden = (p.dataset.panel!==TAB); });
  renderActive();
}

/* shared toolbar (window + smooth), applies to time-series panels */
function renderToolbar(){
  var wins=[[0,'All'],[200,'200'],[100,'100'],[50,'50'],[25,'25']];
  var seg=wins.map(function(w){ return '<button class="seg'+(CTRL.win===w[0]?' sel':'')+'" data-win="'+w[0]+'">'+w[1]+'</button>'; }).join('');
  document.getElementById('toolbar').innerHTML=
    '<span class="seglbl">ikkuna</span><span class="segmented">'+seg+'</span>'
    +'<span class="segmented"><button class="seg'+(CTRL.smooth?' sel':'')+'" id="smoothBtn">smooth</button></span>'
    +'<span class="dim" id="genRange"></span>';
  document.querySelectorAll('[data-win]').forEach(function(b){ b.onclick=function(){ CTRL.win=Number(b.dataset.win); renderToolbar(); renderActive(); }; });
  document.getElementById('smoothBtn').onclick=function(){ CTRL.smooth=!CTRL.smooth; renderToolbar(); renderActive(); };
  var log=(STATE&&STATE.log)||[];
  if(log.length){ var w=lastN(log); document.getElementById('genRange').textContent='gen '+w[0].gen+'–'+w[w.length-1].gen+' · '+w.length+' it.'; }
}

/* ===================== header ===================== */
function renderHeader(){
  var s=STATE||{};
  var champ = championId();
  var latest = s.latest||{}, bench = s.benchLatest||{};
  document.getElementById('runInfo').textContent = (s.dir? ('run: '+String(s.dir).split('/').pop()+'  ') : '')
    + (latest.gen!=null? ('gen '+latest.gen) : '');
  document.getElementById('hmeta').innerHTML =
    'champion: <b>'+esc(champ||'—')+'</b>'
    + (s.champion&&s.champion.git_commit? ' · git '+esc(String(s.champion.git_commit).slice(0,7)) : '')
    + ' · päivitetty '+timeAgo(s.updated);

  // KPI strip — the always-true numbers.
  var hist=s.winHistory||[];
  var trueSpark = spark(hist.map(function(h){return num(h.trueWinVsHard);}), C.good);
  var honest = bench.trueWinVsHard, raw = bench.winRate, tie = num(latest.spTie), gps = latest.gamesPerSec;
  // tie as fraction of self-play games this iter
  var spDec=num(latest.spDecisive), spTie=num(latest.spTie);
  var tieFrac = (spTie!=null&&spDec!=null&&(spTie+spDec)>0)? spTie/(spTie+spDec) : null;
  function kpi(cls,id,lbl,valHtml,extra){ return '<div class="kpi '+cls+'" id="kpi-'+id+'"><span class="lbl">'+lbl+'</span>'
    +'<span class="val">'+valHtml+'</span>'+(extra||'')+'</div>'; }
  document.getElementById('kstrip').innerHTML =
    kpi('honest','honest','Honest win '+tip('trueWinVsHard — voitot pl. vastustajan konkurssi (mirage). Tämä on mittatikku.'),
        '<span id="v-honest">'+pct(honest)+'</span><span class="d" id="d-honest"></span>', trueSpark)
    + kpi('raw','raw','Raw win '+tip('winRate — sis. konkurssivoitot. Ero honestiin = mirage.'),
        '<span id="v-raw">'+pct(raw)+'</span><span class="d" id="d-raw"></span>'
        + '<div class="gapnote">mirage '+(honest!=null&&raw!=null?('+'+(100*(raw-honest)).toFixed(1)+'pp'):'—')+'</div>')
    + kpi('tie','tie','Self-play tie '+tip('spTie / (spTie+spDecisive) — draw-attractor vahti.'),
        '<span id="v-tie">'+(tieFrac!=null?pct(tieFrac):'—')+'</span>')
    + kpi('thru','thru','Läpäisy','<span id="v-thru">'+(gps!=null?f2(gps,3):'—')+'</span><span class="lbl" style="font-size:9px">g/s</span>');
  flashKpi('honest', honest); flashKpi('raw', raw);
}
function flashKpi(id, v){
  var el=document.getElementById('kpi-'+id), d=document.getElementById('d-'+id); if(!el) return;
  var prev=PREV_KPI[id];
  if(prev!=null && v!=null && Math.abs(v-prev)>1e-6){
    var up=v>prev; el.classList.add(up?'flash-up':'flash-dn');
    if(d) d.innerHTML='<span class="'+(up?'up':'dn')+'">'+(up?'▲':'▼')+(100*Math.abs(v-prev)).toFixed(1)+'</span>';
    setTimeout(function(){ el.classList.remove('flash-up','flash-dn'); }, 900);
  }
  if(v!=null) PREV_KPI[id]=v;
}

/* ===================== OVERVIEW ===================== */
function panelOverview(){
  var s=STATE; var log=lastN(s.log||[]), hist=lastN(s.winHistory||[]);
  var cards=[];
  // ★ headline win-rate (honest vs raw + illusion band)
  if(hist.length){
    var hx=hist.map(function(h){return h.gen;});
    var trueV=smooth(hist.map(function(h){return num(h.trueWinVsHard);}));
    var rawV=smooth(hist.map(function(h){return num(h.winRate);}));
    cards.push(card('★ Win-rate vs HARD — honest vs raw','span2 wide',
      chart(hx,[
        {vals:rawV,color:C.raw,label:'raw (sis. konkurssi)',w:1.4},
        {vals:trueV,color:C.good,label:'honest (trueWin)',w:2.4,dots:hist.length<40}
      ],{h:220,pctY:true,y0:0,y1:1,band:CTRL.smooth?null:{lo:trueV,hi:rawV,color:C.illusion}}),
      'Vihreä = rehellinen voittoaste; sininen = raaka (sis. konkurssimirage). Amber-vyö = ero.'));
  } else cards.push(card('★ Win-rate vs HARD','span2 wide','<div class="empty">ei benchmark-historiaa vielä</div>'));

  // outcome composition (champ wins by cause + loss + tie) — from benchmark champWins/hardWins
  if(hist.length && hist.some(function(h){return h.champWins;})){
    var ox=hist.map(function(h){return h.gen;});
    function cw(h,k){ return h.champWins&&h.champWins[k]!=null?h.champWins[k]:0; }
    var L={device:[],domination:[],conquest:[],tiebreak:[],loss:[]};
    hist.forEach(function(h){ var ng=num(h.nGames)||1;
      L.device.push(cw(h,'device')/ng); L.domination.push(cw(h,'domination')/ng);
      L.conquest.push(cw(h,'conquest')/ng); L.tiebreak.push((cw(h,'tiebreak')+cw(h,'bankruptcy'))/ng);
      L.loss.push(num(h.lossRate)); });
    cards.push(card('Win recipe (champion)','',stackedArea(ox,[
      {vals:L.conquest,color:C.bad,label:'conquest'},
      {vals:L.domination,color:C.illusion,label:'domination'},
      {vals:L.device,color:C.mil,label:'device'},
      {vals:L.tiebreak,color:C.tie,label:'tie/bank'},
      {vals:L.loss,color:C.faint,label:'loss'}
    ],{h:190}),'Miten voitot syntyvät — tavoite on siirtää econ-conquestista device/military suuntaan.'));
  }

  // loss + entropy (secondary axis)
  if(has(log,'policyLoss')||has(log,'valueLoss')){
    var lx=log.map(function(r){return r.gen;});
    cards.push(card('Loss & entropy','',chart(lx,[
      {vals:smooth(field(log,'policyLoss')),color:C.raw,label:'policy loss'},
      {vals:smooth(field(log,'valueLoss')),color:C.good,label:'value loss'},
      {vals:smooth(field(log,'policyEntropy')),color:C.illusion,label:'entropy (R)',axis:'r',dash:true}
    ],{h:190,rightAxis:true}),'Entropy oik. akselilla — romahdus = mode-collapse.'));
  }
  // value calibration
  if(has(log,'valPredWin')||has(log,'valPredLoss')){
    var vx=log.map(function(r){return r.gen;});
    cards.push(card('Value-head kalibrointi','',chart(vx,[
      {vals:smooth(field(log,'valPredWin')),color:C.good,label:'pred|win'},
      {vals:smooth(field(log,'valPredDraw')),color:C.tie,label:'pred|draw'},
      {vals:smooth(field(log,'valPredLoss')),color:C.bad,label:'pred|loss'}
    ],{h:190,y0:-1,y1:1}),'Pitäisi erottua: win→+1, loss→−1. Lähellä toisiaan = draw-attractor.'));
  }
  // health micro-tiles
  var b=s.benchLatest||{}, lt=s.latest||{};
  cards.push(card('Terveysmittarit','',
    '<div class="tiles">'
    + tile('Bankruptcy share', pct(b.bankruptcyWinShare), b.bankruptcyWinShare>0.1)
    + tile('Avg game len', b.roundsByOutcome? f2((num(b.roundsByOutcome.win)+num(b.roundsByOutcome.loss))/2,0)+' r':'—')
    + tile('Buffer', lt.bufferSize!=null? Math.round(lt.bufferSize/1000)+'k':'—')
    + tile('New ex/it', lt.newExamples!=null? String(lt.newExamples):'—')
    + tile('Iter wall', lt.elapsedSec!=null? Math.round(lt.elapsedSec/60)+' min':'—')
    + tile('Contact rate', pct(lt.spContactRate), num(lt.spContactRate)!=null&&num(lt.spContactRate)<0.5)
    + '</div>'));
  fill('overview', cards.join(''));
}

/* ===================== ECONOMY ===================== */
function panelEconomy(){
  var s=STATE, hist=lastN(s.winHistory||[]); var cards=[];
  var b=s.benchLatest||{};
  if(hist.length){
    var hx=hist.map(function(h){return h.gen;});
    cards.push(card('Rakennukset / peli','',chart(hx,[
      {vals:smooth(hist.map(function(h){return num(h.villagesPerGame);})),color:C.econ,label:'villages'},
      {vals:smooth(hist.map(function(h){return num(h.outpostsPerGame);})),color:C.mil,label:'outposts'},
      {vals:smooth(hist.map(function(h){return num(h.bridgesPerGame);})),color:C.raw,label:'bridges'}
    ],{h:190,y0:0}),'Talous- ja armeijaketjun rakennustahti.'));
  }
  // ★ MINE STAFFING (mineWorkerBins + expert lever). The REAL per-mine worker
  // distribution: # of champ mines staffed by 1 / 2 / 3+ BasicWorkers, plus how
  // many of those mines have an Expert (an Expert co-located with workers DOUBLES
  // the mine's metal: metal = 20·workers·(expert?2:1)). Graceful '—' for old runs.
  if(b.mineWorkerBins){
    var mwb=b.mineWorkerBins;
    var keys=['1','2','3'], maxv=Math.max(1, num(mwb['1'])||0, num(mwb['2'])||0, num(mwb['3'])||0);
    var ghost = hist.length>1 ? (hist[Math.max(0,hist.length-6)].mineWorkerBins||{}) : {};
    var items=keys.map(function(k){ return { name:k+' työläistä / kaivos', val:num(mwb[k])||0, right:String(num(mwb[k])||0)+(ghost[k]!=null?(' ('+ghost[k]+')'):''),
      color:k==='1'?'#3a5a66':(k==='2'?'#5a93a8':C.econ) }; });
    var nExp=num(b.minesWithExpert)||0, nMines=num(b.mineCount)||0;
    var nPExp=num(b.plantsWithExpert), nPlants=num(b.plantCount);
    var expRow='<div class="hbar"><span class="nm">kaivos expertillä</span><span class="track"><span class="fill" style="width:'
      +(nMines>0?Math.round(nExp/nMines*100):0)+'%;background:'+C.good+'"></span></span>'
      +'<span class="n">'+nExp+' / '+nMines+'</span></div>';
    // Plant (Hydro/Nuclear) expert leverage row — only when the field is present
    // (old benchmark rows lack plantsWithExpert/plantCount). Keeps the EXPERT-VIPU
    // block consistent with the honest standing-expert KPI above.
    if(nPlants!=null){
      expRow+='<div class="hbar"><span class="nm">voimala expertillä</span><span class="track"><span class="fill" style="width:'
        +(nPlants>0?Math.round((nPExp||0)/nPlants*100):0)+'%;background:'+C.good+'"></span></span>'
        +'<span class="n">'+(nPExp||0)+' / '+nPlants+'</span></div>';
    }
    cards.push(card('★ Kaivosten miehitys '+tip('Kuinka monella työläisellä kaivos pyörii (mineWorkerBins, summattu bench-peleistä) + montako kaivosta/voimalaa on expertillä. Expert + työläiset = metalli/energia ×2. Talous-scaffold asettaa expertit (ks. Asiantuntijat-paneeli). Suluissa ~5 gen sitten.'),'',
      hbars(items,{max:maxv})
      +'<div class="well" style="margin-top:8px"><div class="dim" style="font-size:10px;text-transform:uppercase;letter-spacing:.06em">Expert-vipu (tuotanto ×2)</div>'+expRow+'</div>',
      'experttejä kaivoksilla: '+nExp+' / '+nMines+(nPlants!=null?(', voimaloilla: '+(nPExp||0)+' / '+nPlants):'')+'. Ali-miehitetty + ilman expertiä = metalli pullonkaula.'));
  } else {
    cards.push(card('★ Kaivosten miehitys '+tip('Per-kaivos työläisjakauma + expert-vipu (mineWorkerBins / minesWithExpert). Puuttuu vanhoista ajoista.'),'',
      '<div class="empty">—  (ei mineWorkerBins-dataa tässä ajossa)</div>',''));
  }
  // SOLDIER STACKING (stackBins) — peak champ SOLDIERS on a single tile (metric M6),
  // bucketed per bench game. (Was previously MISLABELED as mine manning.)
  if(b.stackBins){
    var sb=b.stackBins;
    var skeys=['1','2','3'], smaxv=Math.max(1, num(sb['1'])||0, num(sb['2'])||0, num(sb['3'])||0);
    var sghost = hist.length>1 ? (hist[Math.max(0,hist.length-6)].stackBins||{}) : {};
    var sitems=skeys.map(function(k){ return { name:k+' sotilasta / ruutu', val:num(sb[k])||0, right:String(num(sb[k])||0)+(sghost[k]!=null?(' ('+sghost[k]+')'):''),
      color:k==='1'?'#5a5a3a':(k==='2'?'#9a8a4a':C.mil) }; });
    cards.push(card('Sotilaspino / ruutu (peak) '+tip('Huippumäärä mestarin sotilaita YHDELLÄ ruudulla (M6), bucketoitu bench-peleittäin. Suluissa ~5 gen sitten.'),'',
      hbars(sitems,{max:smaxv}),'Sotilaiden pinoaminen yhteen ruutuun (ei kaivosten miehitys).'));
  }
  // EXPERTS — HONEST standing-expert metric. The economy SCAFFOLD (controller.rs
  // staff_income → add_expert_reserve) places experts on mines/plants mechanically;
  // these are NOT counted by expertsHiredPerGame (which tallies ONLY experts the
  // learned POLICY explicitly picks — virtually always 0). So we surface the real
  // count standing on the champion's board: standingExpertsPerGame when emitted,
  // else derived from (minesWithExpert + plantsWithExpert) / nGames for old rows.
  // The net-chosen number is shown as a small secondary line, clearly labelled.
  function standExp(h){
    var s=num(h.standingExpertsPerGame);
    if(s!=null) return s;
    var ng=num(h.nGames)||1;
    var me=num(h.minesWithExpert), pe=num(h.plantsWithExpert);
    if(me==null && pe==null) return null; // genuinely no data
    return ((me||0)+(pe||0))/ng;
  }
  if(hist.length){
    var ex=hist.map(function(h){return h.gen;});
    var standVals=hist.map(standExp);
    var hiredVals=hist.map(function(h){return num(h.expertsHiredPerGame);});
    var maxStand=Math.max.apply(null,standVals.map(function(v){return v||0;}).concat([1]));
    cards.push(card('Asiantuntijat / peli '+tip('REHELLINEN standing-expert luku: montako Asiantuntijaa mestarin laudalla / peli (kaivos + ydin/vesi), sis. talous-scaffoldin asettamat. Lähde: standingExpertsPerGame (tai johdettu minesWithExpert+plantsWithExpert / nGames). Eri kuin expertsHiredPerGame, joka laskee VAIN policyn itse valitsemat (≈0; scaffold hoitaa loput).'),'',
      chart(ex,[
        {vals:standVals,color:C.good,label:'standing experts/peli (scaffold+policy)',dots:true,w:2},
        {vals:hiredVals,color:C.illusion,label:'policy-chosen (scaffold asettaa loput)',dots:true,dash:true}
      ],{h:160,y0:0,y1:Math.max(1,maxStand)}),
      'Asiantuntijat boostaavat tuotantoa (metalli/energia ×2) ja porttaavat armeijaketjun. Vihreä = oikeasti laudalla; harmaa = vain policyn valinnat (scaffold asettaa loput).'));
  }
  // win-by-villages / win-by-outposts payoff
  if(b.winByOutpostsBuilt){
    cards.push(card('Maksaako econ-linja takaisin? '+tip('Voittoaste ehdolla outpostien/kylien määrä.'),'',
      '<div class="well"><div class="dim" style="font-size:10px;text-transform:uppercase;letter-spacing:.06em">Outpostit rakennettu</div>'
      + winBy(b.winByOutpostsBuilt) + '</div>'
      + (b.winByVillagesBuilt?'<div class="well" style="margin-top:8px"><div class="dim" style="font-size:10px;text-transform:uppercase;letter-spacing:.06em">Kylät rakennettu</div>'+winBy(b.winByVillagesBuilt)+'</div>':''),
      'Outpostit korreloivat voittamisen kanssa (signaali armeijaketjusta).'));
  }
  // metal balance proxy (client-side, sd3 constants). Uses the REAL per-mine
  // worker distribution (mineWorkerBins) when present; old runs fall back to 0.
  if(b.mineWorkerBins||b.outpostsPerGame!=null){
    var mwb2=b.mineWorkerBins||{};
    var mines = (num(mwb2['1'])||0)+(num(mwb2['2'])||0)+(num(mwb2['3'])||0);
    var nGames=num(b.nGames)||1;
    var minesPerGame = mines/nGames;
    var workerSlots = (num(mwb2['1'])||0)*1 + (num(mwb2['2'])||0)*2 + (num(mwb2['3'])||0)*3;
    var metalIn = workerSlots/nGames*20; // ~20 metal/worker-round (sd3 mine output)
    var metalOut = (num(b.outpostsPerGame)||0)*5 + (num(b.maxSoldiersPerGame)||0)*30; // upkeep proxies
    var bal = metalIn - metalOut;
    cards.push(card('Metallitase (johdettu) '+tip('mine-tuotto − outpost-upkeep(−5) − sotilas-kustannus(−30). Alle 0 = talous ei rahoita militaaria (juurisyy).'),'',
      '<div class="tiles">'
      + tile('Kaivoksia/peli', f2(minesPerGame,2))
      + tile('Metalli sisään ≈', f2(metalIn,0))
      + tile('Metalli ulos ≈', f2(metalOut,0))
      + tile('Tase ≈', f2(bal,0), bal<0)
      + '</div>','Karkea proxy sd3-vakioista (CLAUDE.md). Alle 0 = dokumentoitu pullonkaula.'));
  }
  if(!cards.length) cards.push(card('Talous','wide','<div class="empty">ei talousmittareita tässä ajossa</div>'));
  fill('economy', cards.join(''));
}
function winBy(obj){
  var keys=['0','1','2','3+'];
  return keys.map(function(k){ var o=obj[k]||{games:0,wins:0}; var g=num(o.games)||0, w=num(o.wins)||0;
    var wr=g>0?w/g:null;
    return '<div class="hbar"><span class="nm">'+k+'</span><span class="track"><span class="fill" style="width:'+(wr!=null?(wr*100):0).toFixed(0)+'%;background:'+(g<5?C.illusion:C.good)+'"></span></span>'
      +'<span class="n">'+(wr!=null?pct(wr,0):'—')+' <span class="dim">n='+g+'</span></span></div>';
  }).join('');
}

/* ===================== MILITARY (army-chain funnel) ===================== */
function panelMilitary(){
  var s=STATE, b=s.benchLatest||{}, hist=lastN(s.winHistory||[]); var cards=[];
  // ★ funnel: wood(proxy) -> mines -> experts -> outposts -> soldiers fielded
  var nGames=num(b.nGames)||1;
  var mines=((num(b.mineWorkerBins&&b.mineWorkerBins['1'])||0)+(num(b.mineWorkerBins&&b.mineWorkerBins['2'])||0)+(num(b.mineWorkerBins&&b.mineWorkerBins['3'])||0))/nGames;
  var soldiers = num(b.maxSoldiersPerGame);
  var stages=[
    {nm:'Wood', ct: num(b.villagesPerGame)!=null? f2(num(b.villagesPerGame)+mines+2,1):null, raw:(num(b.villagesPerGame)||0)+mines+2, note:'econ-proxy'},
    {nm:'Mine', ct: f2(mines,2), raw:mines, note:'metal source'},
    {nm:'Expert', ct: f2(b.expertsHiredPerGame,2), raw:num(b.expertsHiredPerGame), note:'prod boost'},
    {nm:'Outpost', ct: f2(b.outpostsPerGame,2), raw:num(b.outpostsPerGame), note:'soldier-cap +3'},
    {nm:'Soldiers', ct: f2(soldiers,2), raw:soldiers, note:'fielded (peak/game)'}
  ];
  var maxRaw=Math.max.apply(null, stages.map(function(s){return s.raw||0;}))||1;
  // bottleneck = biggest relative drop
  var worst=-1, worstDrop=-1;
  for(var i=1;i<stages.length;i++){ var prev=stages[i-1].raw||0.0001, cur=stages[i].raw||0; var drop=(prev-cur)/prev; if(drop>worstDrop){worstDrop=drop;worst=i;} }
  var fhtml='<div class="funnel">';
  stages.forEach(function(st,i){
    if(i>0){ var drop=stages[i-1].raw>0?Math.round((1-(st.raw||0)/stages[i-1].raw)*100):0;
      fhtml+='<div class="farrow"><div style="text-align:center">▶<div class="fdrop">'+(drop>0?'−'+drop+'%':'')+'</div></div></div>'; }
    var h=Math.max(2,(st.raw||0)/maxRaw*116);
    fhtml+='<div class="fstage'+(i===worst?' bottleneck':'')+'"><div class="bar"><div class="fill" style="height:'+h+'px"></div></div>'
      +'<div class="ct'+((st.raw||0)<0.05?' z':'')+'">'+(st.ct||'—')+'</div><div class="nm" title="'+esc(st.note)+'">'+st.nm+'</div></div>';
  });
  fhtml+='</div>';
  cards.push(card('★ Armeijaketju: wood → mine → expert → outpost → soldiers','span2 wide',fhtml,
    'Per peli (uusin benchmark). Amber-reunus = suurin pudotus (pullonkaula). Expert=0 katkaisee ketjun.'));

  // soldier utilization (useful vs useless + attack/defend/idle)
  cards.push(card('Sotilaiden hyödyllisyys '+tip('Kun kone vihdoin rakentaa sotilaita, tekevätkö ne mitään?'),'',
    '<div class="well"><div class="dim" style="font-size:10px">useful vs useless roundit</div>'
    + stackBar([{label:'useful',v:num(b.soldierUsefulRounds)||0,color:C.good},{label:'useless',v:num(b.soldierUselessRounds)||0,color:C.bad}])
    + '<div class="dim" style="font-size:10px;margin-top:8px">attack / defend / idle</div>'
    + stackBar([{label:'attack',v:num(b.soldierAttack)||0,color:C.bad},{label:'defend',v:num(b.soldierDefend)||0,color:C.raw},{label:'idle',v:num(b.soldierIdle)||0,color:C.tie}])
    + '</div>',
    '0% attack / 97% defend = sotilaat puhtaasti puolustavia (ei hyökkää = ei voita conquestilla).'));

  // soldier army size over gens + capacity envelope + champSoldierBins
  if(hist.length){
    var ax=hist.map(function(h){return h.gen;});
    var cap=hist.map(function(h){ return 1 + 3*(num(h.outpostsPerGame)||0); });
    cards.push(card('Armeijan koko vs kapasiteetti','',chart(ax,[
      {vals:cap,color:C.faint,label:'cap (HQ+1+3·outpost)',dash:true},
      {vals:smooth(hist.map(function(h){return num(h.maxSoldiersPerGame);})),color:C.mil,label:'soldiers fielded',w:2.2}
    ],{h:190,y0:0}),'Kapasiteettisokeus: ilman outposteja katto ≈ 1 sotilas koko pelin.'));
  }
  if(b.champSoldierBins){
    var sbins=b.champSoldierBins, ks=['0','1','2','3','4+'];
    var mx=Math.max.apply(null,ks.map(function(k){return num(sbins[k])||0;}))||1;
    cards.push(card('Armeijan kokojakauma (peak/peli)','',hbars(ks.map(function(k){
      return {name:k+' sotilasta',val:num(sbins[k])||0,right:String(num(sbins[k])||0),color:k==='0'?C.bad:C.mil};
    }),{max:mx}),'Suurin osa peleistä armeijattomia.'));
  }
  // assault counters (known-zero)
  cards.push(card('Hyökkäyslaskurit '+tip('crackDevice/HQ — voittoehdon rynnäkkö. 0 = ei koskaan rynnäköi.'),'',
    '<div class="tiles">'
    + tile('CrackDevice yrit.', String(num(b.crackDeviceAttempts)||0), (num(b.crackDeviceAttempts)||0)===0)
    + tile('CrackDevice onn.', String(num(b.crackDeviceSuccesses)||0), (num(b.crackDeviceSuccesses)||0)===0)
    + tile('CrackHQ yrit.', String(num(b.crackHQAttempts)||0), (num(b.crackHQAttempts)||0)===0)
    + tile('CrackHQ onn.', String(num(b.crackHQSuccesses)||0), (num(b.crackHQSuccesses)||0)===0)
    + '</div>'));
  fill('military', cards.join(''));
}

/* ===================== OPPONENTS ===================== */
var BENCH_OPP=[['benchVsHard','Hard'],['benchVsRusher','Rusher'],['benchVsStrongArmy','StrongArmy'],
  ['benchVsFortress','Fortress'],['benchVsDeviceRush','DeviceRush']];
var SP_OPP=[['spVsRusher','Rusher'],['spVsStrongArmy','StrongArmy'],['spVsFortress','Fortress'],
  ['spVsDeviceRush','DeviceRush'],['spVsArmyRush','ArmyRush'],['spVsGarrison','Garrison'],
  ['spVsHqRush','HqRush'],['spVsMarcher','Marcher'],['spVsExpert','Expert']];
function panelOpponents(){
  var s=STATE, b=s.benchLatest||{}, hist=lastN(s.winHistory||[]), log=lastN(s.log||[]); var cards=[];
  var nPer=num(b.benchPerOpp);
  // ★ benchmark win-rate bars (authoritative)
  var items=BENCH_OPP.map(function(o){ var v=num(b[o[0]]);
    return {name:o[1], val:v, right:(v!=null?pct(v,0):'—')+(nPer?(' n='+nPer):''),
      warn:(nPer!=null&&nPer<30), color:(v!=null&&v>=0.5)?C.good:C.illusion}; })
    .filter(function(it){ return it.val!=null; });
  cards.push(card('★ Liiga win-rate (benchmark, vs HARD-bot + skriptit)','span2 wide',
    items.length? hbars(items,{max:1,ref:0.5}) : '<div class="empty">ei benchVs*-dataa tässä ajossa</div>',
    'Auktoritatiivinen: champion vs skriptattu liiga. 50% viiva referenssinä; amber jos n<30.'));

  // per-opponent training trend small-multiples (N-gated)
  if(log.length){
    var sm=SP_OPP.map(function(o){
      var vals=log.map(function(r){ var n=num(r[o[0]+'N']); return (n!=null&&n>0)? num(r[o[0]]) : null; });
      if(!vals.some(function(v){return v!=null;})) return '';
      var last=null; for(var i=vals.length-1;i>=0;i--){ if(vals[i]!=null){last=vals[i];break;} }
      var bad=last!=null&&last<0.4;
      return '<div class="tile"><div class="lbl" style="'+(bad?'color:var(--bad)':'')+'">'+o[1]+'</div>'
        + spark(vals,bad?C.bad:C.good,28) + '<div class="sub">'+(last!=null?pct(last,0):'—')+'</div></div>';
    }).filter(Boolean).join('');
    cards.push(card('Per-vastustaja treeni-trendi '+tip('spVs* self-play probet, N-gated (N>0). Punainen = heikoin matchup.'),'span2 wide',
      sm? '<div class="tiles">'+sm+'</div>' : '<div class="empty">ei spVs*-probeja vielä</div>',
      'Treeni-aikainen self-play-mittaus; benchmark yllä on auktoritatiivinen.'));
  }
  // seat bias
  cards.push(card('Seat-bias '+tip('winSeat0 vs winSeat1 — epäsymmetria.'),'',
    hbars([{name:'seat 0',val:num(b.winSeat0),right:pct(b.winSeat0,0),color:C.raw},
           {name:'seat 1',val:num(b.winSeat1),right:pct(b.winSeat1,0),color:C.mil}],{max:1,ref:0.5})));
  fill('opponents', cards.join(''));
}

/* ===================== REPLAY (canvas viewer) ===================== */
var R_SRC = localStorage.getItem('cp.dash.rsrc') || 'hard';
var R_IDX=0, R_FRAME=0, R_PLAYING=true, R_FPS=48, R_KEY='', R_BATCH='', R_TIMER=null;
var OPP_META = {/* filled from server-aligned constants */};
var OPPS=[
  ['hard','replay','Hard CPU',false],['self','replaySelf','Self-play',false],
  ['rusher','replayVsRusher','Rusher',false],['strongarmy','replayVsStrongArmy','StrongArmy',false],
  ['fortress','replayVsFortress','Fortress',false],['devicerush','replayVsDeviceRush','DeviceRush',false],
  ['armyrush','replayVsArmyRush','ArmyRush',true],['garrison','replayVsGarrison','Garrison',true],
  ['hqrush','replayVsHqRush','HqRush',true],['marcher','replayVsMarcher','Marcher',true],
  ['expert','replayVsExpert','Expert',true]
];
function oppMeta(src){ for(var i=0;i<OPPS.length;i++) if(OPPS[i][0]===src) return OPPS[i]; return ['hard','replay','?',false]; }
function gamesFor(src){ var k=oppMeta(src)[1]; var g=STATE&&STATE[k]; return Array.isArray(g)?g:[]; }
function activeReplay(){ var gs=gamesFor(R_SRC); if(!gs.length) return null; return gs[Math.min(R_IDX,gs.length-1)]; }
function batchKey(src){ var r=gamesFor(src)[0]; return r?(src+':'+r.iter+':'+r.seed):''; }

var BGLYPH={F:'F',M:'M',V:'V',O:'O',H:'H',N:'N',B:'B',D:'◆',Q:'★',K:'K'};
var BCOLOR={D:'#c792ea',Q:'#ffcb6b'};
var TCOLOR={r:'#14506a',m:'#454b54',f:'#1d3a28',a:'#15301e',g:'#23301a'};
function drawReplayFrame(canvas,r,fi){
  if(!canvas||!r||!r.frames) return;
  var f=r.frames[fi]; if(!f) return;
  var W=r.width,H=r.height,terr=r.terrain||'';
  var cell=Math.max(10,Math.min(34,Math.floor(520/W)));
  if(canvas.width!==cell*W){ canvas.width=cell*W; canvas.height=cell*H; }
  var ctx=canvas.getContext('2d'); ctx.clearRect(0,0,canvas.width,canvas.height);
  var own=f.own||'',bld=f.bld||'',sol=f.sol||''; // tolerate a partial mid-write frame
  for(var i=0;i<own.length;i++){
    var x=Math.floor(i/H),y=i%H; var px=x*cell,py=y*cell; // column-major i=x*H+y
    ctx.fillStyle=TCOLOR[terr[i]]||'#161c24'; ctx.fillRect(px,py,cell,cell);
    var o=own[i];
    if(o==='1'||o==='2'){ ctx.fillStyle=o==='1'?'rgba(90,169,255,0.30)':'rgba(255,107,107,0.30)'; ctx.fillRect(px,py,cell,cell);
      ctx.strokeStyle=o==='1'?'#5aa9ff':'#ff6b6b'; ctx.lineWidth=2; ctx.strokeRect(px+1.5,py+1.5,cell-3,cell-3); }
    ctx.strokeStyle='#0b0f14'; ctx.lineWidth=1; ctx.strokeRect(px+0.5,py+0.5,cell-1,cell-1);
    var b=bld[i];
    if(b&&b!=='.'){ ctx.fillStyle=BCOLOR[b]||'#e6edf3'; ctx.font='700 '+Math.floor(cell*0.56)+'px ui-monospace,monospace';
      ctx.textAlign='center'; ctx.textBaseline='middle'; ctx.fillText(BGLYPH[b]||b,px+cell/2,py+cell/2+1); }
    var su=sol[i];
    if(su&&su!=='.'&&su!=='0'){ ctx.fillStyle='#ffcb6b'; ctx.font='700 '+Math.floor(cell*0.4)+'px ui-monospace,monospace';
      ctx.textAlign='right'; ctx.textBaseline='bottom'; ctx.fillText(su,px+cell-1,py+cell-0); }
  }
}
function replaySide(r,fi){
  var f=r.frames[fi]; var meta=oppMeta(R_SRC); var self=r.mode==='self';
  var blue=(self?'AI #1':'Meidän AI')+' (sininen)';
  var red=(self?'AI #2':(r.mode==='hard'?'Hard CPU':meta[2]))+' (punainen)';
  var turn=f.p===0?'<span class="blue">'+blue+'</span>':(f.p===1?'<span class="red">'+red+'</span>':'asetelma');
  var res;
  var rr = r.result||{};
  if(rr.winnerSeat===0) res='<span class="blue">'+blue+'</span> voitti — '+esc(rr.cause);
  else if(rr.winnerSeat===1) res='<span class="red">'+red+'</span> voitti — '+esc(rr.cause);
  else res='ratkeamaton';
  var modeStr=self?' · self-play':(r.mode==='hard'?' · vs hard':' · vs '+(meta[2]));
  return '<div class="big">Iteraatio '+esc(r.iter)+modeStr+'</div>'
    + 'Kierros <b style="color:var(--ink)">'+esc(f.r)+'</b> · vuoro: '+turn+'<br>'
    + 'Ruutu '+(fi+1)+'/'+r.frames.length+'<br><br>'
    + '<b style="color:var(--ink)">Lopputulos:</b><br>'+res+' ('+esc(rr.rounds!=null?rr.rounds:'?')+' kierrosta)';
}
function replaySrcToggle(){
  var legacyStarted=false;
  var html='<span class="seglbl">liiga</span>';
  OPPS.forEach(function(o){
    if(o[3]&&!legacyStarted){ legacyStarted=true; html+='<span class="seglbl">vanhat</span>'; }
    html+='<button class="seg rsel'+(o[3]?' legacy':'')+(o[0]===R_SRC?' sel':'')+'" data-src="'+o[0]+'">'+esc(o[2])+'</button>';
  });
  return '<div class="segmented" style="margin-bottom:10px;width:100%">'+html+'</div>';
}
function ensureReplayTimer(){
  if(R_TIMER) return;
  R_TIMER=setInterval(function(){
    if(TAB!=='replay') return;
    var r=activeReplay(); if(!r||!r.frames||!r.frames.length) return;
    var canvas=document.getElementById('replayCanvas'); if(!canvas) return;
    if(R_PLAYING) R_FRAME=(R_FRAME+1)%r.frames.length;
    drawReplayFrame(canvas,r,R_FRAME);
    var sb=document.getElementById('replayScrub'); if(sb&&document.activeElement!==sb) sb.value=String(R_FRAME);
    var side=document.getElementById('replaySide'); if(side) side.innerHTML=replaySide(r,R_FRAME);
  },Math.round(1000/R_FPS));
}
function restartReplayTimer(){ if(R_TIMER){ clearInterval(R_TIMER); R_TIMER=null; } ensureReplayTimer(); }
function panelReplay(){
  var panel=document.querySelector('[data-panel="replay"]'); if(!panel) return;
  var meta=oppMeta(R_SRC), self=R_SRC==='self', scripted=R_SRC!=='hard'&&R_SRC!=='self';
  var batch=batchKey(R_SRC); if(batch!==R_BATCH){ R_BATCH=batch; R_IDX=0; }
  var r=activeReplay(); var n=gamesFor(R_SRC).length;
  var title='<h3>Live-peli — '+(self?'AI vs AI':(scripted?('AI vs '+meta[2]):'AI vs Hard CPU'))
    +' <span class="dim" style="text-transform:none">· '+(n||0)+' tuoretta peliä/iteraatio</span></h3>';
  var key=R_SRC+':'+R_IDX+':'+(r&&r.frames?(r.iter+':'+r.seed+':'+r.frames.length):'none');
  if(key===R_KEY && panel.dataset.k===key){ return; }
  R_KEY=key; R_FRAME=0;
  if(!r||!r.frames||!r.frames.length){
    var fn='replay'+(R_SRC==='hard'?'':R_SRC==='self'?'_selfplay':'_vs_'+R_SRC)+'.json';
    panel.dataset.k='';
    panel.innerHTML='<div class="card wide">'+title+replaySrcToggle()
      +'<div class="empty">Ei replayta — odotetaan tiedostoa <code>'+esc(fn)+'</code> (kirjoitetaan joka --replay-every).'
      +(meta[3]?' Tämä on vanha liigan ulkopuolinen vastustaja.':'')+'</div></div>';
    wireReplayToggle(); return;
  }
  panel.dataset.k=key;
  var latestIter=STATE.latest&&typeof STATE.latest.gen==='number'?STATE.latest.gen:null;
  var staleBy=(latestIter!=null&&typeof r.iter==='number')?(latestIter-r.iter):null;
  var stale=(staleBy!=null&&staleBy>=25)?'<div class="note" style="color:var(--illusion);margin-bottom:8px">⚠ Replay iteraatiosta '+r.iter+', koulutus jo '+latestIter+' ('+staleBy+' jäljessä).</div>':'';
  var blueLbl=self?'AI #1':'meidän AI', redLbl=self?'AI #2':(scripted?meta[2]:'Hard CPU');
  panel.innerHTML='<div class="card wide">'+title+replaySrcToggle()+stale
    +'<div class="stage"><canvas id="replayCanvas"></canvas><div class="side" id="replaySide"></div></div>'
    +'<div class="ctl"><button class="btn" id="replayPlay"></button>'
    +'<input type="range" id="replayScrub" min="0" max="'+(r.frames.length-1)+'" value="0">'
    +'<span id="replaySpeed" style="cursor:pointer;color:var(--accent);font-weight:600;user-select:none"></span>'
    +'<button class="btn" id="replayNext" title="Selaa tämän iteraation pelit">Seuraava peli ⏭</button>'
    +'<span id="replayGamePos" class="dim" style="font-size:11px"></span></div>'
    +'<div class="leg"><span style="color:#5aa9ff">sininen='+esc(blueLbl)+'</span> · <span style="color:#ff6b6b">punainen='+esc(redLbl)+'</span>'
    +' · maasto: <span style="color:#3a9fd0">joki</span> <span style="color:#8a929c">vuori</span> <span style="color:#3f8a5c">metsä</span> ruoho'
    +' · kirjaimet=rakennukset (F mine V O H N B silta ★ HQ <span style="color:#c792ea">◆ Strange Device</span>) · keltainen numero=sotilaat</div>'
    +'</div>';
  wireReplayToggle();
  var playBtn=document.getElementById('replayPlay'),scrub=document.getElementById('replayScrub'),speed=document.getElementById('replaySpeed');
  function syncPlay(){ playBtn.textContent=R_PLAYING?'⏸ tauko':'▶ toista'; playBtn.className='btn'+(R_PLAYING?' on':''); }
  function syncSpeed(){ speed.textContent=(R_FPS/6).toFixed(1).replace(/\\.0$/,'')+'×'; }
  playBtn.onclick=function(){ R_PLAYING=!R_PLAYING; syncPlay(); };
  scrub.oninput=function(){ R_PLAYING=false; syncPlay(); R_FRAME=Number(scrub.value); var a=activeReplay();
    drawReplayFrame(document.getElementById('replayCanvas'),a,R_FRAME); document.getElementById('replaySide').innerHTML=replaySide(a,R_FRAME); };
  speed.onclick=function(){ var steps=[3,6,12,24,48]; var i=steps.indexOf(R_FPS); R_FPS=steps[(i<0?steps.length-1:i+1)%steps.length]; syncSpeed(); restartReplayTimer(); };
  var nextBtn=document.getElementById('replayNext'),gamePos=document.getElementById('replayGamePos');
  function syncPos(){ var nn=gamesFor(R_SRC).length||1; gamePos.textContent='peli '+(R_IDX+1)+'/'+nn+' · iter '+(r.iter!=null?r.iter:'?'); }
  nextBtn.onclick=function(){ var nn=gamesFor(R_SRC).length; if(nn<2){ syncPos(); return; } R_IDX=(R_IDX+1)%nn; R_KEY=''; panel.dataset.k=''; R_FRAME=0; R_PLAYING=true; panelReplay(); };
  syncPlay(); syncSpeed(); syncPos();
  drawReplayFrame(document.getElementById('replayCanvas'),r,0);
  document.getElementById('replaySide').innerHTML=replaySide(r,0);
  ensureReplayTimer();
}
function wireReplayToggle(){
  document.querySelectorAll('.rsel').forEach(function(b){
    b.classList.toggle('sel', b.dataset.src===R_SRC);
    b.onclick=function(){ if(b.dataset.src!==R_SRC){ R_SRC=b.dataset.src; localStorage.setItem('cp.dash.rsrc',R_SRC); R_IDX=0; R_KEY=''; var p=document.querySelector('[data-panel="replay"]'); if(p)p.dataset.k=''; panelReplay(); } };
  });
}

/* ===================== SPATIAL (CNN heatmap) ===================== */
var SP_MAP='policy', SP_FRAME=null;
var STCOLOR={r:'#14506a',m:'#454b54',f:'#1d3a28',a:'#15301e',g:'#23301a'};
var SP_BGLYPH={F:'F',M:'M',V:'V',O:'O',H:'H',N:'N',B:'B',D:'◆',HQ:'★',K:'K'};
var SP_BCOLOR={D:'#c792ea',HQ:'#ffcb6b'};
function warmRGBA(t){ t=Math.max(0,Math.min(1,t)); var g=Math.round(203-96*t); return 'rgba(255,'+g+',107,'+(0.10+0.80*t).toFixed(3)+')'; }
function divRGBA(s){ s=Math.max(-1,Math.min(1,s)); var a=0.18+0.70*Math.min(1,Math.abs(s));
  if(s>=0){ var t=s; return 'rgba('+Math.round(139-86*t)+','+Math.round(151+59*t)+','+Math.round(163-56*t)+','+a.toFixed(3)+')'; }
  var u=-s; return 'rgba('+Math.round(139+116*u)+','+Math.round(151-44*u)+','+Math.round(163-56*u)+','+a.toFixed(3)+')'; }
function spatialFrames(sp){ if(!sp) return null; if(Array.isArray(sp.frames)&&sp.frames.length) return sp; if(sp.width) return {iter:sp.iter,width:sp.width,height:sp.height,frames:[sp]}; return null; }
function activeSpFrame(spn){ var n=spn.frames.length; var idx=SP_FRAME==null?Math.floor((n-1)/2):SP_FRAME; idx=Math.max(0,Math.min(n-1,idx)); return {idx:idx,frame:spn.frames[idx]}; }
function spatialHeat(f){
  var n=(f.terrain||'').length, pol=f.policy||[], vm=f.valueMap||[], root=num(f.value)||0;
  if(SP_MAP==='policy'){ var mx=0; for(var i=0;i<pol.length;i++) if(pol[i]>mx)mx=pol[i]; return {kind:'seq',vals:pol,max:mx||1}; }
  var vals=new Array(n).fill(null), absMax=1e-6;
  for(var j=0;j<n;j++){ var v=vm[j]; if(v==null) continue; var xx=SP_MAP==='delta'?(v-root):v; vals[j]=xx; if(Math.abs(xx)>absMax) absMax=Math.abs(xx); }
  return {kind:'div',vals:vals,max:absMax};
}
function drawSpatial(canvas,spn){
  var af=activeSpFrame(spn), f=af.frame, W=spn.width,H=spn.height, terr=f.terrain||'';
  var own=f.owner||[],bld=f.building||[],sol=f.soldiers||[];
  var cell=Math.max(24,Math.min(40,Math.floor(540/W)));
  if(canvas.width!==cell*W){ canvas.width=cell*W; canvas.height=cell*H; }
  var ctx=canvas.getContext('2d'); ctx.clearRect(0,0,canvas.width,canvas.height);
  var heat=spatialHeat(f);
  for(var y=0;y<H;y++) for(var x=0;x<W;x++){
    var i=y*W+x; var px=x*cell,py=y*cell; // ROW-MAJOR
    ctx.fillStyle=STCOLOR[terr[i]]||'#161c24'; ctx.fillRect(px,py,cell,cell);
    var hv=heat.vals[i];
    if(hv!=null){ if(heat.kind==='seq'){ if(hv>0){ ctx.fillStyle=warmRGBA(hv/heat.max); ctx.fillRect(px,py,cell,cell); } }
      else { ctx.fillStyle=divRGBA(hv/heat.max); ctx.fillRect(px,py,cell,cell); } }
    var o=own[i];
    if(o===0||o===1){ ctx.fillStyle=o===0?'rgba(90,169,255,0.16)':'rgba(255,107,107,0.16)'; ctx.fillRect(px,py,cell,cell);
      ctx.strokeStyle=o===0?'#5aa9ff':'#ff6b6b'; ctx.lineWidth=2; ctx.strokeRect(px+1.5,py+1.5,cell-3,cell-3); }
    ctx.strokeStyle='#0b0f14'; ctx.lineWidth=1; ctx.strokeRect(px+0.5,py+0.5,cell-1,cell-1);
    var b=bld[i];
    if(b){ var isHq=(i===f.myHq||i===f.enemyHq);
      ctx.font='700 '+Math.floor(cell*0.5)+'px ui-monospace,monospace'; ctx.textAlign='center'; ctx.textBaseline='middle';
      ctx.lineWidth=3; ctx.strokeStyle='rgba(11,15,20,0.85)';
      var gl=isHq?'★':(SP_BGLYPH[b]||b);
      ctx.strokeText(gl,px+cell/2,py+cell/2+1); ctx.fillStyle=isHq?'#ffcb6b':(SP_BCOLOR[b]||'#e6edf3'); ctx.fillText(gl,px+cell/2,py+cell/2+1); }
    var sv=sol[i];
    if(sv){ ctx.fillStyle='#ffcb6b'; ctx.font='700 '+Math.floor(cell*0.34)+'px ui-monospace,monospace';
      ctx.textAlign='right'; ctx.textBaseline='bottom'; ctx.fillText(String(sv),px+cell-2,py+cell-1); }
  }
}
function panelSpatial(){
  var panel=document.querySelector('[data-panel="spatial"]'); if(!panel) return;
  var spn=spatialFrames(STATE&&STATE.spatial);
  if(!spn){ panel.innerHTML='<div class="card wide"><h3>CNN spatial</h3><div class="empty">ei spatial.jsonia (ei-CNN ajo)</div></div>'; return; }
  var af=activeSpFrame(spn), f=af.frame;
  var hasVm=(f.valueMap||[]).some(function(v){return v!=null;});
  var frameSel=spn.frames.map(function(fr,i){ return '<button class="seg'+(i===af.idx?' sel':'')+'" data-spf="'+i+'">'+esc(fr.label||('#'+i))+'</button>'; }).join('');
  var mapSel=[['policy','policy'],['delta','Δ-value'],['valueMap','valueMap']].map(function(m){
    var dis=(m[0]!=='policy'&&!hasVm); return '<button class="seg'+(SP_MAP===m[0]?' sel':'')+(dis?' legacy':'')+'" data-spm="'+m[0]+'"'+(dis?' disabled':'')+'>'+m[1]+'</button>'; }).join('');
  var tm=(f.topMoves||[]).filter(function(m){return m&&m.idx>=0;}).slice(0,6).map(function(m){
    return '<div class="tm"><span>'+esc(m.intent)+' <span class="dim">@'+m.idx+'</span></span><span>p='+f2(m.prob,3)+'</span><span class="dim">v='+f2(m.valueAfter,2)+'</span></div>';
  }).join('');
  panel.innerHTML='<div class="card wide"><h3>CNN spatial — mitä verkko ajattelee '+tip('policy = mihin verkko haluaa toimia; Δ-value = valueAfter − root; value = raaka per-tile arvo.')+'</h3>'
    +'<div class="ctl"><span class="seglbl">vaihe</span><span class="segmented">'+frameSel+'</span>'
    +'<span class="seglbl">kerros</span><span class="segmented">'+mapSel+'</span>'
    +'<span class="dim">round '+esc(f.round)+' · value='+f2(f.value,3)+' · iter '+esc(spn.iter)+'</span></div>'
    +'<div class="stage" style="margin-top:10px"><canvas id="spatialCanvas"></canvas>'
    +'<div class="side"><div class="big">Net verdict: <span style="color:'+((num(f.value)||0)>=0?'var(--good)':'var(--bad)')+'">'+f2(f.value,3)+'</span></div>'
    +'<div class="topmoves"><div class="dim" style="font-size:10px;text-transform:uppercase">top moves</div>'+(tm||'<div class="empty">—</div>')+'</div></div></div>'
    +'<div class="leg">policy: <span style="color:#ffcb6b">heikko</span>→<span style="color:#ff6b6b">vahva</span> · Δ/value: <span style="color:#ff6b6b">neg</span>→<span style="color:#4dd2a0">pos</span></div>'
    +'</div>';
  document.querySelectorAll('[data-spf]').forEach(function(b){ b.onclick=function(){ SP_FRAME=Number(b.dataset.spf); panelSpatial(); }; });
  document.querySelectorAll('[data-spm]').forEach(function(b){ if(b.disabled) return; b.onclick=function(){ SP_MAP=b.dataset.spm; panelSpatial(); }; });
  drawSpatial(document.getElementById('spatialCanvas'),spn);
}

/* ===================== MODELS ===================== */
function championId(){
  var s=STATE||{}; var ptr=s.championPtr&&s.championPtr.champions;
  if(ptr){ // prefer current-arc champion pointer
    var arcs=Object.keys(ptr); if(arcs.length){ // newest arc by name sort
      arcs.sort(); return ptr[arcs[arcs.length-1]]; } }
  // fallback: latest sd3 registry row, else last registry row
  var reg=s.registry||[]; var sd3=reg.filter(function(r){return r.arc==='sd3';});
  if(sd3.length) return sd3[sd3.length-1].id;
  return reg.length? reg[reg.length-1].id : null;
}
function panelModels(){
  var s=STATE, reg=(s.registry||[]).slice(), ptr=(s.championPtr&&s.championPtr.champions)||{}, dep=(s.championPtr&&s.championPtr.deployed)||{};
  var cards=[];
  var champ=championId();
  // champion card
  var crow=reg.filter(function(r){return r.id===champ;})[0];
  cards.push(card('Champion','',
    crow? '<div class="tiles">'+tile('id',crow.id)+tile('arc',crow.arc)+tile('type',crow.type)
      +tile('honest win', crow.winrate_vs_hard!=null?pct(crow.winrate_vs_hard):'—')
      +tile('rekisteröity', crow.created_utc?crow.created_utc.slice(0,10):'—')+'</div>'
      +(crow.notes?'<div class="note" style="margin-top:8px">'+esc(crow.notes)+'</div>':'')
      : '<div class="empty">ei championia rekisterissä</div>',
    s.championPtr&&Object.keys(ptr).length? '' : 'CHAMPION.json tyhjä → fallback: uusin sd3 / korkein trueWin.'));
  // lineage grouped by arc
  var byArc={}; reg.forEach(function(r){ (byArc[r.arc]=byArc[r.arc]||[]).push(r); });
  var arcs=Object.keys(byArc).sort();
  var rows='';
  var arcNote={ 'sd':'Strange-Device perusarkku (vanha)', 'sd2':'Outpost-rebalance', 'sd3':'metalli-/armeijatalous rebalance (nyk.)' };
  arcs.forEach(function(arc){
    var old=(arc!=='sd3');
    rows+='<tr style="background:var(--panel2)"><td colspan="6"><b>'+esc(arc)+'</b> <span class="dim">'+esc(arcNote[arc]||'')+(old?' · ei vertailukelpoinen nyk. arkkuun':'')+'</span></td></tr>';
    byArc[arc].slice().reverse().forEach(function(r){
      var isC=r.id===champ, isDep=Object.keys(dep).some(function(a){return dep[a]===r.id;});
      rows+='<tr class="'+(isC?'champ ':'')+(old?'oldarc':'')+'"><td>'+esc(r.id)+(isDep?' <span class="pill dep">deployed</span>':'')+(isC?' <span class="pill dep">champ</span>':'')+'</td>'
        +'<td><span class="arch">'+esc(r.type)+'</span></td>'
        +'<td>'+(r.winrate_vs_hard!=null?pct(r.winrate_vs_hard):'<span class="dim">—</span>')+'</td>'
        +'<td>'+(r.parent?esc(r.parent):'<span class="dim">—</span>')+'</td>'
        +'<td>'+(r.created_utc?esc(r.created_utc.slice(0,16).replace('T',' ')):'—')+'</td>'
        +'<td><span class="pill exp">'+esc(r.status||'')+'</span></td></tr>';
    });
  });
  cards.push(card('Lineage (rekisteri, ryhmitelty arkun mukaan)','wide',
    reg.length? '<div class="tblwrap"><table><thead><tr><th>id</th><th>type</th><th>honest win</th><th>parent</th><th>rekisteröity</th><th>status</th></tr></thead><tbody>'+rows+'</tbody></table></div>'
      : '<div class="empty">ei rekisteröityjä malleja</div>',
    'Eri arkut (sd/sd2/sd3) eivät ole vertailukelpoisia — eri pelisäännöt.'));
  fill('models', cards.join(''));
}

/* ===================== BUILD / RESEARCH ===================== */
function panelBuild(){
  var s=STATE, bs=s.buildStatus, bl=s.buildLog||[]; var cards=[];
  if(bs&&Array.isArray(bs.phases)){
    var done=bs.phases.filter(function(p){return p.status==='done';}).length;
    cards.push(card('Build-prosessi','',
      '<div class="dim" style="font-size:11px;margin-bottom:6px">'+done+'/'+bs.phases.length+' vaihetta valmis</div>'
      + bs.phases.map(function(p){ return '<div class="phase"><span class="st '+esc(p.status)+'"></span>'+esc(p.name||p.id||'')+'<span class="dim" style="margin-left:auto;font-size:10px">'+esc(p.status||'')+'</span></div>'; }).join('')));
  }
  cards.push(card('Build-loki','wide',
    bl.length? '<div class="feed">'+bl.slice().reverse().slice(0,200).map(function(e){
      var lv=e.level||'info'; var col=lv==='error'?C.bad:(lv==='warn'?C.illusion:C.muted);
      return '<div class="row"><span class="ts">'+esc((e.ts||'').slice(11,19))+'</span> <span style="color:'+col+'">'+esc(e.msg||e.message||JSON.stringify(e))+'</span></div>';
    }).join('')+'</div>' : '<div class="empty">ei build-lokia (build-log.jsonl puuttuu)</div>'));
  fill('build', cards.join(''));
}
function panelResearch(){
  var s=STATE, docs=[];
  // docs are no longer in /data.json by default; keep a graceful empty.
  var research=s.research||[];
  if(!research.length){ fill('research', card('Tutkimus','wide','<div class="empty">ei tutkimusdokumentteja (rust-trainer/*.md). Lisää data.json:iin tarvittaessa.</div>')); return; }
  if(RES_TAB>=research.length) RES_TAB=0;
  var seg=research.map(function(d,i){ return '<button class="seg'+(i===RES_TAB?' sel':'')+'" data-res="'+i+'">'+esc(d.title)+'</button>'; }).join('');
  fill('research', '<div class="card wide"><div class="segmented" style="margin-bottom:12px">'+seg+'</div><div class="md">'+md(research[RES_TAB].md)+'</div></div>');
  document.querySelectorAll('[data-res]').forEach(function(b){ b.onclick=function(){ RES_TAB=Number(b.dataset.res); panelResearch(); }; });
}

/* ===================== card / tile builders ===================== */
function card(title,cls,body,tipText){
  return '<div class="card '+(cls||'')+'"><h3>'+title+(tipText?' '+tip(tipText):'')+'</h3>'+body+'</div>';
}
function tile(lbl,val,bad){
  return '<div class="tile"><div class="lbl">'+esc(lbl)+'</div><div class="v'+(bad?' zero':'')+'">'+val+'</div></div>';
}
function fill(panel,html){ var el=document.querySelector('[data-panel="'+panel+'"]'); if(el) el.innerHTML='<div class="grid">'+html+'</div>'; }

/* ===================== render dispatch ===================== */
function renderActive(){
  if(!STATE) return;
  renderToolbar();
  switch(TAB){
    case 'overview': panelOverview(); break;
    case 'economy': panelEconomy(); break;
    case 'military': panelMilitary(); break;
    case 'opponents': panelOpponents(); break;
    case 'replay': panelReplay(); break;
    case 'spatial': panelSpatial(); break;
    case 'models': panelModels(); break;
    case 'build': panelBuild(); break;
    case 'research': panelResearch(); break;
  }
}

/* ===================== polling ===================== */
function setStatus(){
  var dot=document.getElementById('statusDot'), banner=document.getElementById('banner');
  var age=(Date.now()-LAST_OK)/1000;
  dot.className='dot'+(age>60?' dead':(age>15?' stale':''));
  dot.title='last /data.json OK '+Math.round(age)+'s sitten';
  banner.className='banner'+(age>15&&LAST_OK?' show':'');
}
function poll(){
  fetch('/data.json',{cache:'no-store'}).then(function(r){return r.json();}).then(function(d){
    STATE=d; LAST_OK=Date.now();
    renderHeader(); setStatus(); renderActive();
  }).catch(function(){ setStatus(); });
}

/* ===================== keyboard (replay) ===================== */
document.addEventListener('keydown',function(e){
  if(TAB!=='replay') return;
  var r=activeReplay(); if(!r||!r.frames) return;
  if(e.key==='ArrowRight'){ R_PLAYING=false; R_FRAME=(R_FRAME+1)%r.frames.length; drawReplayFrame(document.getElementById('replayCanvas'),r,R_FRAME); }
  else if(e.key==='ArrowLeft'){ R_PLAYING=false; R_FRAME=(R_FRAME-1+r.frames.length)%r.frames.length; drawReplayFrame(document.getElementById('replayCanvas'),r,R_FRAME); }
  else if(e.key===' '){ e.preventDefault(); R_PLAYING=!R_PLAYING; var pb=document.getElementById('replayPlay'); if(pb){ pb.textContent=R_PLAYING?'⏸ tauko':'▶ toista'; pb.className='btn'+(R_PLAYING?' on':''); } }
});

/* ===================== init ===================== */
renderTabs();
setTab(TAB);
poll();
setInterval(poll, POLL_MS);
setInterval(setStatus, 1000);
window.addEventListener('hashchange',function(){ var h=location.hash.replace('#',''); if(h&&h!==TAB&&TABS.some(function(t){return t[0]===h;})) setTab(h); });
</script>
</body>
</html>`;
