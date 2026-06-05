// HTML training-progress dashboard generator.
//
// Reads a neuroevolution log.jsonl (one JSON record per generation, written by
// the Rust trainer / TS evolve.ts) and emits a SELF-CONTAINED single-file HTML
// report at rust-trainer/report/index.html. The parsed log is inlined as a JS
// array and charts are drawn with a tiny hand-rolled inline-SVG routine, so the
// report works fully offline (no CDN, no build step).
//
// Run:
//   npx vite-node training/make-dashboard.ts -- [--log <path>] [--out <path>] [--benchmark <path>]

import { existsSync, readFileSync, mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';

const REPO_ROOT = resolve(dirname(new URL(import.meta.url).pathname), '..');
const DEFAULT_LOG = 'rust-trainer/checkpoints/log.jsonl';
const SMOKE_LOG = 'rust-trainer/checkpoints/smoke/log.jsonl';
const DEFAULT_OUT = 'rust-trainer/report/index.html';
const DEFAULT_BENCH = 'rust-trainer/checkpoints/benchmark.json';

interface LogRow {
  gen: number;
  bestFit?: number | null;
  meanFit?: number | null;
  medianFit?: number | null;
  fitStd?: number | null;
  sigma?: number | null;
  wT?: number | null;
  avgGameLen?: number | null;
  bankruptRate?: number | null;
  populationDiversity?: number | null;
  gamesPerSec?: number | null;
  winRateVsHeur?: number | null;
  [k: string]: unknown;
}

function parseArgs(argv: string[]): { log?: string; out?: string; benchmark?: string } {
  const args = argv.includes('--') ? argv.slice(argv.indexOf('--') + 1) : argv.slice(2);
  const out: { log?: string; out?: string; benchmark?: string } = {};
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === '--log') out.log = args[++i];
    else if (a === '--out') out.out = args[++i];
    else if (a === '--benchmark') out.benchmark = args[++i];
  }
  return out;
}

function resolveLog(explicit?: string): string {
  if (explicit) {
    const p = resolve(REPO_ROOT, explicit);
    if (!existsSync(p)) throw new Error(`log not found: ${explicit}`);
    return p;
  }
  const def = resolve(REPO_ROOT, DEFAULT_LOG);
  if (existsSync(def)) return def;
  const smoke = resolve(REPO_ROOT, SMOKE_LOG);
  if (existsSync(smoke)) return smoke;
  throw new Error(`no log at ${DEFAULT_LOG} or ${SMOKE_LOG}`);
}

function parseLog(path: string): LogRow[] {
  const rows: LogRow[] = [];
  for (const line of readFileSync(path, 'utf8').split('\n')) {
    const s = line.trim();
    if (!s) continue;
    try { rows.push(JSON.parse(s) as LogRow); } catch { /* skip malformed line */ }
  }
  return rows;
}

function num(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}
function fmt(v: number | null, digits = 3): string {
  return v == null ? '—' : v.toFixed(digits);
}
function pct(v: number | null): string {
  return v == null ? '—' : `${(v * 100).toFixed(1)}%`;
}

function main(): void {
  const opts = parseArgs(process.argv);
  const logPath = resolveLog(opts.log);
  const rows = parseLog(logPath);
  if (rows.length === 0) throw new Error(`log is empty: ${logPath}`);

  // Optional benchmark.json feeds the latest measured win-rate into the header.
  let benchWinRate: number | null = null;
  const benchPath = resolve(REPO_ROOT, opts.benchmark ?? DEFAULT_BENCH);
  if (existsSync(benchPath)) {
    try { benchWinRate = num((JSON.parse(readFileSync(benchPath, 'utf8')) as { winRate?: unknown }).winRate); }
    catch { /* ignore bad benchmark file */ }
  }

  const last = rows[rows.length - 1];
  const gens = rows.length;
  const latestBest = num(last.bestFit);
  const latestWinHeur = num(last.winRateVsHeur);
  const avgGps = (() => {
    const vals = rows.map((r) => num(r.gamesPerSec)).filter((v): v is number => v != null);
    return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
  })();

  const data = rows.map((r) => ({
    gen: r.gen,
    bestFit: num(r.bestFit),
    meanFit: num(r.meanFit),
    medianFit: num(r.medianFit),
    fitStd: num(r.fitStd),
    sigma: num(r.sigma),
    wT: num(r.wT),
    avgGameLen: num(r.avgGameLen),
    bankruptRate: num(r.bankruptRate),
    populationDiversity: num(r.populationDiversity),
    gamesPerSec: num(r.gamesPerSec),
    winRateVsHeur: num(r.winRateVsHeur),
  }));

  const html = renderHtml({
    logPath: logPath.replace(REPO_ROOT + '/', ''),
    gens,
    latestBest,
    latestWinHeur,
    benchWinRate,
    avgGps,
    data,
  });

  const outPath = resolve(REPO_ROOT, opts.out ?? DEFAULT_OUT);
  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, html);
  console.log(`Wrote dashboard: ${outPath.replace(REPO_ROOT + '/', '')}`);
  console.log(`  generations: ${gens}`);
  console.log(`  latest best fitness: ${fmt(latestBest)}`);
  console.log(`  latest winRateVsHeur: ${pct(latestWinHeur)}`);
  if (benchWinRate != null) console.log(`  benchmark win-rate: ${pct(benchWinRate)}`);
  console.log(`  file size: ${html.length} bytes`);
}

interface RenderCtx {
  logPath: string;
  gens: number;
  latestBest: number | null;
  latestWinHeur: number | null;
  benchWinRate: number | null;
  avgGps: number | null;
  data: Array<Record<string, number | null>>;
}

function renderHtml(ctx: RenderCtx): string {
  const dataJson = JSON.stringify(ctx.data);
  const headerWin = ctx.latestWinHeur != null ? pct(ctx.latestWinHeur)
    : ctx.benchWinRate != null ? `${pct(ctx.benchWinRate)} (benchmark)` : '—';

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Colonizing Pirkanmaa — Neural AI Training Dashboard</title>
<style>
  :root { --bg:#0f1419; --panel:#1a2027; --ink:#e6edf3; --muted:#8b97a3;
          --grid:#2a323c; --best:#4dd2a0; --mean:#5aa9ff; --median:#c792ea;
          --win:#ffcb6b; --len:#82aaff; --bank:#ff6b6b; --div:#7fdbff;
          --sigma:#ff9e64; --wt:#b388ff; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:14px/1.5 ui-monospace,Menlo,Consolas,monospace; padding:24px; }
  h1 { font-size:20px; margin:0 0 4px; }
  .sub { color:var(--muted); margin:0 0 20px; font-size:12px; }
  .summary { display:flex; flex-wrap:wrap; gap:12px; margin-bottom:24px; }
  .stat { background:var(--panel); border:1px solid var(--grid); border-radius:8px;
          padding:12px 16px; min-width:140px; }
  .stat .k { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.06em; }
  .stat .v { font-size:22px; font-weight:600; margin-top:4px; }
  .charts { display:grid; grid-template-columns:repeat(auto-fit,minmax(380px,1fr)); gap:16px; }
  .chart { background:var(--panel); border:1px solid var(--grid); border-radius:8px; padding:14px; }
  .chart h2 { font-size:13px; margin:0 0 10px; font-weight:600; }
  .legend { display:flex; flex-wrap:wrap; gap:14px; margin-top:8px; font-size:11px; color:var(--muted); }
  .legend span { display:inline-flex; align-items:center; gap:5px; }
  .swatch { width:12px; height:3px; border-radius:2px; display:inline-block; }
  svg { width:100%; height:200px; display:block; }
  .axis { fill:var(--muted); font-size:10px; }
  .gridline { stroke:var(--grid); stroke-width:1; }
  .empty { color:var(--muted); font-size:12px; padding:20px 0; text-align:center; }
</style>
</head>
<body>
<h1>Neural AI Training Dashboard</h1>
<p class="sub">Colonizing Pirkanmaa &middot; log: ${escapeHtml(ctx.logPath)} &middot; generated ${new Date().toISOString()}</p>

<div class="summary">
  <div class="stat"><div class="k">Generations</div><div class="v">${ctx.gens}</div></div>
  <div class="stat"><div class="k">Latest best fitness</div><div class="v">${fmt(ctx.latestBest)}</div></div>
  <div class="stat"><div class="k">Win-rate vs hard</div><div class="v">${headerWin}</div></div>
  <div class="stat"><div class="k">Throughput</div><div class="v">${ctx.avgGps == null ? '—' : ctx.avgGps.toFixed(0)} <span style="font-size:12px;color:var(--muted)">games/s</span></div></div>
</div>

<div class="charts" id="charts"></div>

<script>
const DATA = ${dataJson};

// --- tiny inline-SVG line chart ------------------------------------------
const W = 380, H = 200, PAD = { l: 44, r: 12, t: 10, b: 24 };
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
// Build a chart: series = [{label, color, values:[number|null]}], gens = number[]
function chart(title, series, opts) {
  opts = opts || {};
  const card = document.createElement('div');
  card.className = 'chart';
  const h = document.createElement('h2');
  h.textContent = title;
  card.appendChild(h);

  const hasData = series.some(s => s.values.some(v => v != null));
  if (!hasData) {
    const e = document.createElement('div');
    e.className = 'empty';
    e.textContent = 'no data for this metric';
    card.appendChild(e);
    return card;
  }

  const gens = DATA.map(d => d.gen);
  const gMin = Math.min.apply(null, gens), gMax = Math.max.apply(null, gens);
  let [yLo, yHi] = opts.range || extent(series);
  const x = g => gMax === gMin ? PAD.l + (W - PAD.l - PAD.r) / 2
    : PAD.l + (g - gMin) / (gMax - gMin) * (W - PAD.l - PAD.r);
  const y = v => PAD.t + (1 - (v - yLo) / (yHi - yLo)) * (H - PAD.t - PAD.b);

  const svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H, preserveAspectRatio: 'none' });

  // horizontal gridlines + y labels
  const TICKS = 4;
  for (let i = 0; i <= TICKS; i++) {
    const v = yLo + (yHi - yLo) * i / TICKS;
    const yy = y(v);
    svg.appendChild(svgEl('line', { class: 'gridline', x1: PAD.l, y1: yy, x2: W - PAD.r, y2: yy }));
    const t = svgEl('text', { class: 'axis', x: PAD.l - 6, y: yy + 3, 'text-anchor': 'end' });
    t.textContent = (opts.pct ? (v * 100).toFixed(0) + '%' : Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2));
    svg.appendChild(t);
  }
  // x labels (first / last gen)
  const xt0 = svgEl('text', { class: 'axis', x: PAD.l, y: H - 8, 'text-anchor': 'start' });
  xt0.textContent = 'gen ' + gMin;
  const xt1 = svgEl('text', { class: 'axis', x: W - PAD.r, y: H - 8, 'text-anchor': 'end' });
  xt1.textContent = 'gen ' + gMax;
  svg.appendChild(xt0); svg.appendChild(xt1);

  // optional band (e.g. fitStd around mean)
  if (opts.band) {
    let d = '';
    const top = [], bot = [];
    DATA.forEach((row, i) => {
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

  // series lines (gaps allowed: break the path on null)
  for (const s of series) {
    let d = '', pen = false;
    DATA.forEach((row, i) => {
      const v = s.values[i];
      if (v == null) { pen = false; return; }
      const px = x(row.gen), py = y(v);
      d += (pen ? ' L' : ' M') + px + ',' + py;
      pen = true;
    });
    if (d) svg.appendChild(svgEl('path', { d, fill: 'none', stroke: s.color, 'stroke-width': '2', 'stroke-linejoin': 'round' }));
    // dots for sparse/single points so they're visible
    DATA.forEach((row, i) => {
      const v = s.values[i];
      if (v == null) return;
      svg.appendChild(svgEl('circle', { cx: x(row.gen), cy: y(v), r: '2', fill: s.color }));
    });
  }
  card.appendChild(svg);

  if (series.length > 1 || series[0].label) {
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
  return card;
}

const col = (d, k) => DATA.map(r => r[k]);
const root = document.getElementById('charts');

root.appendChild(chart('Win-rate vs hard AI', [
  { label: 'winRateVsHeur', color: getColor('--win'), values: col(DATA, 'winRateVsHeur') },
], { pct: true, range: [0, 1] }));

root.appendChild(chart('Fitness', [
  { label: 'bestFit', color: getColor('--best'), values: col(DATA, 'bestFit') },
  { label: 'meanFit', color: getColor('--mean'), values: col(DATA, 'meanFit') },
  { label: 'medianFit', color: getColor('--median'), values: col(DATA, 'medianFit') },
], { band: { center: col(DATA, 'meanFit'), width: col(DATA, 'fitStd'), color: getColor('--mean') } }));

root.appendChild(chart('Avg game length (rounds)', [
  { label: 'avgGameLen', color: getColor('--len'), values: col(DATA, 'avgGameLen') },
]));

root.appendChild(chart('Bankrupt rate', [
  { label: 'bankruptRate', color: getColor('--bank'), values: col(DATA, 'bankruptRate') },
], { pct: true, range: [0, 1] }));

root.appendChild(chart('Population diversity', [
  { label: 'populationDiversity', color: getColor('--div'), values: col(DATA, 'populationDiversity') },
]));

root.appendChild(chart('Annealing (sigma & wT)', [
  { label: 'sigma', color: getColor('--sigma'), values: col(DATA, 'sigma') },
  { label: 'wT', color: getColor('--wt'), values: col(DATA, 'wT') },
]));

function getColor(varName) {
  return getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
}
</script>
</body>
</html>
`;
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]!));
}

main();
