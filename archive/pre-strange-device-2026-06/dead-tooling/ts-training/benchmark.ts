// Cross-language benchmark: a Rust-trained champion genome vs the HELD-OUT hard
// heuristic AI.
//
// This is the milestone's measure of "how good is the AI". It loads a champion
// the SAME way the game does (a plain {arch,params} Genome via mlp.ts), then runs
// N headless games with seat 0 = NeuralAiController(champion, HARD_CONFIG) and
// seat 1 = the hard heuristic AiController. Map sizes / seeds / round caps are
// varied (evaluate.ts curriculum style, mostly 2-player for a clean signal).
//
// The hard heuristic is the held-out opponent — it lives ONLY here for
// measurement and is never fed back into training.
//
// Run:
//   npx vite-node training/benchmark.ts -- --champion <path> --games <N> [--seed S]

import { existsSync, readFileSync, mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { Genome } from '../src/ai/nn/mlp';
import { HARD_CONFIG } from '../src/ai/nn/tiers';
import { playMatch, MatchSpec, makeRng } from './harness';
import { heuristicFactory, neuralFactory } from './factories';

const REPO_ROOT = resolve(dirname(new URL(import.meta.url).pathname), '..');
const DEFAULT_CHAMPION = 'rust-trainer/checkpoints/champion.json';
const SMOKE_CHAMPION = 'rust-trainer/checkpoints/smoke/champion.json';
const OUT_PATH = 'rust-trainer/checkpoints/benchmark.json';

// Mostly 2-player for a clean head-to-head signal; the curriculum's small/medium
// map sizes keep games cheap (engine tile lookups are O(n)).
const SIZES: Array<[number, number]> = [
  [12, 12], [12, 12], [12, 12], [14, 12], [14, 12], [16, 14], [18, 14], [20, 15],
];

function parseArgs(argv: string[]): { champion?: string; games?: number; seed?: number } {
  // vite-node passes script args after a bare `--`; tolerate both forms.
  const args = argv.includes('--') ? argv.slice(argv.indexOf('--') + 1) : argv.slice(2);
  const out: { champion?: string; games?: number; seed?: number } = {};
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === '--champion') out.champion = args[++i];
    else if (a === '--games') out.games = Number(args[++i]);
    else if (a === '--seed') out.seed = Number(args[++i]);
  }
  return out;
}

function resolveChampion(explicit?: string): string {
  if (explicit) {
    const p = resolve(REPO_ROOT, explicit);
    if (!existsSync(p)) throw new Error(`champion not found: ${explicit}`);
    return p;
  }
  const def = resolve(REPO_ROOT, DEFAULT_CHAMPION);
  if (existsSync(def)) return def;
  const smoke = resolve(REPO_ROOT, SMOKE_CHAMPION);
  if (existsSync(smoke)) return smoke;
  throw new Error(`no champion at ${DEFAULT_CHAMPION} or ${SMOKE_CHAMPION}`);
}

function loadGenome(path: string): Genome {
  const g = JSON.parse(readFileSync(path, 'utf8')) as Genome;
  if (!Array.isArray(g.arch) || !Array.isArray(g.params)) {
    throw new Error(`malformed genome (need {arch,params}): ${path}`);
  }
  return g;
}

function main(): void {
  const opts = parseArgs(process.argv);
  const championRel = opts.champion ?? resolveChampion().replace(REPO_ROOT + '/', '');
  const championPath = resolveChampion(opts.champion);
  const games = opts.games && opts.games > 0 ? Math.floor(opts.games) : 50;
  const baseSeed = opts.seed ?? 1;

  const genome = loadGenome(championPath);
  // Seat 0 = champion (neural, hard config). Seat 1 = held-out hard heuristic.
  const championFactory = neuralFactory(genome, HARD_CONFIG);
  const heuristicFac = heuristicFactory('hard');

  const rand = makeRng(baseSeed);

  let wins = 0, losses = 0, timeouts = 0, ties = 0, bankrupts = 0, crashes = 0;
  let totalRounds = 0;
  let totalTileFrac = 0;

  console.log(`Benchmark: champion vs hard heuristic`);
  console.log(`  champion: ${championPath}`);
  console.log(`  games:    ${games}   (seed base ${baseSeed})\n`);

  for (let i = 0; i < games; i++) {
    const [w, h] = SIZES[Math.floor(rand() * SIZES.length)];
    const seed = 1 + Math.floor(rand() * 1000);
    const roundCap = rand() < 0.12 ? 180 : 80;
    const spec: MatchSpec = {
      width: w, height: h, seed, roundCap,
      factories: [championFactory, heuristicFac],
    };
    const r = playMatch(spec);
    totalRounds += r.rounds;
    totalTileFrac += r.tileFrac[0];
    if (r.crashed) crashes++;
    if (r.bankrupt[0]) bankrupts++;
    if (r.winnerSeat === 0) wins++;
    else if (r.winnerSeat === 1) losses++;
    else if (r.reason === 'tie') ties++;
    else timeouts++;
  }

  const rate = (n: number) => n / games;
  const summary = {
    champion: championPath,
    games,
    wins, losses, timeouts, ties, bankrupts, crashes,
    winRate: rate(wins),
    lossRate: rate(losses),
    timeoutRate: rate(timeouts),
    tieRate: rate(ties),
    bankruptRate: rate(bankrupts),
    crashRate: rate(crashes),
    avgGameLen: totalRounds / games,
    avgFinalTileFrac: totalTileFrac / games,
    timestamp: new Date().toISOString(),
  };

  console.log('Results vs hard heuristic AI:');
  console.log(`  win-rate:        ${(summary.winRate * 100).toFixed(1)}%  (${wins}/${games})`);
  console.log(`  loss-rate:       ${(summary.lossRate * 100).toFixed(1)}%  (${losses}/${games})`);
  console.log(`  timeout-rate:    ${(summary.timeoutRate * 100).toFixed(1)}%  (${timeouts}/${games})`);
  if (ties) console.log(`  tie-rate:        ${(summary.tieRate * 100).toFixed(1)}%  (${ties}/${games})`);
  console.log(`  bankrupt-rate:   ${(summary.bankruptRate * 100).toFixed(1)}%  (${bankrupts}/${games})`);
  if (crashes) console.log(`  crash-rate:      ${(summary.crashRate * 100).toFixed(1)}%  (${crashes}/${games})`);
  console.log(`  avg game length: ${summary.avgGameLen.toFixed(1)} rounds`);
  console.log(`  avg tile frac:   ${(summary.avgFinalTileFrac * 100).toFixed(1)}%`);

  const outPath = resolve(REPO_ROOT, OUT_PATH);
  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, JSON.stringify(summary, null, 2));
  console.log(`\nWrote ${OUT_PATH}`);
}

main();
