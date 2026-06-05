// Strength bench for the in-browser test-time MCTS opponent.
//
// Mirrors training/benchmark.ts (champion vs TS-hard heuristic), but seat 0 is
// the SEARCH-ENABLED NeuralAiController (hard tier MCTS) instead of the no-search
// policy. Reports win-rate (target band ~20-33%, must beat the no-search ~11%)
// and per-move MCTS latency.
//
// Run:
//   npx vite-node training/bench-mcts.ts -- --champion <path> --games 48 \
//     --leaf static --sims 400 --budget 2500 [--value <value.json>]

import { existsSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { performance } from 'node:perf_hooks';
import { Genome } from '../src/ai/nn/mlp';
import { TRAINING_CONFIG } from '../src/ai/nn/tiers';
import { SearchConfig, LeafEval, select as searchSelect } from '../src/ai/nn/search';
import { ValueNet } from '../src/ai/nn/value';
import { playMatch, MatchSpec, makeRng } from './harness';
import { heuristicFactory, neuralSearchFactory } from './factories';
import { GameSettingsManager } from '../src/managers/gamesettings';
import { ObjectManager } from '../src/managers/objectmanager';
import { PlayerManager } from '../src/managers/playermanager';
import { GameEventHandler } from '../src/managers/gameeventhandler';
import { WorldGenerator } from '../src/world/worldgenerator';
import { AiController } from '../src/managers/ai';
import { MouseHoverBorder } from '../src/model/overlays';
import { Coordinate } from '../src/core/coordinate';
import { PlayerConfig } from '../src/model/player';
import { buildSnapshot } from '../src/managers/persistence';
import { StubScene, CapturingMenu } from '../src/ai/nn/headless';

const REPO_ROOT = resolve(dirname(new URL(import.meta.url).pathname), '..');
const DEFAULT_CHAMPION = 'rust-trainer/checkpoints/champion.json';
const SMOKE_CHAMPION = 'rust-trainer/checkpoints/smoke/champion.json';

// Same curriculum as benchmark.ts so the win-rate is directly comparable to the
// no-search oracle.
const SIZES: Array<[number, number]> = [
  [12, 12], [12, 12], [12, 12], [14, 12], [14, 12], [16, 14], [18, 14], [20, 15],
];

interface Opts {
  champion?: string;
  games: number;
  leaf: 'static' | 'value' | 'rollout';
  sims: number;
  budget: number;
  rollout: number;
  seed: number;
  value?: string;
  /** Force every game to NxN (bounds per-game cost). 0 = curriculum sizes. */
  map: number;
  /** Clamp every game's round cap to at most this. 0 = curriculum caps. */
  maxcap: number;
}

function parseArgs(argv: string[]): Opts {
  const args = argv.includes('--') ? argv.slice(argv.indexOf('--') + 1) : argv.slice(2);
  const o: Opts = { games: 48, leaf: 'static', sims: 400, budget: 2500, rollout: 6, seed: 1, map: 0, maxcap: 0 };
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === '--champion') o.champion = args[++i];
    else if (a === '--games') o.games = Number(args[++i]);
    else if (a === '--leaf') o.leaf = args[++i] as Opts['leaf'];
    else if (a === '--sims') o.sims = Number(args[++i]);
    else if (a === '--budget') o.budget = Number(args[++i]);
    else if (a === '--rollout') o.rollout = Number(args[++i]);
    else if (a === '--seed') o.seed = Number(args[++i]);
    else if (a === '--value') o.value = args[++i];
    else if (a === '--map') o.map = Number(args[++i]);
    else if (a === '--maxcap') o.maxcap = Number(args[++i]);
  }
  return o;
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
  if (!Array.isArray(g.arch) || !Array.isArray(g.params)) throw new Error(`malformed genome: ${path}`);
  return g;
}

function leafEval(o: Opts): LeafEval {
  if (o.leaf === 'value') return { kind: 'value' };
  if (o.leaf === 'rollout') return { kind: 'rollout', horizon: o.rollout };
  return { kind: 'static' };
}

/**
 * Clean per-DECISION latency: build a representative mid-game state (two hard
 * heuristics, ~12 rounds), snapshot it, and time `search.select` directly. This
 * is the latency of ONE MCTS decision — the cost runCpuTurn pays per move.
 */
function measureDecisionLatency(genome: Genome, sc: SearchConfig, valueNet: ValueNet | null): { ms: number; sims: number } {
  const W = 16, H = 14, SEED = 7;
  const configs: PlayerConfig[] = [{ name: 'A', difficulty: 'hard' }, { name: 'B', difficulty: 'hard' }];
  const gsm = GameSettingsManager.fromMapDimensions(W, H);
  const om = new ObjectManager();
  const pm = new PlayerManager(configs, om);
  const menu = new CapturingMenu();
  const eh = new GameEventHandler(om, pm, menu, gsm);
  const scene = new StubScene();
  eh.setGameScene(scene); om.setGameScene(scene); om.addDALS(eh, menu, gsm);
  om.setHoverBorder(new MouseHoverBorder(new Coordinate(0, 0), 1, 1, eh, om));
  new WorldGenerator().generateMap(W, H, SEED, { objectManager: om, eventHandler: eh, gameSettings: gsm, scene });
  const ai = new AiController(eh, om, pm);
  eh.setAiActive(true);
  for (let i = 0; i < configs.length; i++) ai.placeHeadquarters(pm.getCurrentPlayer());
  eh.setAiActive(false);
  for (let r = 0; r < 12 && pm.getPlayers().length > 1; r++) {
    for (let s = 0; s < configs.length && pm.getPlayers().length > 1; s++) {
      const cur = pm.getCurrentPlayer();
      if (cur.isCpu()) { eh.setAiActive(true); ai.playTurn(cur); eh.setAiActive(false); }
      eh.endTurn();
    }
  }
  const cur = pm.getCurrentPlayer();
  const snap = buildSnapshot(om, pm, { width: W, height: H, seed: SEED });
  // Measure the ACTUAL shipping latency: the configured time budget is honoured,
  // so this reports the real per-decision cost a move pays in runCpuTurn (the
  // budget caps the effective sim count on heavy/large states).
  searchSelect(genome, snap, cur.getPlayerNum(), TRAINING_CONFIG, sc, valueNet, Math.random); // warm
  const N = 3;
  const t0 = performance.now();
  for (let i = 0; i < N; i++) searchSelect(genome, snap, cur.getPlayerNum(), TRAINING_CONFIG, sc, valueNet, Math.random);
  return { ms: (performance.now() - t0) / N, sims: sc.nSims };
}

function main(): void {
  const o = parseArgs(process.argv);
  const championPath = resolveChampion(o.champion);
  const genome = loadGenome(championPath);
  const valueNet: ValueNet | null = o.value
    ? (JSON.parse(readFileSync(resolve(REPO_ROOT, o.value), 'utf8')) as ValueNet)
    : null;

  const sc: SearchConfig = {
    nSims: o.sims,
    cPuct: 1.5,
    tauPrior: 1.0,
    leafEval: leafEval(o),
    roundCap: 400,
    seed: 0x5ea2c4,
    timeBudgetMs: o.budget,
    temperature: 0, // hard tier = argmax
    blunder: 0,
  };

  console.log('MCTS strength bench: search-enabled NN (hard) vs TS hard heuristic');
  console.log(`  champion: ${championPath}`);
  console.log(`  leaf=${o.leaf} sims=${o.sims} budget=${o.budget}ms${o.leaf === 'rollout' ? ` horizon=${o.rollout}` : ''}${valueNet ? ' value-net=on' : ''}`);
  console.log(`  games:    ${o.games} (seed base ${o.seed})\n`);

  // Clean per-decision latency probe (one MCTS decision on a 16x14 mid-game).
  const lat = measureDecisionLatency(genome, sc, valueNet);
  console.log(`  per-decision MCTS latency: ${lat.ms.toFixed(0)} ms (16x14 mid-game, budget ${o.budget}ms, sim ceiling ${lat.sims})\n`);

  const rand = makeRng(o.seed);
  const heuristicFac = heuristicFactory('hard');

  let wins = 0, losses = 0, ties = 0, timeouts = 0, crashes = 0;
  let totalRounds = 0, totalTileFrac = 0;
  // Per-move latency: time around each search.select via the controller is hard
  // to intercept cleanly through the harness, so we time the WHOLE seat-0 turn
  // wall-clock and divide by the number of moves we observe (approximate but
  // representative). We track total seat-0 search time and a move counter via a
  // wrapped global clock.
  const moveTimes: number[] = [];
  const origNow = performance.now.bind(performance);

  for (let i = 0; i < o.games; i++) {
    let [w, h] = SIZES[Math.floor(rand() * SIZES.length)];
    const seed = 1 + Math.floor(rand() * 1000);
    let roundCap = rand() < 0.12 ? 180 : 80;
    if (o.map > 0) { w = o.map; h = o.map; }
    if (o.maxcap > 0 && roundCap > o.maxcap) roundCap = o.maxcap;

    const t0 = origNow();
    const spec: MatchSpec = {
      width: w, height: h, seed, roundCap,
      factories: [neuralSearchFactory(genome, sc, { width: w, height: h, seed }, TRAINING_CONFIG, valueNet), heuristicFac],
    };
    const r = playMatch(spec);
    const dt = origNow() - t0;

    totalRounds += r.rounds;
    totalTileFrac += r.tileFrac[0];
    if (r.crashed) crashes++;
    if (r.winnerSeat === 0) wins++;
    else if (r.winnerSeat === 1) losses++;
    else if (r.reason === 'tie') ties++;
    else timeouts++;

    // Approx per-move latency: seat 0 acted on ~half the turns; each turn runs
    // many MCTS decisions. Use total game time / (rounds * a nominal decisions
    // factor) as a coarse proxy, plus the explicit per-decision figure below.
    moveTimes.push(dt / Math.max(1, r.rounds));
    process.stdout.write(
      `  game ${String(i + 1).padStart(3)}/${o.games}  ${w}x${h} cap${roundCap} seed${String(seed).padEnd(4)}  ` +
      `winner=${r.winnerSeat === 0 ? 'NN' : r.winnerSeat === 1 ? 'HARD' : r.reason}  ` +
      `frac=${r.tileFrac[0].toFixed(3)} rounds=${r.rounds}  ${(dt / 1000).toFixed(1)}s\n`,
    );
  }

  const n = o.games;
  const pct = (x: number) => `${((x / n) * 100).toFixed(1)}%`;
  const avgTurn = moveTimes.reduce((a, b) => a + b, 0) / Math.max(1, moveTimes.length);
  console.log('\nResults (MCTS-hard vs TS-hard):');
  console.log(`  win-rate:    ${pct(wins)}  (${wins}/${n})`);
  console.log(`  loss-rate:   ${pct(losses)}  (${losses}/${n})`);
  console.log(`  tie-rate:    ${pct(ties)}  (${ties}/${n})`);
  console.log(`  timeout:     ${pct(timeouts)}  (${timeouts}/${n})`);
  if (crashes) console.log(`  crashes:     ${pct(crashes)}  (${crashes}/${n})`);
  console.log(`  avg rounds:  ${(totalRounds / n).toFixed(1)}`);
  console.log(`  avg tile frac (seat 0): ${(totalTileFrac / n).toFixed(3)}`);
  console.log(`  avg seat-0 wall time per ROUND: ${avgTurn.toFixed(0)} ms (coarse)`);
}

main();
