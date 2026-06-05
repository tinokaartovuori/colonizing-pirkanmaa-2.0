// Neuroevolution (Evolution Strategies with elitism + Hall-of-Fame self-play).
//
//   vite-node training/evolve.ts -- --gens 120 --pop 30 --elite 8 --games 16
//
// Each generation: build a shared curriculum, evaluate every genome on it
// (common random numbers → fair ranking), keep the elites, breed mutated
// offspring (annealed Gaussian σ + occasional crossover), and periodically
// snapshot the best genome into the Hall of Fame so later generations must beat
// their own ancestors, not just the heuristic. The best genome and a per-gen
// log are checkpointed to training/checkpoints/ after every generation, so a
// long run can be polled and resumed.

import * as fs from 'node:fs';
import * as path from 'node:path';
import { Genome, randomGenome, paramCount } from '../src/ai/nn/mlp';
import { DEFAULT_ARCH } from '../src/ai/nn/policy';
import { makeRng } from './harness';
import { buildCurriculum, evalGenome, GenomeStats } from './evaluate';

function arg(name: string, def: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : def;
}
const GENS = parseInt(arg('gens', '120'), 10);
const POP = parseInt(arg('pop', '30'), 10);
const ELITE = parseInt(arg('elite', '8'), 10);
const GAMES = parseInt(arg('games', '16'), 10);
const SIGMA0 = parseFloat(arg('sigma', '0.18'));
const SIGMA1 = parseFloat(arg('sigmaEnd', '0.04'));
const SEED = parseInt(arg('seed', '12345'), 10);
const HOF_EVERY = parseInt(arg('hofEvery', '4'), 10);
const HOF_MAX = parseInt(arg('hofMax', '8'), 10);
const OUT = arg('out', 'training/checkpoints');
const RESUME = process.argv.includes('--resume');
const HEUR_SHARE = parseFloat(arg('heurShare', '-1')); // -1 = auto
const CAP = parseInt(arg('cap', '80'), 10);
const LONG_SHARE = parseFloat(arg('longShare', '0.15'));
// Warm-start the population from existing genomes (e.g. island bests):
// --warm "a.json,b.json,..."  — each is used as a seed, then mutated to fill POP.
const WARM = arg('warm', '');

const ARCH = DEFAULT_ARCH;
const NP = paramCount(ARCH);

fs.mkdirSync(OUT, { recursive: true });
const bestPath = path.join(OUT, 'best.json');
const hofPath = path.join(OUT, 'hof.json');
const logPath = path.join(OUT, 'log.jsonl');

// A master RNG drives genome init/mutation; per-generation curricula use a
// derived seed so the run is fully reproducible.
const masterRng = makeRng(SEED);
function gaussian(rand: () => number): number {
  const u1 = Math.max(rand(), 1e-9);
  const u2 = rand();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}
function mutate(parent: Genome, sigma: number, rand: () => number): Genome {
  const params = parent.params.slice();
  for (let i = 0; i < params.length; i++) params[i] += sigma * gaussian(rand);
  return { arch: parent.arch, params };
}
function crossover(a: Genome, b: Genome, rand: () => number): Genome {
  const params = new Array<number>(a.params.length);
  for (let i = 0; i < params.length; i++) params[i] = rand() < 0.5 ? a.params[i] : b.params[i];
  return { arch: a.arch, params };
}

// --- population init / resume ---------------------------------------------

let population: Genome[] = [];
let hof: Genome[] = [];
let startGen = 0;

if (WARM) {
  const seeds: Genome[] = WARM.split(',').map((p) => JSON.parse(fs.readFileSync(p.trim(), 'utf8')) as Genome).filter((g) => g.arch.length);
  if (seeds.length === 0) throw new Error('no warm-start genomes loaded');
  population.push(...seeds);
  // Fill the rest with mutated copies of random seeds.
  while (population.length < POP) population.push(mutate(seeds[Math.floor(masterRng() * seeds.length)], SIGMA0, masterRng));
  population = population.slice(0, POP);
  console.log(`warm-started ${seeds.length} seeds -> pop ${population.length}`);
} else if (RESUME && fs.existsSync(bestPath)) {
  const best: Genome = JSON.parse(fs.readFileSync(bestPath, 'utf8'));
  if (fs.existsSync(hofPath)) hof = JSON.parse(fs.readFileSync(hofPath, 'utf8'));
  // Seed the population around the saved best.
  population.push(best);
  for (let i = 1; i < POP; i++) population.push(mutate(best, SIGMA0, masterRng));
  if (fs.existsSync(logPath)) startGen = fs.readFileSync(logPath, 'utf8').trim().split('\n').filter(Boolean).length;
  console.log(`resumed from ${bestPath} (gen ${startGen}), hof=${hof.length}`);
} else {
  for (let i = 0; i < POP; i++) population.push(randomGenome(ARCH, masterRng, 0.4));
}

console.log(`evolve: arch=[${ARCH}] params=${NP} pop=${POP} elite=${ELITE} games/gen=${GAMES} gens=${GENS} seed=${SEED}`);

// --- evolution loop --------------------------------------------------------

const t0 = Date.now();
let globalBest: { genome: Genome; fitness: number } | null = null;

for (let gen = startGen; gen < startGen + GENS; gen++) {
  const genRng = makeRng(SEED * 1000 + gen);
  const heurShare = HEUR_SHARE >= 0 ? HEUR_SHARE : (hof.length ? 0.5 : 1);
  const tasks = buildCurriculum(genRng, { games: GAMES, hofSize: hof.length, heurShare, longShare: LONG_SHARE, cap: CAP });

  const scored = population.map((g) => {
    const s = evalGenome(g, tasks, hof);
    return { genome: g, stats: s };
  });
  scored.sort((a, b) => b.stats.fitness - a.stats.fitness);

  const best = scored[0];
  const meanFit = scored.reduce((s, x) => s + x.stats.fitness, 0) / scored.length;
  const winRateVsHeur = (x: GenomeStats) => (x.gamesVsHeur ? x.winsVsHeur / x.gamesVsHeur : 0);
  const anyBankrupt = scored.some((x) => x.stats.anyBankrupt);
  const anyCrash = scored.some((x) => x.stats.anyCrash);

  if (!globalBest || best.stats.fitness > globalBest.fitness) {
    globalBest = { genome: best.genome, fitness: best.stats.fitness };
    fs.writeFileSync(bestPath, JSON.stringify(best.genome));
  }

  // Hall of Fame: snapshot the current best every HOF_EVERY gens.
  if (gen % HOF_EVERY === 0) {
    hof.push(best.genome);
    if (hof.length > HOF_MAX) hof = hof.slice(hof.length - HOF_MAX);
    fs.writeFileSync(hofPath, JSON.stringify(hof));
  }

  const sigma = SIGMA0 + (SIGMA1 - SIGMA0) * ((gen - startGen) / Math.max(1, GENS - 1));
  const logLine = {
    gen,
    bestFit: +best.stats.fitness.toFixed(4),
    meanFit: +meanFit.toFixed(4),
    bestWinVsHeur: +winRateVsHeur(best.stats).toFixed(3),
    bestWins: best.stats.wins,
    games: best.stats.games,
    hof: hof.length,
    sigma: +sigma.toFixed(4),
    anyBankrupt,
    anyCrash,
    elapsed: +((Date.now() - t0) / 1000).toFixed(1),
  };
  fs.appendFileSync(logPath, JSON.stringify(logLine) + '\n');
  console.log(JSON.stringify(logLine));

  // Breed the next generation: keep elites, fill with mutated/crossed offspring.
  const elites = scored.slice(0, ELITE).map((x) => x.genome);
  const next: Genome[] = elites.slice(); // elitism: elites survive unchanged
  while (next.length < POP) {
    const a = elites[Math.floor(masterRng() * elites.length)];
    let child = a;
    if (masterRng() < 0.3) {
      const b = elites[Math.floor(masterRng() * elites.length)];
      child = crossover(a, b, masterRng);
    }
    next.push(mutate(child, sigma, masterRng));
  }
  population = next;
}

console.log(`done. best fitness=${globalBest?.fitness.toFixed(4)} -> ${bestPath}  (${((Date.now() - t0) / 1000).toFixed(1)}s)`);
