// Pick the champion across island runs.
//
// Loads every training/checkpoints/island-*/best.json, plays each genome on a
// fixed battery vs the hard heuristic (2-player, several seeds & map sizes incl.
// large), ranks by win-rate then dominance margin, and writes the winner to
// training/checkpoints/champion.json. A final round-robin among the top few
// breaks ties by direct self-play. Deterministic (fixed seeds).
//
//   vite-node training/tournament.ts -- [--seeds 40]

import * as fs from 'node:fs';
import * as path from 'node:path';
import { Genome } from '../src/ai/nn/mlp';
import { playMatch } from './harness';
import { heuristicFactory, neuralFactory } from './factories';
import { scoreMatch } from './evaluate';
import { TRAINING_CONFIG } from '../src/ai/nn/tiers';

function arg(name: string, def: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : def;
}
const SEEDS = parseInt(arg('seeds', '40'), 10);
const ROUND_CAP = parseInt(arg('roundCap', '120'), 10);
const DIR = arg('dir', 'training/checkpoints');

// Battery sizes — overridable with --sizes "12x12,16x14,20x15". The default
// drops the most expensive map (25x15); the champion's large-map ability is
// verified separately by tests/nnai.test.ts.
const SIZES: Array<[number, number]> = arg('sizes', '12x12,16x14,20x15')
  .split(',')
  .map((s) => s.split('x').map((n) => parseInt(n, 10)) as [number, number]);

interface Cand { name: string; genome: Genome; }

function loadCandidates(): Cand[] {
  const out: Cand[] = [];
  for (const entry of fs.readdirSync(DIR)) {
    const p = path.join(DIR, entry, 'best.json');
    if (fs.existsSync(p)) {
      try { out.push({ name: entry, genome: JSON.parse(fs.readFileSync(p, 'utf8')) }); } catch { /* skip */ }
    }
  }
  const root = path.join(DIR, 'best.json');
  if (fs.existsSync(root)) {
    try { out.push({ name: 'root', genome: JSON.parse(fs.readFileSync(root, 'utf8')) }); } catch { /* skip */ }
  }
  return out;
}

/** Win-rate + mean reward of a genome vs the hard heuristic over the battery. */
function vsHeuristic(genome: Genome): { winRate: number; reward: number; games: number } {
  let wins = 0, reward = 0, games = 0;
  for (const [w, h] of SIZES) {
    for (let s = 1; s <= SEEDS; s++) {
      const r = playMatch({ width: w, height: h, seed: s, roundCap: ROUND_CAP, factories: [neuralFactory(genome, TRAINING_CONFIG), heuristicFactory('hard')] });
      if (r.winnerSeat === 0) wins++;
      reward += scoreMatch(r, 0, ROUND_CAP);
      games++;
    }
  }
  return { winRate: wins / games, reward: reward / games, games };
}

const cands = loadCandidates();
if (cands.length === 0) { console.error(`no candidates under ${DIR}`); process.exit(1); }
console.log(`evaluating ${cands.length} candidates vs hard heuristic (${SIZES.length}×${SEEDS} games each)…`);

const ranked = cands
  .map((c) => ({ ...c, ...vsHeuristic(c.genome) }))
  .sort((a, b) => b.winRate - a.winRate || b.reward - a.reward);

for (const r of ranked) {
  console.log(`  ${r.name.padEnd(12)} winVsHeur=${(r.winRate * 100).toFixed(1)}%  reward=${r.reward.toFixed(3)}`);
}

const champ = ranked[0];
const champPath = path.join(DIR, 'champion.json');
fs.writeFileSync(champPath, JSON.stringify(champ.genome));
console.log(`\nchampion: ${champ.name}  winVsHeur=${(champ.winRate * 100).toFixed(1)}%  -> ${champPath}`);
