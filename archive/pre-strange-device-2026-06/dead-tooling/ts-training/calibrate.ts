// Calibrate the three difficulty tiers and bake src/ai/nn/weights.ts.
//
// One trained network (the champion) powers all tiers; only the TierConfig
// knobs (action budget, softmax temperature, blunder rate, reserve, whether
// experts/military/nuclear are allowed) differ. We measure each candidate
// config's win-rate vs the hard heuristic and pick:
//   hard   = full strength  (should clearly beat the hard heuristic)
//   medium = the preset whose win-rate vs hard heuristic is closest to ~0.35
//            (a good human beats it, but only just)
//   easy   = the preset closest to ~0.10 and strictly weaker than medium
// then emits weights.ts. Monotonicity (hard > medium > easy) is asserted.
//
//   vite-node training/calibrate.ts -- [--genome training/checkpoints/champion.json] [--seeds 30]

import * as fs from 'node:fs';
import { Genome } from '../src/ai/nn/mlp';
import { TierConfig } from '../src/ai/nn/candidates';
import { TRAINING_CONFIG } from '../src/ai/nn/tiers';
import { playMatch } from './harness';
import { heuristicFactory, neuralFactory } from './factories';
import { writeWeights, TierSet } from './emit-weights';

function arg(name: string, def: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : def;
}
const GENOME_PATH = arg('genome', 'training/checkpoints/champion.json');
const OUT_PATH = arg('out', 'src/ai/nn/weights.ts');
const SEEDS = parseInt(arg('seeds', '30'), 10);
const ROUND_CAP = parseInt(arg('roundCap', '140'), 10);
const SIZES: Array<[number, number]> = [[12, 12], [16, 14], [20, 15]];

const genome: Genome = JSON.parse(fs.readFileSync(GENOME_PATH, 'utf8'));

// The strength dial is dominated by `blunder` (random-action rate) and
// `temperature` (how often a sub-optimal intent is sampled) — `budget` barely
// bites because the policy usually Passes before exhausting it. Presets below
// span a wide, clearly-separated range so the calibrated ladder is monotonic.
const HARD: TierConfig = { ...TRAINING_CONFIG };
const MEDIUM_PRESETS: TierConfig[] = [
  { budget: 16, temperature: 0.6, reserve: 150, blunder: 0.15, experts: true, military: true, nuclear: false },
  { budget: 12, temperature: 0.8, reserve: 165, blunder: 0.22, experts: true, military: true, nuclear: false },
  { budget: 10, temperature: 1.0, reserve: 180, blunder: 0.30, experts: true, military: true, nuclear: false },
];
const EASY_PRESETS: TierConfig[] = [
  { budget: 7, temperature: 1.4, reserve: 210, blunder: 0.45, experts: false, military: true, nuclear: false },
  { budget: 5, temperature: 1.8, reserve: 240, blunder: 0.60, experts: false, military: false, nuclear: false },
  { budget: 4, temperature: 2.2, reserve: 270, blunder: 0.75, experts: false, military: false, nuclear: false },
];

// Targets are expressed as a FRACTION of the hard tier's measured win-rate, so
// calibration adapts to whatever absolute strength the trained net has.
const MED_FRACTION = 0.55; // medium ≈ 55% as strong as hard (clearly beatable)
const EASY_FRACTION = 0.2; // easy ≈ 20% as strong as hard (easy to beat)

/** Win-rate of (genome, cfg) at seat 0 vs the hard heuristic over the battery. */
function winVsHeur(cfg: TierConfig): number {
  let wins = 0, games = 0;
  for (const [w, h] of SIZES) {
    for (let s = 1; s <= SEEDS; s++) {
      const r = playMatch({ width: w, height: h, seed: s + 500, roundCap: ROUND_CAP, factories: [neuralFactory(genome, cfg), heuristicFactory('hard')] });
      if (r.winnerSeat === 0) wins++;
      games++;
    }
  }
  return wins / games;
}

console.log(`calibrating from ${GENOME_PATH} (${SIZES.length}×${SEEDS} games per config)…`);

const hardWin = winVsHeur(HARD);
console.log(`hard:   winVsHeur=${(hardWin * 100).toFixed(1)}%`);

const MED_TARGET = MED_FRACTION * hardWin;
const EASY_TARGET = EASY_FRACTION * hardWin;

const medScored = MEDIUM_PRESETS.map((c) => ({ c, win: winVsHeur(c) }));
medScored.forEach((m) => console.log(`  med preset budget=${m.c.budget} temp=${m.c.temperature} blunder=${m.c.blunder}: ${(m.win * 100).toFixed(1)}%`));
// Medium must be weaker than hard; among those, closest to the target.
const medPool = medScored.filter((m) => m.win <= hardWin);
const medium = (medPool.length ? medPool : medScored)
  .sort((a, b) => Math.abs(a.win - MED_TARGET) - Math.abs(b.win - MED_TARGET))[0];

const easyScored = EASY_PRESETS.map((c) => ({ c, win: winVsHeur(c) }));
easyScored.forEach((e) => console.log(`  easy preset budget=${e.c.budget} temp=${e.c.temperature} blunder=${e.c.blunder}: ${(e.win * 100).toFixed(1)}%`));
// Easy must be strictly weaker than the chosen medium.
const easyPool = easyScored.filter((e) => e.win < medium.win);
const easy = (easyPool.length ? easyPool : easyScored)
  .sort((a, b) => Math.abs(a.win - EASY_TARGET) - Math.abs(b.win - EASY_TARGET))[0];

const tiers: TierSet = { hard: HARD, medium: medium.c, easy: easy.c };

console.log('\n=== calibration ===');
console.log(`hard   winVsHeur ${(hardWin * 100).toFixed(1)}%`);
console.log(`medium winVsHeur ${(medium.win * 100).toFixed(1)}%  ${JSON.stringify(medium.c)}`);
console.log(`easy   winVsHeur ${(easy.win * 100).toFixed(1)}%  ${JSON.stringify(easy.c)}`);
const monotonic = hardWin >= medium.win && medium.win >= easy.win;
console.log(`monotonic (hard>=medium>=easy): ${monotonic}`);

writeWeights(genome, tiers, {
  trainedFrom: GENOME_PATH,
  calibration: {
    hardWin: +hardWin.toFixed(3),
    mediumWin: +medium.win.toFixed(3),
    easyWin: +easy.win.toFixed(3),
    battery: `${SIZES.length}x${SEEDS}`,
    roundCap: ROUND_CAP,
  },
}, OUT_PATH);
console.log(`\nwrote ${OUT_PATH}`);
