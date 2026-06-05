// One-shot: widen the shipped 63-dim policy weights to 64-dim after the
// BuildStrangeDevice intent bumped INTENT_COUNT 11→12 (policy input 63→64).
//
// The extra dim is a single new one-hot slot inserted at input index
// GLOBAL_DIM + 11 = 47 (right after the existing 11 intent one-hots, before the
// 16 local features). Inserting a ZERO weight there for every first-layer neuron
// makes the widened net score IDENTICALLY to the old champion for every existing
// candidate (whose new one-hot[47] = 0), and starts the BuildStrangeDevice intent
// at a neutral 0 weight — a faithful warm-start, not a degenerate placeholder.
// The live client keeps the old champion's strength until Phase E deploys a
// freshly-trained 64-dim net.
//
// Run: npx vite-node training/widen-weights.ts   (rewrites src/ai/nn/weights.ts)

import { writeFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { NEURAL_WEIGHTS } from '../src/ai/nn/weights';
import { GLOBAL_DIM } from '../src/ai/nn/features';
import { paramCount } from '../src/ai/nn/mlp';

const __dirname = dirname(fileURLToPath(import.meta.url));
const OUT = resolve(__dirname, '../src/ai/nn/weights.ts');

const oldArch = NEURAL_WEIGHTS.arch;
const ninOld = oldArch[0]; // 63
const h1 = oldArch[1]; // 24
const insertAt = GLOBAL_DIM + 11; // 47 — the new BuildStrangeDevice one-hot slot
const ninNew = ninOld + 1; // 64
const old = NEURAL_WEIGHTS.params;

if (ninOld !== GLOBAL_DIM + 11 + 16) throw new Error(`unexpected old input dim ${ninOld}`);

const p: number[] = [];
// Layer 0 weights (row-major by output neuron j): copy [0..insertAt), insert 0
// at insertAt, then copy the rest shifted by one.
for (let j = 0; j < h1; j++) {
  for (let i = 0; i < ninNew; i++) {
    if (i < insertAt) p.push(old[j * ninOld + i]);
    else if (i === insertAt) p.push(0);
    else p.push(old[j * ninOld + (i - 1)]);
  }
}
// Layer 0 biases + every subsequent layer: verbatim.
for (let k = h1 * ninOld; k < old.length; k++) p.push(old[k]);

const newArch = [ninNew, ...oldArch.slice(1)];
if (p.length !== paramCount(newArch)) throw new Error(`param count ${p.length} != ${paramCount(newArch)}`);

// Tiers: carry the shipped knobs, add the new `device` flag (hard/medium on, easy off).
const t = NEURAL_WEIGHTS.tiers as Record<string, Record<string, unknown>>;
const tiers = {
  easy: { ...t.easy, device: false },
  medium: { ...t.medium, device: true },
  hard: { ...t.hard, device: true },
};

const meta = {
  ...(NEURAL_WEIGHTS.meta as Record<string, unknown>),
  widenedTo64: true,
  widenNote: 'transplanted 63→64 dim (zero weight at the new BuildStrangeDevice one-hot slot); placeholder pending Phase E retraining',
};

const body =
`// AUTO-GENERATED (widened to 64-dim by training/widen-weights.ts) — do not edit by hand.
//
// Trained policy-network weights for the neural CPU opponents, plus the
// per-difficulty tier configs and (optionally) the value net + MCTS search
// configs. Pure data: imported by src/ai/nn/index.ts and shipped to the browser
// (no runtime dependency). Regenerate via training/calibrate.ts after (re)training.

import { TierConfig } from './candidates';
import type { SearchConfig } from './search';
import type { ValueNet } from './value';

export interface NeuralWeights {
  arch: number[];
  params: number[];
  tiers: { easy: TierConfig; medium: TierConfig; hard: TierConfig };
  meta: Record<string, unknown>;
  valueNet?: ValueNet;
  search?: { easy: SearchConfig; medium: SearchConfig; hard: SearchConfig };
}

export const NEURAL_WEIGHTS: NeuralWeights = {
  arch: ${JSON.stringify(newArch)},
  tiers: {
    easy: ${JSON.stringify(tiers.easy)},
    medium: ${JSON.stringify(tiers.medium)},
    hard: ${JSON.stringify(tiers.hard)},
  },
  meta: ${JSON.stringify(meta)},
  search: ${JSON.stringify(NEURAL_WEIGHTS.search)},
  params: [${p.join(',')}],
};
`;

writeFileSync(OUT, body);
console.log(`widened weights: arch ${JSON.stringify(oldArch)} -> ${JSON.stringify(newArch)}, params ${old.length} -> ${p.length}`);
