// Maps the network over candidate actions and selects one.
//
// The network input is [ global-features | intent one-hot | local-features ],
// a fixed-width vector independent of map size. The single linear output is the
// candidate's score. Selection is greedy (hard) or temperature-softmax with an
// optional blunder rate (weaker tiers) — these are the only knobs that separate
// the difficulty levels.

import { Genome, score as netScore } from './mlp';
import { GLOBAL_DIM } from './features';
import { Candidate, INTENT_COUNT, LOCAL_DIM, TierConfig } from './candidates';

/** MLP input width for the action-scoring network. */
export const POLICY_INPUT_DIM = GLOBAL_DIM + INTENT_COUNT + LOCAL_DIM;

/** Default architecture: input → 24 → 16 → 1 (tanh hidden, linear head). */
export const DEFAULT_ARCH = [POLICY_INPUT_DIM, 24, 16, 1];

/** Build the network input for one candidate given the shared global vector. */
export function policyInput(globalVec: number[], c: Candidate): number[] {
  const input = new Array<number>(POLICY_INPUT_DIM);
  let k = 0;
  for (let i = 0; i < globalVec.length; i++) input[k++] = globalVec[i];
  for (let i = 0; i < INTENT_COUNT; i++) input[k++] = i === c.intent ? 1 : 0;
  for (let i = 0; i < LOCAL_DIM; i++) input[k++] = c.local[i] ?? 0;
  return input;
}

export function scoreCandidate(genome: Genome, globalVec: number[], c: Candidate): number {
  return netScore(genome, policyInput(globalVec, c));
}

/**
 * Select a candidate. `rand` is the (seedable) RNG used for blunders/sampling so
 * training is reproducible; in the client pass Math.random.
 */
export function select(
  genome: Genome,
  globalVec: number[],
  candidates: Candidate[],
  cfg: TierConfig,
  rand: () => number,
): Candidate {
  if (candidates.length === 1) return candidates[0];

  // Deliberate blunder: pick a uniformly random legal intent (weak tiers).
  if (cfg.blunder > 0 && rand() < cfg.blunder) {
    return candidates[Math.floor(rand() * candidates.length)];
  }

  const scores = candidates.map((c) => scoreCandidate(genome, globalVec, c));

  if (cfg.temperature <= 1e-6) {
    let best = 0;
    for (let i = 1; i < scores.length; i++) if (scores[i] > scores[best]) best = i;
    return candidates[best];
  }

  // Temperature softmax sampling.
  const t = cfg.temperature;
  let max = -Infinity;
  for (const s of scores) if (s > max) max = s;
  let sum = 0;
  const w = scores.map((s) => {
    const e = Math.exp((s - max) / t);
    sum += e;
    return e;
  });
  let r = rand() * sum;
  for (let i = 0; i < w.length; i++) {
    r -= w[i];
    if (r <= 0) return candidates[i];
  }
  return candidates[candidates.length - 1];
}
