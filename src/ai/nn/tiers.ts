// Difficulty tier configs for the neural AI. A single trained network powers
// all three tiers; only these knobs differ. The values here are sensible
// defaults — calibration (training/calibrate.ts) may overwrite the medium/easy
// knobs in weights.ts to hit the target win-rates, but these keep the game
// playable even before/without calibration.

import { TierConfig } from './candidates';
import { SearchConfig } from './search';
import { SpatialSearchConfig, C_PUCT, TAU } from './spatial_search';

/**
 * Per-tier test-time MCTS configs. ALL tiers work with NO value net (static leaf
 * fallback) so they ship immediately; the value net is used only if supplied and
 * the leaf is `value`. Final-choice softening (temperature/blunder) mirrors the
 * policy.select cascade but over visit counts.
 *
 * Empirically (see report): static leaf at high sims is the strongest affordable
 * hard config in node within the ~2.5 s/move budget, so hard uses `static`.
 */
export const HARD_SEARCH: SearchConfig = {
  nSims: 400,
  cPuct: 1.5,
  tauPrior: 1.0,
  leafEval: { kind: 'static' },
  roundCap: 400,
  seed: 0x5ea2c4,
  timeBudgetMs: 2500,
  temperature: 0, // argmax most-visited
  blunder: 0,
};

/**
 * Spatial-CNN champion deploy MCTS config — the in-browser twin of the Rust
 * deploy/bench `mcts_select` (cnn_train.rs): same PUCT (c_puct=C_PUCT), prior
 * temperature (tau=TAU), single-intent edges, opponent HARD-bot rollouts, value-
 * head leaves, and most-visited-root action selection.
 *
 * SIMS REDUCED 64 → 32 for the in-browser single-threaded JS budget. Each sim
 * rebuilds a sandbox and rolls the opponents a full turn, so cost scales ~linearly
 * with sims: measured (under heavy CPU contention) ≈350 ms @16, ≈1.2 s @32,
 * ≈2.6 s @64 per decision on a developed 28-tile board. At the Rust bench's 64 a
 * single decision can approach ~2.5 s — too slow for a turn with several decisions —
 * whereas 32 keeps it ~1 s/decision while still searching deep enough to beat the
 * greedy policy. The 2.5 s wall-clock cap is a hard safety bound against
 * pathological branching (it rarely trips at 32). Raise back to 64 if running on a
 * fast deploy host with no competing load.
 */
export const SPATIAL_HARD_SEARCH: SpatialSearchConfig = {
  nSims: 32,
  cPuct: C_PUCT,
  tauPrior: TAU,
  timeBudgetMs: 2500,
};

export const MEDIUM_SEARCH: SearchConfig = {
  nSims: 150,
  cPuct: 1.5,
  tauPrior: 1.0,
  leafEval: { kind: 'static' },
  roundCap: 400,
  seed: 0x5ea2c4,
  timeBudgetMs: 1200,
  temperature: 0.5,
  blunder: 0.05,
};

export const EASY_SEARCH: SearchConfig = {
  nSims: 30,
  cPuct: 1.5,
  tauPrior: 1.0,
  leafEval: { kind: 'static' },
  roundCap: 400,
  seed: 0x5ea2c4,
  timeBudgetMs: 500,
  temperature: 1.2,
  blunder: 0.25,
};

export const TIER_SEARCH: { easy: SearchConfig; medium: SearchConfig; hard: SearchConfig } = {
  easy: EASY_SEARCH,
  medium: MEDIUM_SEARCH,
  hard: HARD_SEARCH,
};

/** Full-strength config used during training and by nn-hard. */
export const TRAINING_CONFIG: TierConfig = {
  budget: 40,
  temperature: 0,
  reserve: 120,
  blunder: 0,
  experts: true,
  military: true,
  nuclear: true,
  device: true,
};

export const HARD_CONFIG: TierConfig = { ...TRAINING_CONFIG };

/** A good human beats it, but only just. */
export const MEDIUM_CONFIG: TierConfig = {
  budget: 14,
  temperature: 0.6,
  reserve: 160,
  blunder: 0.08,
  experts: true,
  military: true,
  nuclear: false,
  device: true,
};

/** Naturally easy to beat. */
export const EASY_CONFIG: TierConfig = {
  budget: 6,
  temperature: 1.2,
  reserve: 220,
  blunder: 0.25,
  experts: false,
  military: false,
  nuclear: false,
  device: false,
};
