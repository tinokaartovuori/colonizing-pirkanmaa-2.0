// Public entry point for the neural AI: turns an nn-* difficulty into a ready
// NeuralAiController, using the baked, trained weights. This is what main.ts
// calls — the heuristic AiController is untouched and used for easy/medium/hard.

import type { GameEventHandler } from '../../managers/gameeventhandler';
import type { ObjectManager } from '../../managers/objectmanager';
import type { PlayerManager } from '../../managers/playermanager';
import type { NeuralDifficulty } from '../../model/player';
import { Genome } from './mlp';
import { TierConfig } from './candidates';
import { NeuralAiController, SearchWiring, SpatialSearchWiring } from './controller';
import { NEURAL_WEIGHTS } from './weights';
import { NEURAL_MODELS_BY_ID } from './models';
import { TIER_SEARCH, HARD_CONFIG, HARD_SEARCH, SPATIAL_HARD_SEARCH } from './tiers';
import type { SearchConfig } from './search';
import type { ValueNet } from './value';
import { SpatialNetTS } from './spatial_net';
import {
  SPATIAL_CHAMPION_ID, SPATIAL_CHAMPION_LABEL, SPATIAL_CHAMPION_NOTE, SPATIAL_CHAMPION_WEIGHTS,
} from './models_spatial';
import type { SpatialWeights } from './spatial_net';
import { SPATIAL_ROSTER_BY_ID } from './models_spatial_roster';

/** Difficulty key for the bundled CNN champion (reachable as `model:<id>`). */
export const SPATIAL_CHAMPION_DIFFICULTY = `model:${SPATIAL_CHAMPION_ID}`;
export { SPATIAL_CHAMPION_ID, SPATIAL_CHAMPION_LABEL, SPATIAL_CHAMPION_NOTE };
export { SPATIAL_ROSTER, SPATIAL_ROSTER_BY_ID } from './models_spatial_roster';
export type { SpatialRosterModel } from './models_spatial_roster';

/** A zero MLP genome the spatial-CNN controller carries but never uses for
 *  scoring (the spatial net drives all decisions). Sized to the policy input. */
function spatialPlaceholderGenome(): Genome {
  return { arch: [1, 1], params: [0, 0] };
}

const TIER_KEY: Record<NeuralDifficulty, 'easy' | 'medium' | 'hard'> = {
  'nn-easy': 'easy',
  'nn-medium': 'medium',
  'nn-hard': 'hard',
};

export function neuralTierConfig(difficulty: NeuralDifficulty): TierConfig {
  return NEURAL_WEIGHTS.tiers[TIER_KEY[difficulty]];
}

export function neuralGenome(): Genome {
  return { arch: NEURAL_WEIGHTS.arch, params: NEURAL_WEIGHTS.params };
}

/** Per-tier MCTS config (shipped override in weights.ts, else tiers.ts default). */
export function neuralSearchConfig(difficulty: NeuralDifficulty): SearchConfig {
  const key = TIER_KEY[difficulty];
  return (NEURAL_WEIGHTS.search ?? TIER_SEARCH)[key];
}

/** Optional shipped value net (absent ⇒ static-leaf fallback). */
export function neuralValueNet(): ValueNet | null {
  return NEURAL_WEIGHTS.valueNet ?? null;
}

/**
 * Build a neural CPU controller for the given nn-* difficulty.
 *
 * If `mapInfo` (seed + dimensions) is supplied, the controller runs test-time
 * MCTS (search.ts) for its discretionary decisions — this is what the in-game
 * nn-* opponents use. Without `mapInfo` the controller is byte-identical to the
 * pre-MCTS behaviour (search OFF), preserving the existing nn tiers / parity /
 * golden traces for tests and training.
 */
export function createNeuralController(
  eh: GameEventHandler,
  om: ObjectManager,
  pm: PlayerManager,
  difficulty: NeuralDifficulty,
  rand: () => number = Math.random,
  mapInfo?: { width: number; height: number; seed: number },
): NeuralAiController {
  const search: SearchWiring | undefined = mapInfo
    ? { config: neuralSearchConfig(difficulty), valueNet: neuralValueNet(), mapInfo }
    : undefined;
  return new NeuralAiController(
    eh, om, pm, neuralGenome(), neuralTierConfig(difficulty), rand, search,
  );
}

/**
 * Resolve a `model:<id>` id to its bundled spatial-CNN weights, or null if the
 * id is not a spatial net. Checks the headline champion first (default
 * fallback), then the multi-model roster (`kalevi` / `gunnar` / …).
 */
function lookupSpatialWeights(modelId: string): SpatialWeights | null {
  if (modelId === SPATIAL_CHAMPION_ID) return SPATIAL_CHAMPION_WEIGHTS;
  return SPATIAL_ROSTER_BY_ID[modelId]?.weights ?? null;
}

/**
 * Build a controller for a specific bundled trained model (difficulty
 * `model:<id>`). Plays at full strength (hard tier + hard MCTS when `mapInfo`
 * is supplied). This is the entry point for the named "Trained: …" opponents;
 * the tiered nn-* champion (createNeuralController) is untouched.
 */
export function createModelController(
  eh: GameEventHandler,
  om: ObjectManager,
  pm: PlayerManager,
  modelId: string,
  rand: () => number = Math.random,
  mapInfo?: { width: number; height: number; seed: number },
): NeuralAiController {
  // Bundled CNN AlphaZero nets (the headline champion in models_spatial.ts plus
  // the multi-model roster in models_spatial_roster.ts, e.g. `model:kalevi` /
  // `model:gunnar`): drive the controller with the trained spatial net (board
  // planes + per-tile target embed) at HARD strength. No MLP genome — the
  // deployed nets play net-greedy (their benchmarked policy mode), with the
  // army-economy scaffold the nets were trained on.
  const spatialWeights = lookupSpatialWeights(modelId);
  if (spatialWeights) {
    const net = new SpatialNetTS(spatialWeights);
    // With map info, deploy at FULL bench strength via the spatial deploy MCTS
    // (policy prior + value-head leaves, sims≈64 — the Rust deploy config). Without
    // it (e.g. golden-trace export / parity), fall back to the greedy net policy so
    // those paths stay byte-identical. The army-economy scaffold runs first either way.
    const spatialSearch: SpatialSearchWiring | undefined = mapInfo
      ? { config: SPATIAL_HARD_SEARCH, mapInfo }
      : undefined;
    return new NeuralAiController(
      eh, om, pm, spatialPlaceholderGenome(), HARD_CONFIG, rand, undefined, net, spatialSearch,
    );
  }
  const model = NEURAL_MODELS_BY_ID[modelId];
  if (!model) throw new Error(`unknown neural model: ${modelId}`);
  const search: SearchWiring | undefined = mapInfo
    ? { config: HARD_SEARCH, valueNet: null, mapInfo }
    : undefined;
  return new NeuralAiController(eh, om, pm, model.genome, HARD_CONFIG, rand, search);
}

export { NeuralAiController } from './controller';
export { NEURAL_MODELS } from './models';
