// Controller factories for the training harness.

import { AiController, PARAMS, AiParams } from '../src/managers/ai';
import { NeuralAiController, SearchWiring } from '../src/ai/nn/controller';
import { Genome } from '../src/ai/nn/mlp';
import { TierConfig } from '../src/ai/nn/candidates';
import { TRAINING_CONFIG } from '../src/ai/nn/tiers';
import { SearchConfig } from '../src/ai/nn/search';
import { ValueNet } from '../src/ai/nn/value';
import { Difficulty } from '../src/model/player';
import type { ControllerFactory } from './harness';

const PARAM_OVERRIDE: Record<string, Partial<AiParams>> = {
  easy: PARAMS.easy,
  medium: PARAMS.medium,
  hard: PARAMS.hard,
};

/**
 * Heuristic AiController at a given difficulty (the anchor opponent). The
 * harness flags every seat 'hard', so for non-hard anchors we pass the matching
 * PARAMS as a full override (AiController merges it over the player's params).
 */
export function heuristicFactory(difficulty: Difficulty = 'hard'): ControllerFactory {
  return (eh, om, pm) =>
    difficulty === 'hard'
      ? new AiController(eh, om, pm)
      : new AiController(eh, om, pm, PARAM_OVERRIDE[difficulty]);
}

/** Neural controller from a genome + tier config (defaults to full training strength). */
export function neuralFactory(genome: Genome, cfg: TierConfig = TRAINING_CONFIG): ControllerFactory {
  return (eh, om, pm, _seat, rand) => new NeuralAiController(eh, om, pm, genome, cfg, rand);
}

/**
 * Search-enabled neural controller: same policy genome + tier config but with
 * test-time MCTS attached (search.ts). `mapInfo` (seed/dims) is supplied per
 * match by the bench so the sandbox can regenerate the deterministic terrain.
 */
export function neuralSearchFactory(
  genome: Genome,
  sc: SearchConfig,
  mapInfo: { width: number; height: number; seed: number },
  cfg: TierConfig = TRAINING_CONFIG,
  valueNet: ValueNet | null = null,
): ControllerFactory {
  const wiring: SearchWiring = { config: sc, valueNet, mapInfo };
  return (eh, om, pm, _seat, rand) => new NeuralAiController(eh, om, pm, genome, cfg, rand, wiring);
}
