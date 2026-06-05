// Shared shape of a CPU controller, satisfied by both the heuristic
// AiController and the NeuralAiController. main.ts and the training harness
// program against this so the two AIs are interchangeable per seat.

import type { PlayerBase } from '../model/player';

export interface ICpuController {
  placeHeadquarters(player: PlayerBase): void;
  planTurn(player: PlayerBase): Generator<void>;
  playTurn(player: PlayerBase): void;
}
