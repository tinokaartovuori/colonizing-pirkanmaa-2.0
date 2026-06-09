// Shared shape of a CPU controller, satisfied by both the heuristic
// AiController and the NeuralAiController. main.ts and the training harness
// program against this so the two AIs are interchangeable per seat.

import type { PlayerBase } from '../model/player';

export interface ICpuController {
  placeHeadquarters(player: PlayerBase): void;
  planTurn(player: PlayerBase): Generator<void>;
  playTurn(player: PlayerBase): void;
  /**
   * Optional async turn generator. When present (the NeuralAiController), the
   * heavy spatial MCTS step runs in a Web Worker so the main thread keeps
   * rendering at 60fps; the browser turn-driver prefers it over `planTurn` and
   * advances it with `await steps.next()`. Absent on the heuristic controller,
   * which drives through `planTurn` (sync) unchanged.
   */
  planTurnAsync?(player: PlayerBase): AsyncGenerator<void>;
}
