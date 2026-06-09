// Web Worker entry for the neural spatial MCTS search. The search
// (spatial_search.ts → selectSpatialMcts) is 100% Phaser/DOM-free: it builds a
// headless sandbox from a JSON snapshot and runs N PUCT sims, returning the
// chosen candidate INDEX into enumerate() at the root state. Running it here
// keeps the main thread free to render at 60fps while a neural CPU "thinks".
//
// Protocol (main thread ⇄ worker), all messages are structured-clone-safe JSON:
//   → { kind: 'init', weights, cfg, sc }          construct the net + hold config
//   → { kind: 'search', reqId, snapshot, playerNum }  run one search
//   ← { kind: 'ready' }                            init acknowledged
//   ← { kind: 'result', reqId, chosenIndex, intent }  search done
//   ← { kind: 'error', reqId?, message }           construction / search failure
//
// The worker is READ-ONLY: it never mutates a live engine (there is none here),
// only the throwaway sandbox the search builds internally. The main thread
// re-enumerates the SAME deterministic candidate list on the live state and
// executes the index-th candidate, so the chosen move is identical to the
// in-thread search.

import { SpatialNetTS, SpatialWeights } from './spatial_net';
import { TierConfig } from './candidates';
import { SpatialSearchConfig, selectSpatialMcts } from './spatial_search';
import { GameSnapshot } from '../../managers/persistence';

export interface WorkerInitMsg {
  kind: 'init';
  weights: SpatialWeights;
  cfg: TierConfig;
  sc: SpatialSearchConfig;
}
export interface WorkerSearchMsg {
  kind: 'search';
  reqId: number;
  snapshot: GameSnapshot;
  playerNum: number;
}
export type WorkerRequest = WorkerInitMsg | WorkerSearchMsg;

export interface WorkerReadyMsg { kind: 'ready' }
export interface WorkerResultMsg {
  kind: 'result';
  reqId: number;
  /**
   * Chosen candidate INDEX into enumerate() at the snapshot's root state. The
   * main thread re-enumerates the SAME deterministic list on the live state and
   * executes the index-th candidate — identical to the in-thread search.
   */
  chosenIndex: number;
}
export interface WorkerErrorMsg { kind: 'error'; reqId?: number; message: string }
export type WorkerResponse = WorkerReadyMsg | WorkerResultMsg | WorkerErrorMsg;

let net: SpatialNetTS | null = null;
let cfg: TierConfig | null = null;
let sc: SpatialSearchConfig | null = null;

const post = (msg: WorkerResponse): void => {
  (self as unknown as { postMessage: (m: WorkerResponse) => void }).postMessage(msg);
};

self.onmessage = (ev: MessageEvent<WorkerRequest>): void => {
  const msg = ev.data;
  try {
    if (msg.kind === 'init') {
      net = new SpatialNetTS(msg.weights);
      cfg = msg.cfg;
      sc = msg.sc;
      post({ kind: 'ready' });
      return;
    }
    if (msg.kind === 'search') {
      if (!net || !cfg || !sc) {
        post({ kind: 'error', reqId: msg.reqId, message: 'worker not initialised' });
        return;
      }
      const chosenIndex = selectSpatialMcts(net, msg.snapshot, msg.playerNum, cfg, sc);
      post({ kind: 'result', reqId: msg.reqId, chosenIndex });
    }
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    post({ kind: 'error', reqId: msg.kind === 'search' ? msg.reqId : undefined, message });
  }
};
