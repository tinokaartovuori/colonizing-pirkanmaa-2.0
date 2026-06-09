// Main-thread client for the neural spatial-MCTS Web Worker (worker.ts). One
// client = one worker, owned by a single NeuralAiController and reused across
// every turn/sim (the worker is constructed lazily on the first search). The
// worker runs the SAME selectSpatialMcts search headlessly off the main thread,
// so the UI keeps rendering at 60fps while a neural CPU thinks; the chosen
// candidate INDEX comes back and the controller executes it on the LIVE engine.
//
// All payloads are structured-clone-safe JSON (SpatialWeights / TierConfig /
// SpatialSearchConfig / GameSnapshot are plain data). If Workers are unavailable
// or the worker errors, the controller catches and falls back to the in-thread
// search — see controller.ts.

import type { SpatialWeights } from './spatial_net';
import type { TierConfig } from './candidates';
import type { SpatialSearchConfig } from './spatial_search';
import type { GameSnapshot } from '../../managers/persistence';
import type { WorkerRequest, WorkerResponse } from './worker';

interface Pending {
  resolve: (chosenIndex: number) => void;
  reject: (err: Error) => void;
}

export class MctsWorkerClient {
  private worker: Worker | null = null;
  private ready: Promise<void> | null = null;
  private nextReqId = 1;
  private readonly pending = new Map<number, Pending>();

  constructor(
    private readonly weights: SpatialWeights,
    private readonly cfg: TierConfig,
    private readonly sc: SpatialSearchConfig,
  ) {}

  /** Construct the worker + send the init message lazily, exactly once. */
  private ensureWorker(): Promise<void> {
    if (this.ready) return this.ready;
    if (typeof Worker === 'undefined') {
      this.ready = Promise.reject(new Error('Web Workers unavailable'));
      return this.ready;
    }
    this.ready = new Promise<void>((resolve, reject) => {
      let worker: Worker;
      try {
        worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
      } catch (e) {
        reject(e instanceof Error ? e : new Error(String(e)));
        return;
      }
      this.worker = worker;
      worker.onmessage = (ev: MessageEvent<WorkerResponse>) => {
        const msg = ev.data;
        if (msg.kind === 'ready') { resolve(); return; }
        if (msg.kind === 'result') {
          const p = this.pending.get(msg.reqId);
          if (p) { this.pending.delete(msg.reqId); p.resolve(msg.chosenIndex); }
          return;
        }
        if (msg.kind === 'error') {
          if (msg.reqId !== undefined) {
            const p = this.pending.get(msg.reqId);
            if (p) { this.pending.delete(msg.reqId); p.reject(new Error(msg.message)); }
          } else {
            reject(new Error(msg.message)); // init-time failure
          }
        }
      };
      worker.onerror = (ev: ErrorEvent) => {
        const err = new Error(ev.message || 'worker error');
        reject(err); // before-ready failures surface as an init rejection
        // Fail any in-flight searches so the controller can fall back.
        for (const [, p] of this.pending) p.reject(err);
        this.pending.clear();
      };
      const init: WorkerRequest = { kind: 'init', weights: this.weights, cfg: this.cfg, sc: this.sc };
      worker.postMessage(init);
    });
    return this.ready;
  }

  /**
   * Run the spatial MCTS search in the worker and resolve with the chosen
   * candidate INDEX into enumerate() at the snapshot's root state. Rejects if the
   * worker is unavailable or the search fails (the caller falls back in-thread).
   */
  async searchViaWorker(snapshot: GameSnapshot, playerNum: number): Promise<number> {
    await this.ensureWorker();
    const worker = this.worker;
    if (!worker) throw new Error('worker not available');
    const reqId = this.nextReqId++;
    return new Promise<number>((resolve, reject) => {
      this.pending.set(reqId, { resolve, reject });
      const req: WorkerRequest = { kind: 'search', reqId, snapshot, playerNum };
      worker.postMessage(req);
    });
  }
}
