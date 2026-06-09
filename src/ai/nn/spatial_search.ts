// Spatial-net test-time MCTS (PUCT) — the in-browser twin of the DEPLOY/bench
// MCTS in rust-trainer/crates/cp-train/src/bin/cnn_train.rs (`mcts_select` /
// `simulate` / `make_node` / `leaf_value`, with the deploy config:
// turn_search=false, eval_prior_floor=0, forced_playouts=false).
//
// One decision = one PUCT search. A node is a mid-turn state for the ROOT player;
// its edges are the candidates from enumerate() at that node. Each EXPANDED edge
// applies ONE intent, then advances every NON-root seat by one forced HARD-bot
// turn and ends the round (advance_after_root). Leaf value = the spatial net's
// value head (`valueFrom`), already in the root player's frame; exact ±1 on a
// survivorship terminal. Final action = MOST-VISITED root edge (ties → lowest).
//
// State machinery mirrors search.ts: the LIVE engine is never mutated. Each node
// caches a state snapshot when built, so expanding a child restores the parent's
// snapshot and applies ONE edge — semantically identical to rebuilding from the
// root snapshot and replaying the whole edge path (both round-trip through the same
// buildSnapshot/restoreSnapshot), but it skips the O(depth) opponent-turn rolls
// that replay would redo on every expansion (a pure speedup; move choices are
// unchanged). Priors = softmax(tau=TAU) over the spatial net's per-candidate score;
// the leaf trunk is cached per node.

import { enumerate, AiCtx, TierConfig, Intent, Candidate } from './candidates';
import {
  SpatialNetTS, BoardCache, boardPlanes, valueScalars, candLocal, intentOnehot, targetXY,
} from './spatial_net';
import { createSandbox, Sandbox } from './sandbox';
import { GameSnapshot, buildSnapshot } from '../../managers/persistence';
import { PlayerBase } from '../../model/player';
import { ObjectManager } from '../../managers/objectmanager';
import { PlayerManager } from '../../managers/playermanager';
import { AiController } from '../../managers/ai';

// Deploy/bench constants (cnn_train.rs).
export const C_PUCT = 1.5;
export const TAU = 1.0;

/**
 * TEST-ONLY equivalence switch. When true, `sandboxAt` ignores the per-node state
 * cache and falls back to the original replay-from-root reconstruction. Used by the
 * move-equivalence test to prove the snapshot cache yields byte-identical choices.
 * Always false in deploy. Toggle via `setForceReplayFromRoot`.
 */
let FORCE_REPLAY_FROM_ROOT = false;
export function setForceReplayFromRoot(on: boolean): void { FORCE_REPLAY_FROM_ROOT = on; }

/** Spatial-MCTS deploy config (mirrors the Rust deploy `mcts_select` call). */
export interface SpatialSearchConfig {
  /** Simulations per real decision (deploy/bench = 64). */
  nSims: number;
  /** PUCT exploration constant (= C_PUCT). */
  cPuct: number;
  /** Softmax temperature for priors (= TAU). */
  tauPrior: number;
  /** Wall-clock cap (ms): break the sim loop early when exceeded. 0 = no cap. */
  timeBudgetMs: number;
}

export function defaultSpatialSearchConfig(): SpatialSearchConfig {
  return { nSims: 64, cPuct: C_PUCT, tauPrior: TAU, timeBudgetMs: 0 };
}

/** Numerically-stable softmax with temperature `tau` (matches softmax_tau). */
function softmaxTau(scores: number[], tau: number): number[] {
  const n = scores.length;
  if (n === 0) return [];
  let max = -Infinity;
  for (const s of scores) if (s > max) max = s;
  const t = Math.max(tau, 1e-9);
  let sum = 0;
  const p = scores.map((s) => { const e = Math.exp((s - max) / t); sum += e; return e; });
  if (sum > 0) for (let i = 0; i < n; i++) p[i] /= sum;
  else for (let i = 0; i < n; i++) p[i] = 1 / n;
  return p;
}

interface Node {
  /** Edge indices from the root down to (excluding) this node. */
  path: number[];
  candidates: Candidate[];
  priors: number[];
  children: number[];
  edgeVisits: number[];
  edgeValue: number[];
  visits: number;
  expanded: boolean;
  terminal: boolean;
  /** Cached spatial value of this node's state (root frame), or undefined for
   *  terminal/survivorship nodes that short-circuit. */
  cachedValue?: number;
  /**
   * Cached snapshot of this node's sandbox state, captured the moment the node was
   * built. Restoring a sandbox from THIS snapshot and applying one child edge is
   * semantically identical to rebuilding from the root snapshot and replaying the
   * whole edge path (both round-trip through the same snapshot/restore), but it
   * avoids re-running the O(depth) opponent-turn rolls on every expansion — the
   * dominant per-sim cost. Undefined only for terminal/survivorship leaves (never
   * expanded). PURE PERF: move choices are unchanged.
   */
  snapshot?: GameSnapshot;
}

/** PUCT edge selection. Ties → LOWEST index (cnn_train.rs puct_select). */
function puctSelect(node: Node, cPuct: number): number {
  const sqrtN = Math.sqrt(Math.max(node.visits, 0));
  let best = 0;
  let bestScore = -Infinity;
  for (let a = 0; a < node.candidates.length; a++) {
    const nSa = node.edgeVisits[a];
    const q = nSa > 0 ? node.edgeValue[a] / nSa : 0;
    const u = (cPuct * node.priors[a] * sqrtN) / (1 + nSa);
    const s = q + u;
    if (s > bestScore) { bestScore = s; best = a; }
  }
  return best;
}

function now(): number {
  return typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now();
}

class SpatialSearch {
  private nodes: Node[] = [];

  constructor(
    private net: SpatialNetTS,
    private rootSnap: GameSnapshot,
    private rootPlayerNum: number,
    private cfg: TierConfig,
    private sc: SpatialSearchConfig,
  ) {}

  private rootPlayer(pm: PlayerManager): PlayerBase {
    const players = pm.getPlayers();
    return players.find((p) => p.getPlayerNum() === this.rootPlayerNum) ?? players[0];
  }

  /** Build a node from an already-replayed sandbox state. */
  private makeNode(sb: Sandbox, path: number[]): Node {
    // Survivorship terminal: ≤1 live player. Never enumerate (would deref a
    // finished player_order). leaf_value short-circuits via cachedValue below.
    if (sb.pm.getPlayers().length <= 1) {
      return {
        path, candidates: [], priors: [], children: [], edgeVisits: [], edgeValue: [],
        visits: 0, expanded: false, terminal: true,
        cachedValue: this.survivorshipValue(sb),
      };
    }
    const player = this.rootPlayer(sb.pm);
    const ctx: AiCtx = { eh: sb.eh, om: sb.om, pm: sb.pm, player, cfg: this.cfg };
    const cands = enumerate(ctx);
    const { planes, h, w } = boardPlanes(player, sb.om, sb.pm);
    const cache: BoardCache = this.net.forwardBoard(planes, h, w, valueScalars(player, sb.om, sb.pm));
    const localDim = this.net.w.local_dim;
    const intentDim = this.net.w.intent_dim;
    const scores = cands.map((c) =>
      this.net.scoreCandidate(cache, targetXY(c), candLocal(c, player, localDim), intentOnehot(c, intentDim)));
    const priors = softmaxTau(scores, this.sc.tauPrior);
    const n = cands.length;
    const terminal = cands.every((c) => c.intent === Intent.Pass);
    return {
      path, candidates: cands, priors,
      children: new Array<number>(n).fill(-1),
      edgeVisits: new Array<number>(n).fill(0),
      edgeValue: new Array<number>(n).fill(0),
      visits: 0, expanded: false, terminal,
      cachedValue: this.net.valueFrom(cache),
      // Capture this node's exact state for O(1) child expansion (see Node.snapshot).
      snapshot: buildSnapshot(sb.om, sb.pm, this.rootSnap.settings),
    };
  }

  /** Root-frame survivorship value for a (near-)terminal sandbox. */
  private survivorshipValue(sb: Sandbox): number {
    const alive = sb.pm.getPlayers().some((p) => p.getPlayerNum() === this.rootPlayerNum);
    if (!alive) return -1;
    return 1; // sole survivor
  }

  private opponent(sb: Sandbox): AiController {
    return new AiController(sb.eh, sb.om, sb.pm);
  }

  /**
   * Reconstruct the sandbox AT a node. Restores directly from the node's cached
   * state snapshot (captured in `makeNode` the moment the node was built), which is
   * semantically identical to rebuilding from the root snapshot and replaying the
   * whole edge path — both round-trip through the same buildSnapshot/restoreSnapshot
   * — but skips the O(depth) opponent-turn rolls of that replay. If the snapshot is
   * somehow absent (defensive; terminal nodes are never expanded), falls back to the
   * exact replay-from-root path so semantics are preserved.
   */
  private sandboxAt(nodeIdx: number): Sandbox {
    const snap = this.nodes[nodeIdx].snapshot;
    if (snap && !FORCE_REPLAY_FROM_ROOT) return createSandbox(snap);
    const sb = createSandbox(this.rootSnap);
    for (const edge of this.nodes[nodeIdx].path) this.applyEdge(sb, edge);
    return sb;
  }

  /** Apply one root candidate edge to `sb`: execute the intent, then advance the
   *  round (forced HARD-bot opponents + endTurn). Tolerant of failures. */
  private applyEdge(sb: Sandbox, edge: number): void {
    if (sb.pm.getPlayers().length <= 1) return;
    const player = this.rootPlayer(sb.pm);
    const ctx: AiCtx = { eh: sb.eh, om: sb.om, pm: sb.pm, player, cfg: this.cfg };
    const cands = enumerate(ctx);
    const c = cands[edge];
    if (c && c.intent !== Intent.Pass) {
      try { c.execute(); } catch { /* tolerated */ }
    }
    this.advanceAfterRoot(sb);
  }

  /** End the root's turn, then play every non-root seat one forced HARD-bot turn
   *  until it is the root's turn again. Mirrors advance_after_root. */
  private advanceAfterRoot(sb: Sandbox): void {
    sb.eh.endTurn();
    if (sb.menu.winner || sb.menu.tie) return;
    while (sb.pm.getPlayers().length > 1) {
      const cur = sb.pm.getCurrentPlayer();
      if (cur.getPlayerNum() === this.rootPlayerNum) break;
      try { this.opponent(sb).playTurn(cur); } catch { /* keep going */ }
      sb.eh.endTurn();
      if (sb.menu.winner || sb.menu.tie) return;
    }
  }

  /** One simulation: descend via PUCT, expand+evaluate a leaf, back up. */
  private simulate(): void {
    const visited: Array<[number, number]> = [];
    let node = 0;
    for (;;) {
      if (this.nodes[node].terminal || !this.nodes[node].expanded) break;
      const edge = puctSelect(this.nodes[node], this.sc.cPuct);
      visited.push([node, edge]);
      const child = this.nodes[node].children[edge];
      if (child >= 0) { node = child; continue; }
      // Expand this edge.
      const path = this.nodes[node].path.slice();
      path.push(edge);
      const sb = this.sandboxAt(node); // restore the parent's cached state (O(1))
      this.applyEdge(sb, edge); // advance one full round via this edge
      const childNode = this.makeNode(sb, path);
      this.nodes.push(childNode);
      const idx = this.nodes.length - 1;
      this.nodes[node].children[edge] = idx;
      node = idx;
      break;
    }
    const value = this.nodes[node].cachedValue ?? 0;
    this.nodes[node].expanded = true;
    this.nodes[node].visits += 1;
    for (const [n, e] of visited) {
      this.nodes[n].edgeVisits[e] += 1;
      this.nodes[n].edgeValue[e] += value;
      this.nodes[n].visits += 1;
    }
  }

  /** Run the search; return the chosen candidate index (most-visited root edge). */
  run(): number {
    const rootSb = createSandbox(this.rootSnap);
    const root = this.makeNode(rootSb, []);
    const n = root.candidates.length;
    this.nodes.push(root);
    if (n <= 1) return 0;
    const startMs = this.sc.timeBudgetMs > 0 ? now() : 0;
    for (let i = 0; i < this.sc.nSims; i++) {
      this.simulate();
      if (this.sc.timeBudgetMs > 0 && (i & 3) === 3 && now() - startMs >= this.sc.timeBudgetMs) break;
    }
    const visits = this.nodes[0].edgeVisits;
    let chosen = 0, best = -1;
    for (let a = 0; a < visits.length; a++) if (visits[a] > best) { best = visits[a]; chosen = a; }
    return chosen;
  }
}

/**
 * Run spatial-net deploy MCTS for the current mid-turn decision and return the
 * chosen candidate INDEX into enumerate() at the CURRENT live state (same order
 * the caller re-enumerates). The LIVE engine is never mutated.
 */
export function selectSpatialMcts(
  net: SpatialNetTS,
  rootSnap: GameSnapshot,
  rootPlayerNum: number,
  cfg: TierConfig,
  sc: SpatialSearchConfig,
): number {
  return new SpatialSearch(net, rootSnap, rootPlayerNum, cfg, sc).run();
}
