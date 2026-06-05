// Test-time MCTS — PUCT search over per-candidate nodes, using the trained
// policy net as a PRIOR and the deterministic engine forward model for
// lookahead. 1:1 port of rust-trainer/crates/cp-ai/src/search.rs.
//
// This module is OPT-IN: it is only reached when a SearchConfig is attached to
// the controller (controller.planTurn). With search OFF, the controller path is
// byte-identical to today, so the parity gate is untouched.
//
// Node granularity = one candidate. A node is a mid-turn state for the ROOT
// player; its edges are the candidates from enumerate() at that node, in
// enumerate() order. We do NOT store an engine per node — node state is
// re-derived by rebuilding the root sandbox (from the captured root snapshot)
// and replaying the edge actions down the path (replay()). The sandbox's current
// player is the root player (sandbox sets it from snapshot.currentPlayerNum), and
// candidate execute() acts on the current player — so replay always acts as root.
//
// PUCT: argmax Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a)), Q = W/N.
// Priors: softmax (temperature tau_prior) over policy.scoreCandidate values.
// Leaf eval: static (metric), value-net, or short rollout.
// Opponents: only the root's candidate choices branch; every other seat is a
// forced deterministic transition (its planTurn search-OFF + endTurn). Value is
// always from the root player's perspective.

import { Genome } from './mlp';
import { globalFeatures } from './features';
import { enumerate, AiCtx, TierConfig, Intent, Candidate } from './candidates';
import { scoreCandidate, select as policySelect } from './policy';
import { ValueNet, valueForward } from './value';
import { NeuralAiController } from './controller';
import { createSandbox, Sandbox } from './sandbox';
import { GameSnapshot } from '../../managers/persistence';
import { PlayerBase } from '../../model/player';
import { ObjectManager } from '../../managers/objectmanager';
import { PlayerManager } from '../../managers/playermanager';
import * as M from './metrics';
import { TileBase } from '../../model/tile';
import { UnitBase } from '../../model/unit';
import { BasicResource,
  FARM_BUILD_COST, MINE_BUILD_COST, VILLAGE_BUILD_COST, OUTPOST_BUILD_COST,
  HEPP_BUILD_COST, NUCLEARPP_BUILD_COST, BRIDGE_BUILD_COST,
  BASIC_WORKER_COST, EXPERT_COST, SOLDIER_COST, ResourceMap as RMap,
} from '../../core/resources';

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

export type LeafEval =
  | { kind: 'static' }
  | { kind: 'value' }
  | { kind: 'rollout'; horizon: number };

export interface SearchConfig {
  /** Simulations per real decision. Most-visited root edge wins. */
  nSims: number;
  /** PUCT exploration constant. */
  cPuct: number;
  /** Softmax temperature applied to scoreCandidate values to form priors. */
  tauPrior: number;
  /** Leaf-evaluation mode. */
  leafEval: LeafEval;
  /** Hard cap on game-rounds inside a rollout. Ignored for static/value. */
  roundCap: number;
  /** Seed offset for the per-search RNG. */
  seed: number;
  /** Wall-clock cap (ms): break the sim loop early when exceeded. 0 = no cap. */
  timeBudgetMs: number;
  /**
   * Final-choice softening over the visit counts (mirrors policy.select). 0 =
   * argmax most-visited. >0 = temperature sampling over visit counts.
   */
  temperature: number;
  /** Probability of a deliberate blunder on the FINAL choice (weak tiers). */
  blunder: number;
}

export function defaultSearchConfig(): SearchConfig {
  return {
    nSims: 200,
    cPuct: 1.5,
    tauPrior: 1.0,
    leafEval: { kind: 'rollout', horizon: 10 },
    roundCap: 400,
    seed: 0x5ea2c4,
    timeBudgetMs: 0,
    temperature: 0,
    blunder: 0,
  };
}

// ---------------------------------------------------------------------------
// XorShift32 — matches training/harness.ts makeRng and cp-ai policy.rs.
// ---------------------------------------------------------------------------

export class XorShift32 {
  private s: number;
  constructor(seed: number) {
    // JS does the multiply in f64 then truncates to u32 (>>> 0).
    let s = (seed * 2654435761) >>> 0;
    if (s === 0) s = 0x9e3779b9;
    this.s = s;
  }
  next(): number {
    let s = this.s;
    s ^= s << 13; s >>>= 0;
    s ^= s >>> 17;
    s ^= s << 5; s >>>= 0;
    this.s = s >>> 0;
    return this.s / 4294967296;
  }
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

/** Numerically-stable softmax with temperature `tau`. */
export function softmax(scores: number[], tau: number): number[] {
  const n = scores.length;
  if (n === 0) return [];
  if (tau <= 0) {
    let best = 0;
    for (let i = 1; i < n; i++) if (scores[i] > scores[best]) best = i;
    const p = new Array<number>(n).fill(0);
    p[best] = 1;
    return p;
  }
  let max = -Infinity;
  for (const s of scores) if (s > max) max = s;
  let sum = 0;
  const p = scores.map((s) => {
    const e = Math.exp((s - max) / tau);
    sum += e;
    return e;
  });
  if (sum > 0) {
    for (let i = 0; i < n; i++) p[i] /= sum;
  } else {
    const u = 1 / n;
    for (let i = 0; i < n; i++) p[i] = u;
  }
  return p;
}

// ---------------------------------------------------------------------------
// Node / Tree
// ---------------------------------------------------------------------------

interface Node {
  /** Edge indices from the root down to (excluding) this node. */
  path: number[];
  /** Candidate labels/intents enumerated at this node, enumerate() order. */
  candidates: Candidate[];
  /** Per-edge priors P(s,a). */
  priors: number[];
  /** Per-edge child node index into the arena (-1 = not yet expanded). */
  children: number[];
  /** Per-edge visit count N(s,a). */
  edgeVisits: number[];
  /** Per-edge total value W(s,a) (root-player perspective). */
  edgeValue: number[];
  /** Total visits N(s). */
  visits: number;
  /** True once this node has been visited at least once (leaf evaluated). */
  expanded: boolean;
  /** True if this node is terminal for the root player's turn (all Pass). */
  terminal: boolean;
}

/** PUCT edge selection. Ties resolve to the LOWEST index (argmax convention). */
function puctSelect(node: Node, cPuct: number): number {
  const sqrtN = Math.sqrt(Math.max(node.visits, 0));
  let best = 0;
  let bestScore = -Infinity;
  for (let a = 0; a < node.candidates.length; a++) {
    const nSa = node.edgeVisits[a];
    const q = nSa > 0 ? node.edgeValue[a] / nSa : 0;
    const u = (cPuct * node.priors[a] * sqrtN) / (1 + nSa);
    const score = q + u;
    if (score > bestScore) {
      bestScore = score;
      best = a;
    }
  }
  return best;
}

// ---------------------------------------------------------------------------
// metricValue — 4-lead blend (mirror search.rs metric_value, reusing metrics.ts)
// ---------------------------------------------------------------------------

function livePlayers(pm: PlayerManager): PlayerBase[] {
  return pm.getPlayers();
}

function metricValue(om: ObjectManager, pm: PlayerManager, rootNum: number): number {
  const live = livePlayers(pm);
  const root = live.find((p) => p.getPlayerNum() === rootNum);
  if (!root) return -1; // root eliminated → loss
  const opp = live.filter((p) => p.getPlayerNum() !== rootNum);
  if (opp.length === 0) return 1; // sole survivor
  const k = opp.length;
  const total = Math.max(1, om.getTileCount());

  const myTiles = om.getTileCountForPlayer(root);
  const oppTiles = opp.reduce((s, p) => s + om.getTileCountForPlayer(p), 0) / k;
  const tileLead = clamp(((myTiles - oppTiles) / total) / 0.33, -1, 1);

  const myIncome = M.netMoneyPerRound(root);
  const oppIncome = opp.reduce((s, p) => s + M.netMoneyPerRound(p), 0) / k;
  const incomeLead = clamp((myIncome - oppIncome) / 200, -1, 1);

  const myWealth = totalWealth(root);
  const oppWealth = opp.reduce((s, p) => s + totalWealth(p), 0) / k;
  const wealthLead = clamp((myWealth - oppWealth) / 2000, -1, 1);

  const mySol = root.getCurrentSoldierAmount();
  const oppSol = opp.reduce((s, p) => s + p.getCurrentSoldierAmount(), 0) / k;
  const milLead = clamp((mySol - oppSol) / 5, -1, 1);

  const v = 0.5 * tileLead + 0.2 * incomeLead + 0.2 * wealthLead + 0.1 * milLead;
  return clamp(v, -1, 1);
}

const BUILDING_MONEY_COST: Record<string, RMap> = {
  Farm: FARM_BUILD_COST, Mine: MINE_BUILD_COST, Village: VILLAGE_BUILD_COST,
  Outpost: OUTPOST_BUILD_COST, 'Hydroelectric Power Plant': HEPP_BUILD_COST,
  'Nuclear Power Plant': NUCLEARPP_BUILD_COST, Bridge: BRIDGE_BUILD_COST,
  // Headquarters / Mikontalo are free / terrain-placed → no sunk cost.
};
const UNIT_MONEY_COST: Record<string, RMap> = {
  BasicWorker: BASIC_WORKER_COST, Expert: EXPERT_COST, Soldier: SOLDIER_COST,
};
const moneyOf = (c: RMap): number => -(c.get(BasicResource.MONEY) ?? 0);

/**
 * Sum of a player's four resources PLUS the sunk money cost of every owned
 * building and unit (mirror metrics::total_wealth in cp-ai/src/metrics.rs).
 */
function totalWealth(p: PlayerBase): number {
  let w = M.money(p) + M.wood(p) + M.stone(p) + M.metal(p);
  // Mirror cp-ai metrics::total_wealth: the player's object list holds Tiles AND
  // Units as SEPARATE entries, so count building cost off tiles and unit cost off
  // unit entries directly (do NOT also walk tile.getUnits() — that double-counts).
  for (const obj of p.getObjects()) {
    if (obj instanceof TileBase) {
      const bt = obj.getBuilding()?.getType();
      if (bt && BUILDING_MONEY_COST[bt]) w += moneyOf(BUILDING_MONEY_COST[bt]);
    } else if (obj instanceof UnitBase) {
      const uc = UNIT_MONEY_COST[obj.getType()];
      if (uc) w += moneyOf(uc);
    }
  }
  return w;
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

class Search {
  private nodes: Node[] = [];
  // The captured root snapshot — every replay rebuilds from this.
  constructor(
    private genome: Genome,
    private rootSnap: GameSnapshot,
    private rootPlayerNum: number,
    private cfg: TierConfig,
    private sc: SearchConfig,
    private valueNet: ValueNet | null,
    private rand: () => number,
  ) {}

  /**
   * The root player in this sandbox. Used only where the root is guaranteed live
   * (root state / mid-replay / its own turn boundary). If the root has been
   * eliminated (deep rollout), callers go through metricValue(rootNum) instead,
   * which resolves a dead root to -1 without dereferencing — so the `??` fallback
   * here is just a defensive guard that never feeds a real decision.
   */
  private rootPlayer(pm: PlayerManager): PlayerBase {
    const players = pm.getPlayers();
    return players.find((p) => p.getPlayerNum() === this.rootPlayerNum) ?? players[0];
  }

  private ctrl(sb: Sandbox): NeuralAiController {
    return new NeuralAiController(sb.eh, sb.om, sb.pm, this.genome, this.cfg, this.rand);
  }

  /** Build a node for the given path; `sb` is already at the replayed state. */
  private makeNode(sb: Sandbox, path: number[]): Node {
    const player = this.rootPlayer(sb.pm);
    const ctx: AiCtx = { eh: sb.eh, om: sb.om, pm: sb.pm, player, cfg: this.cfg };
    const round = sb.pm.getRoundsPlayed();
    const gvec = globalFeatures(player, sb.om, sb.pm, round);
    const candidates = enumerate(ctx);
    const scores = candidates.map((c) => scoreCandidate(this.genome, gvec, c));
    const priors = softmax(scores, this.sc.tauPrior);
    const n = candidates.length;
    const terminal = candidates.every((c) => c.intent === Intent.Pass);
    return {
      path,
      candidates,
      priors,
      children: new Array<number>(n).fill(-1),
      edgeVisits: new Array<number>(n).fill(0),
      edgeValue: new Array<number>(n).fill(0),
      visits: 0,
      expanded: false,
      terminal,
    };
  }

  /**
   * Reconstruct the sandbox at a node by rebuilding from the root snapshot and
   * replaying the node's edge actions (re-enumerating at each step so the chosen
   * edge's execute() closure is bound to the live sandbox, mirroring the Rust
   * `replay` which re-derives the action per edge). Returns the sandbox at that
   * node's state.
   */
  private replay(nodeIdx: number): Sandbox {
    const sb = createSandbox(this.rootSnap);
    const ctrl = this.ctrl(sb);
    const player = this.rootPlayer(sb.pm);
    let cur = 0; // root node index
    for (const edge of this.nodes[nodeIdx].path) {
      // Re-enumerate at the current sandbox state to obtain a freshly-bound
      // candidate for this edge (same enumerate() order as when the node was
      // built, since the sandbox state matches).
      const ctx: AiCtx = { eh: sb.eh, om: sb.om, pm: sb.pm, player, cfg: this.cfg };
      const cands = enumerate(ctx);
      const c = cands[edge];
      if (c) {
        try { c.execute(); } catch { /* tolerated: edge may no longer apply */ }
        ctrl.staffIncomePub(player);
      }
      cur = this.nodes[cur].children[edge];
    }
    return sb;
  }

  /**
   * Advance every NON-root seat by one forced deterministic turn, then settle on
   * the root player's turn boundary. Mirrors search.rs advance_round_after_root_turn.
   * Returns 'win'|'tie'|null (null = game continues, current==root).
   */
  private advanceRoundAfterRootTurn(sb: Sandbox): 'win' | 'tie' | null {
    const root = this.rootPlayer(sb.pm);
    // 1. End the root player's turn.
    sb.eh.endTurn();
    if (sb.menu.winner) return 'win';
    if (sb.menu.tie) return 'tie';
    // 2. Play every other live seat until it is the root player's turn again.
    while (sb.pm.getPlayers().length > 1) {
      const cur = sb.pm.getCurrentPlayer();
      if (cur.getPlayerNum() === root.getPlayerNum()) break;
      this.ctrl(sb).playTurn(cur); // search OFF (no recursion)
      sb.eh.endTurn();
      if (sb.menu.winner) return 'win';
      if (sb.menu.tie) return 'tie';
    }
    return null;
  }

  private terminalValue(sb: Sandbox, out: 'win' | 'tie'): number {
    if (out === 'tie') return 0;
    const w = sb.menu.winner;
    // Winner captured from setWinMenu (sole survivor) or domination.
    return w && w.getPlayerNum() === this.rootPlayerNum ? 1 : -1;
  }

  /** value-net leaf: exact terminal first, else net forward (fallback static). */
  private valueLeaf(sb: Sandbox): number {
    const live = sb.pm.getPlayers();
    const alive = live.some((p) => p.getPlayerNum() === this.rootPlayerNum);
    if (!alive) return -1;
    if (live.length <= 1) return 1; // sole survivor
    if (this.valueNet) {
      const player = this.rootPlayer(sb.pm);
      const round = sb.pm.getRoundsPlayed();
      const gvec = globalFeatures(player, sb.om, sb.pm, round);
      return valueForward(this.valueNet, gvec);
    }
    return metricValue(sb.om, sb.pm, this.rootPlayerNum);
  }

  /** Leaf evaluation. Dispatches on leafEval. `sb` is the leaf-state sandbox. */
  private evaluateLeaf(sb: Sandbox): number {
    const le = this.sc.leafEval;
    if (le.kind === 'static') return metricValue(sb.om, sb.pm, this.rootPlayerNum);
    if (le.kind === 'value') return this.valueLeaf(sb);

    // Rollout.
    const startRound = sb.pm.getRoundsPlayed();
    const horizon = Math.min(startRound + le.horizon, this.sc.roundCap);
    const ctrl = this.ctrl(sb);

    let out = this.advanceRoundAfterRootTurn(sb);
    if (out) return this.terminalValue(sb, out);

    while (sb.pm.getPlayers().length > 1 && sb.pm.getRoundsPlayed() < horizon) {
      const cur = sb.pm.getCurrentPlayer();
      if (cur.getPlayerNum() !== this.rootPlayerNum) break; // defensive
      ctrl.playTurn(cur); // root player, search OFF
      out = this.advanceRoundAfterRootTurn(sb);
      if (out) return this.terminalValue(sb, out);
    }
    return metricValue(sb.om, sb.pm, this.rootPlayerNum);
  }

  /** One MCTS simulation. Returns the leaf value (root perspective). */
  private simulate(): number {
    const visited: Array<[number, number]> = [];
    let node = 0;
    // The sandbox at the reached leaf state. When we EXPAND a new child we already
    // build its sandbox here, so we reuse it for the leaf eval instead of a second
    // full replay (a pure perf win — the state is identical). On the terminal /
    // existing-unexpanded path we rebuild it once below.
    let leafSb: Sandbox | null = null;

    for (;;) {
      if (this.nodes[node].terminal) break;
      if (!this.nodes[node].expanded) break;
      const edge = puctSelect(this.nodes[node], this.sc.cPuct);
      visited.push([node, edge]);
      const child = this.nodes[node].children[edge];
      if (child >= 0) {
        node = child;
      } else {
        // Expand this edge: build the child node from the replayed+stepped state.
        const path = this.nodes[node].path.slice();
        path.push(edge);
        const sb = this.replay(node);
        const player = this.rootPlayer(sb.pm);
        const ctx: AiCtx = { eh: sb.eh, om: sb.om, pm: sb.pm, player, cfg: this.cfg };
        const cands = enumerate(ctx);
        const c = cands[edge];
        if (c) {
          try { c.execute(); } catch { /* tolerated */ }
          this.ctrl(sb).staffIncomePub(player);
        }
        const childNode = this.makeNode(sb, path);
        this.nodes.push(childNode);
        const childIdx = this.nodes.length - 1;
        this.nodes[node].children[edge] = childIdx;
        node = childIdx;
        leafSb = sb; // reuse the freshly-built leaf sandbox for the eval below
        break;
      }
    }

    // Evaluate the reached leaf/terminal node. Reuse the sandbox built during
    // expansion if available; otherwise reconstruct it (terminal / re-evaluated
    // unexpanded node). Note rollout leaf-eval mutates this throwaway sandbox.
    if (!leafSb) leafSb = this.replay(node);
    const value = this.evaluateLeaf(leafSb);
    this.nodes[node].expanded = true;
    this.nodes[node].visits += 1;

    for (const [n, e] of visited) {
      this.nodes[n].edgeVisits[e] += 1;
      this.nodes[n].edgeValue[e] += value;
      this.nodes[n].visits += 1;
    }
    return value;
  }

  /** Run the search and return the chosen candidate index. */
  run(): number {
    const rootSb = createSandbox(this.rootSnap);
    const root = this.makeNode(rootSb, []);
    const nCands = root.candidates.length;
    this.nodes.push(root);
    if (nCands <= 1) return 0;

    // Mirror search.rs: seed ^ (rounds.wrapping_mul(2654435761)). Math.imul does
    // the u32-wrapping multiply; >>>0 keeps the XOR unsigned (the constructor also
    // truncates, but match the Rust intermediate exactly).
    const rng = new XorShift32(
      (this.sc.seed ^ Math.imul(this.rootSnap.roundsPlayed >>> 0, 2654435761)) >>> 0,
    );
    const startMs = this.sc.timeBudgetMs > 0 ? now() : 0;

    for (let i = 0; i < this.sc.nSims; i++) {
      this.simulate();
      if (this.sc.timeBudgetMs > 0 && (i & 3) === 3) {
        if (now() - startMs >= this.sc.timeBudgetMs) break;
      }
    }

    // Final choice over the ROOT edge visit counts. argmax (ties → lowest index)
    // unless tier temperature/blunder soften it (mirror policy.select).
    const rootNode = this.nodes[0];
    const visits = rootNode.edgeVisits;
    return finalChoice(visits, this.sc.temperature, this.sc.blunder, rng);
  }
}

function now(): number {
  return typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now();
}

/**
 * Pick the final root edge from visit counts, mirroring policy.select's
 * blunder→temperature→argmax cascade but over VISIT COUNTS instead of scores.
 */
function finalChoice(visits: number[], temperature: number, blunder: number, rng: XorShift32): number {
  const n = visits.length;
  if (n === 1) return 0;
  // Deliberate blunder: uniformly random edge.
  if (blunder > 0 && rng.next() < blunder) {
    return Math.floor(rng.next() * n);
  }
  if (temperature <= 1e-6) {
    let best = 0;
    let bestN = -1;
    for (let a = 0; a < n; a++) {
      if (visits[a] > bestN) { bestN = visits[a]; best = a; }
    }
    return best;
  }
  // Temperature softmax over visit counts.
  const t = temperature;
  let max = -Infinity;
  for (const v of visits) if (v > max) max = v;
  let sum = 0;
  const w = visits.map((v) => {
    const e = Math.exp((v - max) / t);
    sum += e;
    return e;
  });
  let r = rng.next() * sum;
  for (let i = 0; i < n; i++) {
    r -= w[i];
    if (r <= 0) return i;
  }
  return n - 1;
}

/**
 * Run test-time MCTS for the current mid-turn decision and return the chosen
 * candidate index (into enumerate() at the CURRENT live state — the same indexing
 * policy.select returns). The LIVE engine is NEVER mutated: all branching happens
 * on sandboxes rebuilt from `rootSnap`.
 */
export function select(
  genome: Genome,
  rootSnap: GameSnapshot,
  rootPlayerNum: number,
  cfg: TierConfig,
  sc: SearchConfig,
  valueNet: ValueNet | null,
  rand: () => number,
): number {
  const search = new Search(genome, rootSnap, rootPlayerNum, cfg, sc, valueNet, rand);
  return search.run();
}

export { policySelect };
