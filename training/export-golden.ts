// MILESTONE 4 — Golden-trace exporter for the Rust parity harness.
//
// Produces fully-deterministic traces from the authoritative TS engine that a
// Rust port can replay against to prove bit-for-bit behavioural parity. Because
// the whole stack is deterministic (seeded MSVCRT worldgen, TRAINING_CONFIG with
// temperature=0 and blunder=0, a fixed genome, and the harness' seeded xorshift
// RNG), a matching Rust engine reproduces the ENTIRE trajectory; any divergence
// shows up as the first differing field in these traces.
//
// Run:    npx vite-node training/export-golden.ts            (default 8 games)
//         npx vite-node training/export-golden.ts 4          (first 4 games)
// Output: rust-trainer/golden/trace-<seed>.json  (+ SCHEMA.md, written once)
//
// Determinism check: run twice, diff the files — they are byte-identical.
//
// The ONLY engine change this relies on is an optional, additive `trace` sink
// argument on NeuralAiController.planTurn (default undefined → zero behavioural
// change; see controller.ts DecisionTrace). Everything else is read through
// existing public getters / existing exported pure functions.

import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { GameSettingsManager } from '../src/managers/gamesettings';
import { ObjectManager } from '../src/managers/objectmanager';
import { PlayerManager } from '../src/managers/playermanager';
import { GameEventHandler } from '../src/managers/gameeventhandler';
import { WorldGenerator } from '../src/world/worldgenerator';
import { MouseHoverBorder } from '../src/model/overlays';
import { Coordinate } from '../src/core/coordinate';
import { IGameScene, ISceneObjectHandle } from '../src/model/base';
import { PlayerBase, PlayerConfig } from '../src/model/player';
import { IMenuObjectManager } from '../src/managers/menu-interface';
import { TileBase } from '../src/model/tile';
import { BasicResource } from '../src/core/resources';
import { NeuralAiController, DecisionTrace } from '../src/ai/nn/controller';
import { Genome, paramCount } from '../src/ai/nn/mlp';
import { TierConfig } from '../src/ai/nn/candidates';
import { TRAINING_CONFIG } from '../src/ai/nn/tiers';
import { makeRng } from './harness';

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(__dirname, '..');
const OUT_DIR = resolve(REPO, 'rust-trainer/golden');
const CHECKPOINT = resolve(REPO, 'training/checkpoints/champion.json');

const SCHEMA_VERSION = 4; // Strange-Device arc: +BuildStrangeDevice intent, input 63→64
const POLICY_ARCH = [64, 24, 16, 1];

// --- headless scene + menu stubs (same pattern as training/harness.ts) -------

class StubScene implements IGameScene {
  drawItem(): void {}
  removeItem(): void {}
  updateItem(): void {}
  updateTile(): void {}
  isObjectInScene(): boolean { return true; }
  getObjectInScene(): ISceneObjectHandle { return { setAnimationOption() {}, setAnimationFrame() {} }; }
  addMouseFollowPicture(): void {}
  removeMouseFollowItem(): void {}
  deleteObjects(): void {}
}

class CapturingMenu implements IMenuObjectManager {
  winner: PlayerBase | null = null;
  tie = false;
  selectFirstTileMenuView(): void {}
  setTileInspectionMenuView(): void {}
  setStatMenuView(): void {}
  setDefaultMenuView(): void {}
  setUnitShopMenuView(): void {}
  setTieMenu(): void { this.tie = true; }
  setWinMenu(p: PlayerBase): void { this.winner = p; }
  setPlayerLostMenu(): void {}
  setCpuTurnMenuView(): void {}
}

// --- the fixed game suite ----------------------------------------------------
// (seed, width, height, players, roundCap) tuples covering small/medium maps and
// 2–3 players. Seeds are arbitrary fixed integers; they ARE the trace filename.

interface GameSpec { seed: number; width: number; height: number; players: number; roundCap: number; }

const SUITE: GameSpec[] = [
  { seed: 1,   width: 12, height: 12, players: 2, roundCap: 80 },
  { seed: 2,   width: 12, height: 12, players: 3, roundCap: 80 },
  { seed: 7,   width: 14, height: 12, players: 2, roundCap: 80 },
  { seed: 13,  width: 14, height: 12, players: 3, roundCap: 80 },
  { seed: 42,  width: 16, height: 14, players: 2, roundCap: 80 },
  { seed: 99,  width: 16, height: 14, players: 3, roundCap: 80 },
  { seed: 123, width: 18, height: 14, players: 2, roundCap: 80 },
  { seed: 256, width: 20, height: 15, players: 3, roundCap: 80 },
];

// --- genome source -----------------------------------------------------------
// Prefer the committed champion checkpoint (reproducibility = whatever ships).
// Fall back to a deterministic pseudo-random genome derived from a fixed seed so
// the exporter still runs on a clean checkout. Documented in the trace.

interface GenomeSource { genome: Genome; source: string; }

function deterministicGenome(seed: number): Genome {
  // Self-contained LCG → small Box–Muller normals. NOT randomGenome() (which
  // takes a rand()); we inline a fixed sequence so the genome is byte-stable and
  // independent of any other RNG state. Documented in SCHEMA.md.
  const n = paramCount(POLICY_ARCH);
  const params = new Array<number>(n);
  let s = (seed >>> 0) || 1;
  const next = () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; };
  for (let i = 0; i < n; i++) {
    const u1 = Math.max(next(), 1e-9);
    const u2 = next();
    params[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) * 0.5;
  }
  return { arch: POLICY_ARCH, params };
}

function loadGenome(): GenomeSource {
  if (existsSync(CHECKPOINT)) {
    const raw = JSON.parse(readFileSync(CHECKPOINT, 'utf8')) as Genome;
    // Only use the committed champion if its arch matches the CURRENT input dim;
    // after the intent change (63→64) an old 63-dim champion is incompatible, so
    // fall back to the deterministic genome rather than emit a mismatched trace.
    if (Array.isArray(raw.arch) && Array.isArray(raw.params) &&
        raw.arch.length === POLICY_ARCH.length && raw.arch.every((v, i) => v === POLICY_ARCH[i])) {
      return { genome: { arch: raw.arch, params: raw.params }, source: 'training/checkpoints/champion.json' };
    }
  }
  return { genome: deterministicGenome(0xC0FFEE), source: 'deterministic-lcg:seed=0xC0FFEE' };
}

// --- state fingerprinting ----------------------------------------------------
// Compact, complete-enough-to-detect-any-divergence snapshot of the whole game.

const RES = [BasicResource.MONEY, BasicResource.WOOD, BasicResource.STONE, BasicResource.METAL];

function unitCodes(units: { getType(): string; getOwner(): PlayerBase | null }[]): string {
  // Each unit -> "<seat>:<typecode>"; seat is player num (1-based) or 0 if none.
  // typecode: W=BasicWorker, E=Expert, S=Soldier, ?=other.
  const code = (t: string) => (t === 'BasicWorker' ? 'W' : t === 'Expert' ? 'E' : t === 'Soldier' ? 'S' : '?');
  return units.map((u) => `${u.getOwner()?.getPlayerNum() ?? 0}:${code(u.getType())}`).join(',');
}

interface TileFingerprint {
  /** owner player num (1-based), or 0 for neutral. */
  o: number;
  /** building type string, or "" if none. */
  b: string;
  /** owned units, "seat:code" joined by commas, "" if none. */
  u: string;
  /** conquering units (assault stacks), same encoding, "" if none. */
  c: string;
}

interface PlayerFingerprint {
  num: number;
  alive: boolean;
  money: number;
  wood: number;
  stone: number;
  metal: number;
}

interface Fingerprint {
  players: PlayerFingerprint[];
  /** tile fingerprints in om.getTiles() order. */
  tiles: TileFingerprint[];
}

function playerFingerprint(p: PlayerBase, alive: boolean): PlayerFingerprint {
  const r = p.getResources();
  return {
    num: p.getPlayerNum(),
    alive,
    money: r.get(BasicResource.MONEY) ?? 0,
    wood: r.get(BasicResource.WOOD) ?? 0,
    stone: r.get(BasicResource.STONE) ?? 0,
    metal: r.get(BasicResource.METAL) ?? 0,
  };
}

function fingerprint(om: ObjectManager, pm: PlayerManager, allPlayers: PlayerBase[]): Fingerprint {
  const alive = new Set(pm.getPlayers());
  const tiles = om.getTiles().map((t): TileFingerprint => ({
    o: t.getOwner()?.getPlayerNum() ?? 0,
    b: t.getBuilding()?.getType() ?? '',
    u: unitCodes(t.getUnits()),
    c: unitCodes(t.getConqueringUnits()),
  }));
  return {
    players: allPlayers.map((p) => playerFingerprint(p, alive.has(p))),
    tiles,
  };
}

// --- map snapshot ------------------------------------------------------------

interface MapTile { x: number; y: number; type: string; building: string; }

function mapSnapshot(om: ObjectManager): MapTile[] {
  return om.getTiles().map((t: TileBase) => {
    const c = t.getCoordinate();
    return { x: c.x(), y: c.y(), type: t.getType(), building: t.getBuilding()?.getType() ?? '' };
  });
}

// --- trace types -------------------------------------------------------------

interface TurnRecord {
  round: number;
  currentPlayerNum: number;
  /** fingerprint immediately BEFORE this player's turn (before the safety scaffold). */
  before: Fingerprint;
  /** fingerprint AFTER the safety scaffold + decision loop, before endTurn. */
  afterTurn: Fingerprint;
  /** discretionary decisions taken this turn, in chronological order. */
  decisions: DecisionTrace[];
}

interface RoundRecord {
  round: number;
  turns: TurnRecord[];
  /** fingerprint AFTER endTurn resolution for this round. */
  afterEndTurn: Fingerprint;
}

interface Trace {
  schemaVersion: number;
  seed: number;
  mapWidth: number;
  mapHeight: number;
  playerCount: number;
  roundCap: number;
  config: TierConfig;
  genomeSource: string;
  genomeArch: number[];
  genomeParamCount: number;
  /** xorshift32 (harness.makeRng) is seeded with `seed` per game; documented in SCHEMA. */
  rngKind: string;
  hqPlacementTileIndex: number[];
  map: MapTile[];
  rounds: RoundRecord[];
  result: { winnerNum: number | null; reason: string; rounds: number; crashed: boolean };
}

// --- one game ----------------------------------------------------------------

function runGame(spec: GameSpec, genome: Genome, genomeSource: string, cfg: TierConfig): Trace {
  const { seed, width, height, players: n, roundCap } = spec;

  const gsm = GameSettingsManager.fromMapDimensions(width, height);
  const om = new ObjectManager();
  const configs: PlayerConfig[] = Array.from({ length: n }, (_, i) => ({ name: `P${i + 1}`, difficulty: 'hard' as const }));
  const pm = new PlayerManager(configs, om);
  const menu = new CapturingMenu();
  const eh = new GameEventHandler(om, pm, menu, gsm);
  const scene = new StubScene();
  eh.setGameScene(scene);
  om.setGameScene(scene);
  om.addDALS(eh, menu, gsm);
  om.setHoverBorder(new MouseHoverBorder(new Coordinate(0, 0), 1, 1, eh, om));
  new WorldGenerator().generateMap(width, height, seed, { objectManager: om, eventHandler: eh, gameSettings: gsm, scene });

  const allPlayers = pm.getPlayers().slice();
  const rand = makeRng(seed);
  const ctrls = allPlayers.map(() => new NeuralAiController(eh, om, pm, genome, cfg, rand));
  const ctrlFor = (p: PlayerBase) => ctrls[p.getPlayerNum() - 1];

  // Index lookup: tile object -> position in om.getTiles() (HQ placement record).
  const tileIndex = new Map<TileBase, number>();
  om.getTiles().forEach((t, i) => tileIndex.set(t, i));

  // --- round 0: HQ placement for every seat ----------------------------------
  const hqPlacementTileIndex: number[] = [];
  eh.setAiActive(true);
  for (let i = 0; i < n; i++) {
    const cur = pm.getCurrentPlayer();
    ctrlFor(cur).placeHeadquarters(cur);
    const hq = om.getHqTile(cur);
    hqPlacementTileIndex.push(hq ? (tileIndex.get(hq) ?? -1) : -1);
  }
  eh.setAiActive(false);

  const rounds: RoundRecord[] = [];
  let crashed = false;

  try {
    while (pm.getPlayers().length > 1 && pm.getRoundsPlayed() < roundCap) {
      const round = pm.getRoundsPlayed();
      const turns: TurnRecord[] = [];

      const cur = pm.getCurrentPlayer();
      if (cur.isCpu()) {
        const before = fingerprint(om, pm, allPlayers);
        const decisions: DecisionTrace[] = [];
        eh.setAiActive(true);
        // Drive the REAL controller generator with the additive trace sink so the
        // captured decisions are exactly what the engine acted on.
        for (const _ of ctrlFor(cur).planTurn(cur, (d) => decisions.push(d))) { /* drain */ }
        eh.setAiActive(false);
        const afterTurn = fingerprint(om, pm, allPlayers);
        turns.push({ round, currentPlayerNum: cur.getPlayerNum(), before, afterTurn, decisions });
      } else {
        const fp = fingerprint(om, pm, allPlayers);
        turns.push({ round, currentPlayerNum: cur.getPlayerNum(), before: fp, afterTurn: fp, decisions: [] });
      }

      eh.endTurn();
      const afterEndTurn = fingerprint(om, pm, allPlayers);
      rounds.push({ round, turns, afterEndTurn });

      if (menu.winner || menu.tie) break;
    }
  } catch {
    crashed = true;
  }

  let winnerNum: number | null = null;
  let reason: string;
  if (menu.winner) {
    winnerNum = menu.winner.getPlayerNum();
    reason = (om.getTileCountForPlayer(menu.winner) * 100) / om.getTileCount() >= 70 ? 'domination' : 'last-standing';
  } else if (menu.tie) {
    reason = 'tie';
  } else if (pm.getPlayers().length === 1) {
    winnerNum = pm.getPlayers()[0].getPlayerNum();
    reason = 'last-standing';
  } else {
    reason = 'timeout';
  }

  return {
    schemaVersion: SCHEMA_VERSION,
    seed, mapWidth: width, mapHeight: height, playerCount: n, roundCap,
    config: cfg,
    genomeSource,
    genomeArch: genome.arch,
    genomeParamCount: genome.params.length,
    rngKind: 'xorshift32(makeRng); seeded with `seed`; drives blunder/softmax only (unused at temperature=0,blunder=0)',
    hqPlacementTileIndex,
    map: mapSnapshot(om),
    rounds,
    result: { winnerNum, reason, rounds: pm.getRoundsPlayed(), crashed },
  };
}

// --- main --------------------------------------------------------------------

function main(): void {
  const argv = process.argv.slice(2);
  const k = argv.length > 0 ? Math.max(1, Math.min(SUITE.length, parseInt(argv[0], 10) || SUITE.length)) : SUITE.length;
  const suite = SUITE.slice(0, k);

  const { genome, source } = loadGenome();
  mkdirSync(OUT_DIR, { recursive: true });

  let totalBytes = 0;
  for (const spec of suite) {
    const trace = runGame(spec, genome, source, TRAINING_CONFIG);
    const json = JSON.stringify(trace); // compact, deterministic key order
    const path = resolve(OUT_DIR, `trace-${spec.seed}.json`);
    writeFileSync(path, json);
    totalBytes += json.length;
    const decisionCount = trace.rounds.reduce((a, r) => a + r.turns.reduce((b, t) => b + t.decisions.length, 0), 0);
    console.log(
      `trace-${spec.seed}.json  ${spec.width}x${spec.height} p${spec.players}  ` +
      `rounds=${trace.result.rounds} decisions=${decisionCount} result=${trace.result.reason}` +
      (trace.result.winnerNum ? `(P${trace.result.winnerNum})` : '') +
      `  ${(json.length / 1024).toFixed(1)}KiB`,
    );
  }

  writeSchema();
  console.log(`\n${suite.length} traces, ${(totalBytes / 1024).toFixed(1)}KiB total. genome: ${source}`);
}

// --- SCHEMA.md (written once per run; deterministic) -------------------------

function writeSchema(): void {
  const md = `# Golden-trace JSON schema (v${SCHEMA_VERSION})

Files: \`rust-trainer/golden/trace-<seed>.json\`, one per game in the fixed suite.
Compact JSON, UTF-8, no trailing newline. Object **key order is fixed** by the
exporter (the order documented below) — a Rust serializer must match field order
only if it byte-compares; a value-comparing parity harness can ignore key order.

All floats are IEEE-754 doubles serialized by \`JSON.stringify\` (shortest
round-trippable form). The TS engine does all math in f64; the Rust port MUST use
f64 throughout (no f32) to reproduce feature vectors and network scores exactly.

## Top-level object

| field | type | notes |
|---|---|---|
| \`schemaVersion\` | int | currently ${SCHEMA_VERSION} |
| \`seed\` | int | worldgen seed AND xorshift RNG seed for this game |
| \`mapWidth\` | int | tiles in x |
| \`mapHeight\` | int | tiles in y |
| \`playerCount\` | int | seats; all are CPU (difficulty "hard") |
| \`roundCap\` | int | max rounds before timeout |
| \`config\` | TierConfig | the TierConfig used (TRAINING_CONFIG) — see below |
| \`genomeSource\` | string | provenance of the genome (path or "deterministic-lcg:...") |
| \`genomeArch\` | int[] | MLP layer sizes, \`[64,24,16,1]\` |
| \`genomeParamCount\` | int | flat param count, 1977 for the default arch |
| \`rngKind\` | string | RNG description; at temperature=0/blunder=0 the RNG is never consumed in the decision loop |
| \`hqPlacementTileIndex\` | int[] | per seat (seat order = player num order, 0-based), the **tile index** (into \`map\`/fingerprint \`tiles\`) the controller claimed as HQ in round 0; -1 if none |
| \`map\` | MapTile[] | full tile grid (see below), in \`ObjectManager.getTiles()\` order |
| \`rounds\` | RoundRecord[] | chronological |
| \`result\` | Result | final outcome |

### TierConfig
\`{ budget:int, temperature:float, reserve:int, blunder:float, experts:bool, military:bool, nuclear:bool, device:bool }\`.
For these traces: \`budget=40, temperature=0, reserve=120, blunder=0, experts=true, military=true, nuclear=true, device=true\`.
With temperature=0 and blunder=0 selection is a deterministic argmax over network scores (ties → lowest index, since \`>\` strict). No RNG draws occur in the decision loop.

### MapTile
\`{ x:int, y:int, type:string, building:string }\`.
\`type\` is the tile class' \`getType()\`: one of \`"Grassland"\`, \`"Forest"\`,
\`"Abundant Forest"\`, \`"Mountain"\`, \`"River"\`. \`building\` is the building's
\`getType()\` ("" if none); at map-gen the only building present is \`"Mikontalo"\`.
**Ordering**: \`getTiles()\` order is generation order — the worldgen loop is
\`for x in 0..sizeX: for y in 0..sizeY\`, i.e. **column-major** (index = x*height + y).
All fingerprint \`tiles\` arrays use this SAME order and length.

## RoundRecord
\`{ round:int, turns:TurnRecord[], afterEndTurn:Fingerprint }\`.
\`round\` = \`PlayerManager.getRoundsPlayed()\` at the start of the round.
The exporter advances **one player turn per outer iteration** then calls
\`endTurn()\`; the engine's \`getRoundsPlayed()\` increments inside \`endTurn\` when
the turn order wraps, so multiple RoundRecords can share the same \`round\` value
(one per seat within a game-round). \`turns\` here always has length 1 (the single
current player); \`afterEndTurn\` is the global fingerprint after \`endTurn()\`
resolved production / connectivity-cut / win-loss for that turn.

## TurnRecord
\`{ round:int, currentPlayerNum:int, before:Fingerprint, afterTurn:Fingerprint, decisions:DecisionRecord[] }\`.
- \`currentPlayerNum\`: 1-based player number whose turn this is.
- \`before\`: fingerprint taken **before** the safety scaffold runs (i.e. exactly
  at turn start). For non-CPU seats (none in these traces) before==afterTurn.
- \`afterTurn\`: fingerprint after the safety scaffold (ensureWoodIncome +
  staffIncome) AND the full NN decision loop, but **before** \`endTurn()\`.
  Diff \`before\`→\`afterTurn\` to verify the scaffold + actions; diff
  \`afterTurn\`→ this round's \`afterEndTurn\` to verify endTurn resolution.
- \`decisions\`: chronological discretionary decisions (see below).

## DecisionRecord
One per iteration of the NN decision loop, captured **before** the chosen intent
executes, with exactly the vectors/candidates/scores the policy saw:
\`{ round:int, globalVec:float[36], candidates:Candidate[], scores:float[], chosenCandidateIndex:int, chosenIntent:int }\`.
- \`globalVec\`: the 36-dim global feature vector. **Ordering** (index → name):
  ${nameTable()}
  Each is clamped to roughly [-3,3] (a few to [0,1]/[0,2]/[0,3]); index 35 is the constant \`1\` bias. See \`src/ai/nn/features.ts\` for exact formulas.
- \`candidates\`: array of \`{ intent:int, local:float[16], label:string }\` in
  **enumeration order** (see Intent enum + enumerate order below). Expand(6) and
  Attack(8) are **multi-candidate**: each emits one entry per plausible target tile
  (own per-tile \`local\`), so those intent values may repeat. The last candidate is
  always Pass.
- \`scores\`: \`scoreCandidate(genome, globalVec, candidate)\` per candidate, SAME
  index order as \`candidates\`. The network input per candidate is
  \`[globalVec(36) | intent one-hot(12) | local(16)]\` = 64 dims, fed to the MLP
  (tanh hidden layers, linear scalar head). \`scores[i]\` is that scalar.
- \`chosenCandidateIndex\`: index into \`candidates\`/\`scores\` of the selected
  candidate (argmax of \`scores\` at temperature 0).
- \`chosenIntent\`: the chosen candidate's intent int (redundant with
  \`candidates[chosenCandidateIndex].intent\`).

NOTE on the rare retry path: if the chosen intent's \`execute()\` fails (a race the
controller guards against), the controller re-selects from a filtered candidate
list. Only the **initial** selection of each loop iteration is recorded as a
DecisionRecord. Such failures do not occur in deterministic headless replay; the
\`afterTurn\` fingerprint still captures the true post-turn state regardless.

### Intent enum (integer values 0..11)
| value | intent |
|---|---|
| 0 | BuildFarm |
| 1 | BuildMine |
| 2 | BuildVillage |
| 3 | BuildOutpost |
| 4 | BuildHydro |
| 5 | BuildNuclear |
| 6 | Expand |
| 7 | HireSoldier |
| 8 | Attack |
| 9 | StackProducer |
| 10 | Pass |
| 11 | BuildStrangeDevice |

\`enumerate()\` evaluates intent builders in this order:
\`[BuildFarm, BuildMine, BuildVillage, BuildOutpost, BuildHydro, BuildNuclear, BuildStrangeDevice, Expand, HireSoldier, Attack, StackProducer]\`,
appends each that returns a candidate (legal+affordable), then **always appends Pass**.
Note BuildStrangeDevice has intent VALUE 11 but sits at list POSITION 6 (after BuildNuclear): the one-hot encodes the value; the argmax tie-break uses list position.

Build* / StackProducer / HireSoldier are **single-candidate** (contribute 0 or 1).
Expand and Attack are **multi-candidate**: each emits one Candidate per plausible
target tile (each with its own per-tile \`local\` vector), spread into the list in
that builder's total-sorted order:
- **Expand** targets = neutral tiles (owner null, space for units, not threatened),
  sorted \`claimValue\` DESC then **tile index in \`getTiles()\` ASC** (the
  column-major generation order, index = x*height+y) as a strict tie-break, then
  capped to the top \`EXPAND_CANDIDATE_CAP = 6\`.
- **Attack** targets = enemy-owned conquerable tiles (not Outpost, <3 defenders),
  sorted HQ-first then fewest-defenders then **tile index ASC**; kept only if the
  assault is feasible (movable+buyable soldiers ≥ needed), capped to the top
  \`ATTACK_CANDIDATE_CAP = 4\` feasible targets.

So \`candidates\` is **non-decreasing** in intent value, with Expand(6) and Attack(8)
possibly repeated, and Pass(10) always last. The tile-index tie-break makes emission
order fully deterministic (argmax is strict \`>\`, lowest index wins on ties).

### Candidate local feature vector (16 dims), index → meaning
0. money cost / 1000 (negative-of-cost magnitude, clamped)
1. netDelta / 100 (heuristic income delta of the action)
2. targetValue / 6 (tile/value heuristic)
3. unitCapGain / 3
4. soldierCapGain / 3
5. threatened (0/1)
6. (money - 120 - moneyDrain*5) / 1000 (spend headroom)
7. incomeStaffing (0/1)
8. (wood - woodNeed - buffer) / 500 (wood headroom)
9. (metal - 50) / 500
Indices 0–9 are clamped to [-3,3] except the 0/1 flags.

**Spatial/positional per-target features (indices 10–15)** — added in schema v3.
Only Expand and Attack candidates carry non-zero values here; all other intents
(Build*/HireSoldier/StackProducer/Pass) emit all 6 as 0.0 (keeping them
value-equivalent and single-candidate). \`tile\` = the target tile, \`p\` = the
acting player. "enemy" = a tile whose owner is non-null AND ≠ \`p\` (neutral
excluded). 8-neighbours come from \`TileBase.getNeighbourTiles()\` (Chebyshev r=1,
clamped to map bounds, so edge/corner tiles have fewer than 8). Distances are
Manhattan |dx|+|dy| on \`getCoordinate().x()/y()\`; the sentinel \`99\` (missing HQ /
no enemy tiles) is substituted BEFORE dividing.
10. enemyNeighbors = (# 8-neighbours owned by an enemy) / 8, clamp[0,1]
11. ownNeighbors = (# 8-neighbours owned by \`p\`) / 8, clamp[0,1]
12. neutralNeighbors = (# 8-neighbours with no owner) / 8, clamp[0,1]
13. distOwnHq = Manhattan(tile, \`om.getHqTile(p)\`; none→99) / 20, clamp[0,3]
14. distNearestEnemyTile = min over all enemy-owned tiles of Manhattan(tile, enemyTile); none→99; / 20, clamp[0,3]
15. frontier (dual meaning): for **Attack** = (# Soldiers owned by \`p\` standing on
    the target's 8-neighbours) / 3, clamp[0,3]; for **Expand** = 1.0 if
    enemyNeighbors>0 else 0.0; for every other intent = 0.0.
Pass has all-zero local (16 zeros). See \`localVec()\` / \`tileSpatial()\` in
\`src/ai/nn/candidates.ts\` for exact constants per intent.

## Fingerprint
\`{ players:PlayerFingerprint[], tiles:TileFingerprint[] }\`.

### PlayerFingerprint
\`{ num:int, alive:bool, money:int, wood:int, stone:int, metal:int }\`.
One per player in the game's **original** player order (1..playerCount), even
after elimination. \`alive\` is false once the player is removed from
\`PlayerManager.getPlayers()\`. Resource ints are the raw treasury totals.

### TileFingerprint
\`{ o:int, b:string, u:string, c:string }\`, one per tile in \`map\` order.
- \`o\`: owner player num (1-based), \`0\` = neutral/unowned.
- \`b\`: building \`getType()\` ("" if none): \`"Farm"\`, \`"Mine"\`, \`"Village"\`,
  \`"Outpost"\`, \`"Headquarters"\`, \`"Nuclear Power Plant"\`,
  \`"Hydroelectric Power Plant"\`, \`"Mikontalo"\`, \`"Bridge"\`, ...
- \`u\`: **owned** units on the tile, encoded \`"<seat>:<code>"\` comma-joined in
  tile unit order, "" if none. \`<seat>\` = owner player num (0 if none).
  \`<code>\`: \`W\`=BasicWorker, \`E\`=Expert, \`S\`=Soldier, \`?\`=other.
- \`c\`: **conquering** units (assault stacks staged on the tile), same encoding,
  "" if none.

## Result
\`{ winnerNum:int|null, reason:string, rounds:int, crashed:bool }\`.
\`reason\` ∈ \`"domination"\` (winner ≥70% tiles), \`"last-standing"\`, \`"tie"\`,
\`"timeout"\`. \`rounds\` = final \`getRoundsPlayed()\`. \`winnerNum\` is null on
tie/timeout.

## Determinism guarantees
- Worldgen: MSVCRT \`rand()\`/\`srand()\` (RAND_MAX 32767) in \`src/core/rng.ts\`,
  consumed in the exact order in \`src/world/worldgenerator.ts\`. Seed → exact map.
- Policy: temperature=0, blunder=0 → pure argmax, RNG never drawn in the loop.
- Genome: fixed (see \`genomeSource\`). If \`training/checkpoints/champion.json\`
  exists it is used verbatim; otherwise a deterministic LCG genome (seed
  0xC0FFEE, inline LCG \`s=s*1664525+1013904223 mod 2^32\`, Box–Muller normals
  *0.5) is generated — identical every run.
- Re-running the exporter with the same args yields **byte-identical** files.
`;
  writeFileSync(resolve(OUT_DIR, 'SCHEMA.md'), md);
}

function nameTable(): string {
  const names = [
    'money', 'wood', 'stone', 'metal',
    'netMoney', 'metalIncome', 'netWood', 'moneyDrain',
    'tileFraction', 'tileAbs',
    'maxUnit', 'freeUnit', 'workers', 'experts',
    'maxSoldier', 'freeSoldier', 'soldiers',
    'staffedFarms', 'mines', 'villages', 'outposts', 'powerplants', 'harvesters',
    'freeGrass', 'freeMountain', 'freeRiver',
    'round', 'threat',
    'oppMaxFraction', 'leadMargin', 'oppSoldiers', 'oppAlive',
    'dominationProgress', 'neutralFraction', 'reachableEnemy',
    'bias',
  ];
  return names.map((n, i) => `${i}=${n}`).join(', ');
}

main();
