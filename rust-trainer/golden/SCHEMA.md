# Golden-trace JSON schema (v6)

Files: `rust-trainer/golden/trace-<seed>.json`, one per game in the fixed suite.
Compact JSON, UTF-8, no trailing newline. Object **key order is fixed** by the
exporter (the order documented below) — a Rust serializer must match field order
only if it byte-compares; a value-comparing parity harness can ignore key order.

All floats are IEEE-754 doubles serialized by `JSON.stringify` (shortest
round-trippable form). The TS engine does all math in f64; the Rust port MUST use
f64 throughout (no f32) to reproduce feature vectors and network scores exactly.

## Top-level object

| field | type | notes |
|---|---|---|
| `schemaVersion` | int | currently 6 |
| `seed` | int | worldgen seed AND xorshift RNG seed for this game |
| `mapWidth` | int | tiles in x |
| `mapHeight` | int | tiles in y |
| `playerCount` | int | seats; all are CPU (difficulty "hard") |
| `roundCap` | int | max rounds before timeout |
| `config` | TierConfig | the TierConfig used (TRAINING_CONFIG) — see below |
| `genomeSource` | string | provenance of the genome (path or "deterministic-lcg:...") |
| `genomeArch` | int[] | MLP layer sizes, `[68,24,16,1]` |
| `genomeParamCount` | int | flat param count, 1977 for the default arch |
| `rngKind` | string | RNG description; at temperature=0/blunder=0 the RNG is never consumed in the decision loop |
| `hqPlacementTileIndex` | int[] | per seat (seat order = player num order, 0-based), the **tile index** (into `map`/fingerprint `tiles`) the controller claimed as HQ in round 0; -1 if none |
| `map` | MapTile[] | full tile grid (see below), in `ObjectManager.getTiles()` order |
| `rounds` | RoundRecord[] | chronological |
| `result` | Result | final outcome |

### TierConfig
`{ budget:int, temperature:float, reserve:int, blunder:float, experts:bool, military:bool, nuclear:bool, device:bool }`.
For these traces: `budget=40, temperature=0, reserve=120, blunder=0, experts=true, military=true, nuclear=true, device=true`.
With temperature=0 and blunder=0 selection is a deterministic argmax over network scores (ties → lowest index, since `>` strict). No RNG draws occur in the decision loop.

### MapTile
`{ x:int, y:int, type:string, building:string }`.
`type` is the tile class' `getType()`: one of `"Grassland"`, `"Forest"`,
`"Abundant Forest"`, `"Mountain"`, `"River"`. `building` is the building's
`getType()` ("" if none); at map-gen the only building present is `"Mikontalo"`.
**Ordering**: `getTiles()` order is generation order — the worldgen loop is
`for x in 0..sizeX: for y in 0..sizeY`, i.e. **column-major** (index = x*height + y).
All fingerprint `tiles` arrays use this SAME order and length.

## RoundRecord
`{ round:int, turns:TurnRecord[], afterEndTurn:Fingerprint }`.
`round` = `PlayerManager.getRoundsPlayed()` at the start of the round.
The exporter advances **one player turn per outer iteration** then calls
`endTurn()`; the engine's `getRoundsPlayed()` increments inside `endTurn` when
the turn order wraps, so multiple RoundRecords can share the same `round` value
(one per seat within a game-round). `turns` here always has length 1 (the single
current player); `afterEndTurn` is the global fingerprint after `endTurn()`
resolved production / connectivity-cut / win-loss for that turn.

## TurnRecord
`{ round:int, currentPlayerNum:int, before:Fingerprint, afterTurn:Fingerprint, decisions:DecisionRecord[] }`.
- `currentPlayerNum`: 1-based player number whose turn this is.
- `before`: fingerprint taken **before** the safety scaffold runs (i.e. exactly
  at turn start). For non-CPU seats (none in these traces) before==afterTurn.
- `afterTurn`: fingerprint after the safety scaffold (ensureWoodIncome +
  staffIncome) AND the full NN decision loop, but **before** `endTurn()`.
  Diff `before`→`afterTurn` to verify the scaffold + actions; diff
  `afterTurn`→ this round's `afterEndTurn` to verify endTurn resolution.
- `decisions`: chronological discretionary decisions (see below).

## DecisionRecord
One per iteration of the NN decision loop, captured **before** the chosen intent
executes, with exactly the vectors/candidates/scores the policy saw:
`{ round:int, globalVec:float[36], candidates:Candidate[], scores:float[], chosenCandidateIndex:int, chosenIntent:int }`.
- `globalVec`: the 36-dim global feature vector. **Ordering** (index → name):
  0=money, 1=wood, 2=stone, 3=metal, 4=netMoney, 5=metalIncome, 6=netWood, 7=moneyDrain, 8=tileFraction, 9=tileAbs, 10=maxUnit, 11=freeUnit, 12=workers, 13=experts, 14=maxSoldier, 15=freeSoldier, 16=soldiers, 17=staffedFarms, 18=mines, 19=villages, 20=outposts, 21=powerplants, 22=harvesters, 23=freeGrass, 24=freeMountain, 25=freeRiver, 26=round, 27=threat, 28=oppMaxFraction, 29=leadMargin, 30=oppSoldiers, 31=oppAlive, 32=dominationProgress, 33=neutralFraction, 34=reachableEnemy, 35=bias
  Each is clamped to roughly [-3,3] (a few to [0,1]/[0,2]/[0,3]); index 35 is the constant `1` bias. See `src/ai/nn/features.ts` for exact formulas.
- `candidates`: array of `{ intent:int, local:float[16], label:string }` in
  **enumeration order** (see Intent enum + enumerate order below). Expand(6) and
  Attack(8) are **multi-candidate**: each emits one entry per plausible target tile
  (own per-tile `local`), so those intent values may repeat. The last candidate is
  always Pass.
- `scores`: `scoreCandidate(genome, globalVec, candidate)` per candidate, SAME
  index order as `candidates`. The network input per candidate is
  `[globalVec(36) | intent one-hot(16) | local(16)]` = 68 dims, fed to the MLP
  (tanh hidden layers, linear scalar head). `scores[i]` is that scalar.
- `chosenCandidateIndex`: index into `candidates`/`scores` of the selected
  candidate (argmax of `scores` at temperature 0).
- `chosenIntent`: the chosen candidate's intent int (redundant with
  `candidates[chosenCandidateIndex].intent`).

NOTE on the rare retry path: if the chosen intent's `execute()` fails (a race the
controller guards against), the controller re-selects from a filtered candidate
list. Only the **initial** selection of each loop iteration is recorded as a
DecisionRecord. Such failures do not occur in deterministic headless replay; the
`afterTurn` fingerprint still captures the true post-turn state regardless.

### Intent enum (integer values 0..15)
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
| 12 | BuildBridge |
| 13 | CrackDevice |
| 14 | CrackHQ |
| 15 | MarchSoldier |

`enumerate()` evaluates intent builders in this order:
`[BuildFarm, BuildMine, BuildVillage, BuildOutpost, BuildHydro, BuildNuclear, BuildStrangeDevice, BuildBridge, Expand, HireSoldier, Attack, CrackDevice, CrackHQ, MarchSoldier, StackProducer]`,
appends each that returns a candidate (legal+affordable), then **always appends Pass**.
Note BuildStrangeDevice has intent VALUE 11 but sits at list POSITION 6 (after BuildNuclear). BuildBridge (12) sits at position 7. CrackDevice/CrackHQ (13/14) sit after Attack. MarchSoldier (15) is **multi-candidate** (one per movable own Soldier, capped 4) and sits after CrackHQ, before StackProducer. The one-hot encodes the value; the argmax tie-break uses list position.

Build* / StackProducer / HireSoldier are **single-candidate** (contribute 0 or 1).
Expand and Attack are **multi-candidate**: each emits one Candidate per plausible
target tile (each with its own per-tile `local` vector), spread into the list in
that builder's total-sorted order:
- **Expand** targets = neutral tiles (owner null, space for units, not threatened),
  sorted `claimValue` DESC then **tile index in `getTiles()` ASC** (the
  column-major generation order, index = x*height+y) as a strict tie-break, then
  capped to the top `EXPAND_CANDIDATE_CAP = 6`.
- **Attack** targets = enemy-owned conquerable tiles (not Outpost, <3 defenders),
  sorted HQ-first then fewest-defenders then **tile index ASC**; kept only if the
  assault is feasible (movable+buyable soldiers ≥ needed), capped to the top
  `ATTACK_CANDIDATE_CAP = 4` feasible targets.

So `candidates` is **non-decreasing** in intent value, with Expand(6) and Attack(8)
possibly repeated, and Pass(10) always last. The tile-index tie-break makes emission
order fully deterministic (argmax is strict `>`, lowest index wins on ties).

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
9. (metal - 50) / 500  (the 50 is a normalization offset, NOT the soldier metal
   cost — that was rebalanced 50 → 30 in arc sd3; this scale constant is left at 50
   to avoid a feature-distribution shift mid-arc)
Indices 0–9 are clamped to [-3,3] except the 0/1 flags.

**Spatial/positional per-target features (indices 10–15)** — added in schema v3.
Only Expand and Attack candidates carry non-zero values here; all other intents
(Build*/HireSoldier/StackProducer/Pass) emit all 6 as 0.0 (keeping them
value-equivalent and single-candidate). `tile` = the target tile, `p` = the
acting player. "enemy" = a tile whose owner is non-null AND ≠ `p` (neutral
excluded). 8-neighbours come from `TileBase.getNeighbourTiles()` (Chebyshev r=1,
clamped to map bounds, so edge/corner tiles have fewer than 8). Distances are
Manhattan |dx|+|dy| on `getCoordinate().x()/y()`; the sentinel `99` (missing HQ /
no enemy tiles) is substituted BEFORE dividing.
10. enemyNeighbors = (# 8-neighbours owned by an enemy) / 8, clamp[0,1]
11. ownNeighbors = (# 8-neighbours owned by `p`) / 8, clamp[0,1]
12. neutralNeighbors = (# 8-neighbours with no owner) / 8, clamp[0,1]
13. distOwnHq = Manhattan(tile, `om.getHqTile(p)`; none→99) / 20, clamp[0,3]
14. distNearestEnemyTile = min over all enemy-owned tiles of Manhattan(tile, enemyTile); none→99; / 20, clamp[0,3]
15. frontier (dual meaning): for **Attack** = (# Soldiers owned by `p` standing on
    the target's 8-neighbours) / 3, clamp[0,3]; for **Expand** = 1.0 if
    enemyNeighbors>0 else 0.0; for every other intent = 0.0.
Pass has all-zero local (16 zeros). See `localVec()` / `tileSpatial()` in
`src/ai/nn/candidates.ts` for exact constants per intent.

## Fingerprint
`{ players:PlayerFingerprint[], tiles:TileFingerprint[] }`.

### PlayerFingerprint
`{ num:int, alive:bool, money:int, wood:int, stone:int, metal:int }`.
One per player in the game's **original** player order (1..playerCount), even
after elimination. `alive` is false once the player is removed from
`PlayerManager.getPlayers()`. Resource ints are the raw treasury totals.

### TileFingerprint
`{ o:int, b:string, u:string, c:string }`, one per tile in `map` order.
- `o`: owner player num (1-based), `0` = neutral/unowned.
- `b`: building `getType()` ("" if none): `"Farm"`, `"Mine"`, `"Village"`,
  `"Outpost"`, `"Headquarters"`, `"Nuclear Power Plant"`,
  `"Hydroelectric Power Plant"`, `"Mikontalo"`, `"Bridge"`, ...
- `u`: **owned** units on the tile, encoded `"<seat>:<code>"` comma-joined in
  tile unit order, "" if none. `<seat>` = owner player num (0 if none).
  `<code>`: `W`=BasicWorker, `E`=Expert, `S`=Soldier, `?`=other.
- `c`: **conquering** units (assault stacks staged on the tile), same encoding,
  "" if none.

## Result
`{ winnerNum:int|null, reason:string, rounds:int, crashed:bool }`.
`reason` ∈ `"domination"` (winner ≥70% tiles), `"last-standing"`, `"tie"`,
`"timeout"`. `rounds` = final `getRoundsPlayed()`. `winnerNum` is null on
tie/timeout.

## Determinism guarantees
- Worldgen: MSVCRT `rand()`/`srand()` (RAND_MAX 32767) in `src/core/rng.ts`,
  consumed in the exact order in `src/world/worldgenerator.ts`. Seed → exact map.
- Policy: temperature=0, blunder=0 → pure argmax, RNG never drawn in the loop.
- Genome: fixed (see `genomeSource`). If `training/checkpoints/champion.json`
  exists it is used verbatim; otherwise a deterministic LCG genome (seed
  0xC0FFEE, inline LCG `s=s*1664525+1013904223 mod 2^32`, Box–Muller normals
  *0.5) is generated — identical every run.
- Re-running the exporter with the same args yields **byte-identical** files.
