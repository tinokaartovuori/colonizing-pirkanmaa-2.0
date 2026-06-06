# Game Mechanics — canonical spec (verified, source of truth for the net's EYES)

_Authored 2026-06-05. Cross-checked across all three sources with file:line citations and
USER-CONFIRMED. C++ `reference/` is canonical; TS `src/` is a verified bit-for-bit port for all of
movement/conquest/capacity; Rust `cp-sim` is parity-locked to TS. The only deliberate divergences
are the economy numbers (Mine/Hydro/Nuclear, see CLAUDE.md) and the **Strange Device** (a new
building with no C++ counterpart). **This doc gates the cold-start net-input (planes/scalars)
design — any "eyes" feature must be faithful to these rules.**_

A prior design encoded threat as "a tile orthogonally adjacent to an enemy soldier's CURRENT cell."
**That is WRONG** and is the reason this doc exists. Threat is a *frontier-reachability × mobile-army*
property, not a soldier-position property.

---

## 1. Unit movement — NO range, NO move-budget, NO has-moved flag

A unit can move, in ONE action, to **any tile in `getAvailableTiles()`** — the unit's current cell is
irrelevant to legality. There is no movement range, no movement-point budget, and no per-turn
has-moved flag anywhere (grep across all three sources = empty).

`getAvailableTiles()` (current player) — `objectmanager.ts:131-153`, C++ `objectmanager.cpp:213-259`,
Rust `managers.rs:380-407` (identical):
- **all owned tiles** (passing `hasOpponentHeadquarters`), PLUS
- **their orthogonal-4 neighbours** (`getNeighbourFourTiles`) passing `hasOpponentHeadquarters`.
- Excludes the player's **own un-conquered HQ** tile (`hasOpponentHeadquarters` false only there).
- An owned **River with NO building does NOT contribute its neighbours** (the `continue`) — the river
  tile itself stays available, but you cannot expand *through* it. Bridge/Hydro on the river re-enables.

Adjacency for movement/availability is **orthogonal-4** (`coordinate.ts:86-103`). The 8-neighbourhood
is used only for build-adjacency (outpost spacing), NOT movement.

`canBePlacedOnTile` = "tile has room for the kind AND `availableTiles.includes(target)`"
(`unit.ts:76-85`). AI move: `ai_move_unit` `managers.rs:1356-1390`.

→ **CONFIRMED:** all units move freely within own territory; a soldier reaches any border tile in one move.

## 2. Owned units vs conquering units

A unit's `isConqueringUnit_` is set by the owner-vs-tile relationship at placement/move
(`unit.ts:34-59`, C++ `unitbase.cpp:32-73`):
- onto a tile **you own** → owned unit (stored in `tile.units_`).
- onto a tile **you don't own** (enemy/neutral) → **conquering** unit (stored in `tile.conqueringUnits_`).

Two separate per-tile lists, each capped at **MAX_UNITS = 3** (`tile.ts:276-287`). So a tile can hold
**≤3 owned defenders AND ≤3 conquering attackers simultaneously**.
- **Strange Device tile:** `hasSpaceForUnits() === false` always (`tile.ts:282`) → holds **zero owned
  defenders**, but attackers may still stage → a single attacker can crack an undefended device.

## 3. Conquest / assault resolution (`conquerTile`, end of turn)

Resolved at **end of the acting player's turn**, over all tiles (`gameeventhandler.ts:326`,
Rust `managers.rs:970-974`). Logic `tile.ts:128-179`, C++ `tilebase.cpp:174-249`, Rust `managers.rs:497-581`:
- **(a) Claim an UNOWNED tile** with your conquering units → become owner, units flip to owned. No combat.
- **(b) ASSAULT an enemy tile:** attacker takes it **iff `attackerConqueringSoldiers > defenderOwnedSoldiers`
  (strict; tie → defender holds) AND the tile has no Outpost.** **Outpost = impregnable by assault**
  regardless of count. Only **Soldiers** count (workers/experts = 0 combat) → a soldier-undefended tile
  falls to a single attacking soldier. On win: defenders deleted, attackers flip to owned, HQ flips to
  `conquered`. On loss: **all the attacker's conquering units on that tile are destroyed.**

## 4. Threat / reachability — the CORRECT model

Move (1) ⇒ an enemy can teleport any free soldier from anywhere in its territory onto any tile in **its**
`getAvailableTiles()`, stack ≤3, and resolve at its end-of-turn. So:

**Your tile is threatened next turn iff:** (1) it orthogonally borders (or is) ANY enemy-owned tile —
i.e. it sits on the enemy frontier; AND (2) it isn't an Outpost and its conquering slots aren't full; AND
(3) the enemy can muster **(your soldiers there)+1** soldiers — drawing free soldiers from **anywhere** in
its territory and/or buying within its (possibly Device-halved) soldier cap. **Where the enemy's soldiers
currently sit is irrelevant.** Enemy attack-target enumeration: `candidates.rs:1155-1174` (`attack`)
(`get_available_tiles().filter(enemy-owned, has conquering room, not outpost)`); free-soldier scan:
`find_free_soldier` `candidates.rs:197-212`. **Move and attack are separate** — staging this turn,
capture at the staging player's end-of-turn.

## 5. Capacity (recomputed from owned tiles each call)

`player.ts:153-171`, Rust `managers.rs:617-643`; per-tile `tiles.ts:182-199`:
- **Soldier cap** = HQ(un-conquered) **+1** + Outpost **+3**. (`HQ_SOLDIER_VALUE=1`, `OUTPOST_SOLDIER_VALUE=3`)
- **Unit (worker/expert) cap** = HQ **+3** + Village **+3** + Mikontalo **+2**.
- `freeSoldier = maxSoldier − soldiers`; `freeUnit = maxUnit − workers − experts`. Over-cap disbanded at
  end of turn / on Device build (`eliminateExcessUnits` `player.ts:181-204`).
- **HQ only ⇒ soldier cap = 1** (the capacity-blindness root). No Outpost ⇒ ≤1 soldier all game.
- **Strange Device HALVES the soldier cap** (floored), applied after summation (`player.ts:168`,
  Rust `managers.rs:651-652`).

## 6. The Strange Device — double-edged (USER-emphasised)

A new building (no C++ counterpart), the intended decisive win. On build:
- starts a **countdown** (`strange_device_countdown(tile_count)`, `resources.rs`); if it survives to 0 at
  the owner's end-of-turn → **all others lose** (`WinCause::Device`, `managers.rs:1061-1110`).
- **HALVES the builder's soldier cap** → the builder is *weaker on defense* exactly when it most needs to
  defend the device.
- the device tile holds **zero owned defenders** (§2) → a single enemy soldier staged on it destroys it
  (also: if the device tile leaves the builder's ownership, the device is destroyed, `managers.rs:1081-1086`).
- One device per game per player (`managers.rs:1247-1250`).

→ Device is **a threat to the enemy AND a self-inflicted defensive weakness.** The net must perceive: who
owns a standing device, its countdown, the cap-halving, and the device's defenselessness.

## 7. HQ-connectivity

`getHqConnectedTiles` = BFS from the un-conquered HQ over orthogonal-4 owned tiles (`objectmanager.ts:102-115`,
Rust `managers.rs:296+`); empty if no HQ. At end of turn, each opponent's tiles **not** HQ-connected are
**neutralised** (owner→null) — or **confiscated** by the current player if that opponent has no HQ
(`gameeventhandler.ts:334-356`). → **Cutting a chokepoint prunes everything behind it**; articulation points
can be worth far more than their local tile value.

## 8. River / bridge / blocked tiles

Units CAN stand on rivers (swim sprite only). An **unbridged owned river = expansion dead-end** (its
neighbours aren't made available). **Bridge or Hydro on a river re-enables expansion/movement across it.**
"Blocked tile" overlays are purely visual (every non-available tile).

## 9. Turn structure

Not one action. AI turn = budget-limited loop (`controller.rs:210-303`): staffing scaffolding, then
`while budget>0` { enumerate candidates → pick by policy/MCTS → execute → budget−1 } → one `end_turn()`.
Multiple build/hire/move/attack intents per turn.

---

## IMPLICATIONS FOR THE NET'S EYES (faithful encodings)

- **Threat:** NOT soldier-cell adjacency. Use an **enemy-reachability** plane = each enemy's
  `getAvailableTiles()` set (owned ∪ orthogonal-4 border, minus unbridged-river expansion, minus own HQ),
  **gated by that enemy's mobile-soldier budget** (free soldiers anywhere + affordable new soldiers within
  the Device-aware cap). Mirror a **self-reachability** plane (where I can strike).
- **Army strength:** separate planes for **owned soldiers** and **conquering (staged-attacker) soldiers**,
  per side; what matters at a contested cell is `attacker − defender` with **ties favouring the defender**
  (strict `>`). Keep workers/experts in separate planes (0 combat).
- **Impregnability / specials:** Outpost = impregnable binary plane; own-HQ special; **Device tile =
  defenseless** (binary), device-owner + countdown features per side.
- **Capacity (capacity-blindness fix):** per-side scalars for **free soldier slots** (HQ+1/Outpost+3,
  Device-halving baked in) and **free unit slots** (HQ+3/Village+3/Mikontalo+2). Without these the net
  cannot see "no Outpost ⇒ ~1 soldier ever."
- **Connectivity:** per-side HQ-connected mask; consider a cut-vulnerability / articulation-point feature.
- **River gates:** "can't expand through here" (unbridged river) + bridge/hydro-as-gate features.
