//! Per-tile spatial plane extractor for the AlphaZero spatial representation —
//! the net's "EYES". Faithful to `rust-trainer/GAME-MECHANICS.md` (verified spec).
//!
//! ADDITIVE and AZ-ONLY: nothing here is wired into the parity-locked feature /
//! policy / search path. Produces a board-size-agnostic `(C, H, W)` tensor (same
//! flat layout as `cnn.rs`: `idx(c,y,x) = (c*H + y)*W + x`) describing the board
//! from one player's perspective, to feed the spatial CNN.
//!
//! Each plane is `H*W`, with tile `(x,y)` mapped to cell `(y,x)`. Values are
//! roughly in `[0,1]` (the att−def diff plane is signed in `[-1,1]`).
//!
//! ## The threat-model fix (why this file was rewritten)
//!
//! A prior version encoded threat as "a tile orthogonally adjacent to an enemy
//! soldier's CURRENT cell" (`C_THREAT`). **That is WRONG** (GAME-MECHANICS.md §1,
//! §4): there is no movement range / move-budget / has-moved flag. A unit moves in
//! ONE action to ANY tile in `getAvailableTiles()` (owned ∪ orthogonal-4 border,
//! minus own un-conquered HQ, minus unbridged-owned-river neighbours). So a soldier
//! anywhere in enemy territory threatens the WHOLE enemy frontier. Threat is a
//! **frontier-reachability × mobile-army-budget** property, NOT a soldier-position
//! property. The fix: an `C_ENEMY_REACH` plane = the union over live enemies of each
//! enemy's `getAvailableTiles()` set (where it can stage attackers next turn),
//! gated by a broadcast `C_ENEMY_BUDGET` plane (the enemy's mobile-soldier budget).

use crate::cnn::idx;
use cp_sim::model::{BuildingType, TileType};
use cp_sim::resources::BasicResource;
use cp_sim::{Game, PlayerId, UnitType};

/// Number of channels produced by [`board_planes`]. See the per-channel doc on
/// each `C_*` constant below for semantics and the GAME-MECHANICS.md § it is
/// faithful to.
pub const PLANE_COUNT: usize = 27;

// ── Channel indices (named for clarity; each cites its GAME-MECHANICS.md §) ──

/// 0 — owned-by-me. (§ownership)
const C_MINE: usize = 0;
/// 1 — owned-by-any-live-enemy.
const C_ENEMY: usize = 1;
/// 2 — neutral / unowned.
const C_NEUTRAL: usize = 2;
/// 3 — my un-conquered HQ tile (special; its loss = death). (§7)
const C_MY_HQ: usize = 3;
/// 4 — any live-enemy HQ.
const C_ENEMY_HQ: usize = 4;
/// 5 — producer building present (Farm/Mine/Village/Hydro/Nuclear), staffing-agnostic.
const C_PRODUCER: usize = 5;
/// 6 — Outpost present = **impregnable by assault** regardless of soldier counts. (§3)
const C_MILITARY: usize = 6;
/// 7 — Strange Device present on the tile. (§6)
const C_DEVICE: usize = 7;
/// 8 — MY **owned** soldiers (defenders, in `tile.units`), `(count/5).min(1)`. (§2,§5)
const C_MY_OWNED_SOLDIERS: usize = 8;
/// 9 — my HQ-connected mask (BFS over orthogonal-4 owned tiles from un-conquered HQ).
/// Disconnected tiles die at end of turn. (§7)
const C_HQ_CONNECTED: usize = 9;
/// 10 — terrain: grassland (farm/nuclear-able).
const C_T_GRASSLAND: usize = 10;
/// 11 — terrain: forest (Forest ∪ AbundantForest, wood tiles).
const C_T_FOREST: usize = 11;
/// 12 — terrain: mountain (mine-able).
const C_T_MOUNTAIN: usize = 12;
/// 13 — terrain: river (hydro/bridge-able; unbridged = expansion dead-end, see C_RIVER_BLOCK). (§8)
const C_T_RIVER: usize = 13;
/// 14 — my PRODUCING producer (mine AND actually generating resources this turn). (§5)
const C_PRODUCING: usize = 14;
/// 15 — ENEMY **owned** soldiers (live-enemy defenders), `(count/5).min(1)`. (§2)
const C_ENEMY_OWNED_SOLDIERS: usize = 15;
/// 16 — **ENEMY-REACHABILITY** = union over live enemies of each enemy's
/// `getAvailableTiles()` (owned ∪ orthogonal-4 border, minus unbridged-river
/// expansion, minus that enemy's own un-conquered HQ): every cell an enemy can
/// stage attackers on next turn. **REPLACES the wrong adjacency `C_THREAT`.**
/// (§1, §4 — the correct frontier-reachability threat model.)
const C_ENEMY_REACH: usize = 16;
/// 17 — **SELF-REACHABILITY** = my own `getAvailableTiles()` (where I can strike/expand). (§1)
const C_MY_REACH: usize = 17;
/// 18 — MY **conquering** (staged-attacker) soldiers (in `tile.conqueringUnits`),
/// `(count/5).min(1)`. Distinct list / role from owned defenders. (§2, §3)
const C_MY_CONQ_SOLDIERS: usize = 18;
/// 19 — ENEMY **conquering** (staged-attacker) soldiers, `(count/5).min(1)`. (§2, §3)
const C_ENEMY_CONQ_SOLDIERS: usize = 19;
/// 20 — **attacker − defender** soldier balance at each cell, signed `(diff/5)` in
/// `[-1,1]`, from MY perspective: + where I (as attacker on an enemy/neutral tile,
/// or defender on my tile) out-number, − where I am out-numbered. Combat is strict
/// `>` (tie → defender holds), so the sign + a +1 margin is what matters. (§3)
const C_ATT_MINUS_DEF: usize = 20;
/// 21 — **Device tile = DEFENSELESS** binary: a standing Device holds ZERO owned
/// defenders (`hasSpaceForUnits()==false`) → one staged attacker cracks it. (§2, §6)
const C_DEVICE_DEFENSELESS: usize = 21;
/// 22 — **unbridged-owned-river expansion block**: my owned River tile with no
/// building → a movement/expansion dead-end (its neighbours are NOT made
/// available; bridge/hydro re-enables). (§8)
const C_RIVER_BLOCK: usize = 22;
/// 23 — **ENEMY mobile-army budget** (BROADCAST, constant across the board): the
/// max over live enemies of that enemy's deployable soldier count — free/movable
/// owned soldiers + affordable new soldiers within its (Device-halved) soldier cap
/// — normalised `(budget/6).min(1)`. Gates C_ENEMY_REACH: reachable cells are only
/// a real threat if the enemy can actually FIELD `(my soldiers there)+1`. (§4, §5, §6)
const C_ENEMY_BUDGET: usize = 23;
/// 24 — **DISTANCE-TO-NEAREST-ENEMY-HQ** (per-cell potential field): for each cell,
/// `1 - clamp01(Manhattan distance from THIS cell to the nearest live-enemy
/// un-conquered HQ / diameter)`, where `diameter = w + h` (the board-size-agnostic
/// max Manhattan distance). 1.0 on top of an enemy HQ, decaying outward; all-zero
/// when there is no live-enemy HQ. Gives the trunk a direct "march toward the kill"
/// gradient the one-hot C_ENEMY_HQ plane cannot. (§7)
const C_DIST_TO_ENEMY_HQ: usize = 24;
/// 25 — **DISTANCE-TO-NEAREST-ENEMY-DEVICE** (per-cell potential field): same
/// `1 - clamp01(Manhattan / diameter)` form, measured to the nearest ENEMY-OWNED
/// standing Strange Device. All-zero when there is no standing Device, or it is
/// unowned / mine (only an enemy-owned Device is a race to crack). (§6)
const C_DIST_TO_ENEMY_DEVICE: usize = 25;
/// 26 — **MY mobile-army budget** (BROADCAST, constant across the board): my own
/// deployable-soldier budget, symmetric to C_ENEMY_BUDGET — `(budget/6).min(1)`
/// using the SAME `enemy_mobile_budget` definition applied to `player`. Lets the
/// trunk read how much striking power I can actually field this turn. (§4, §5, §6)
const C_MY_BUDGET: usize = 26;

/// Is this building a per-turn resource producer for the economy?
///
/// NOTE: forest harvesting in this engine is a tile/worker mechanic, not a
/// building, so there is no "forest-harvester" building to flag here; the
/// producer plane therefore covers Farm/Mine/Village/Hydro/Nuclear only.
fn is_producer(kind: BuildingType) -> bool {
    matches!(
        kind,
        BuildingType::Farm
            | BuildingType::Mine
            | BuildingType::Village
            | BuildingType::Hydro
            | BuildingType::Nuclear
    )
}

/// True if the producer building on this tile is ACTUALLY producing resources
/// this turn, using the *same* staffing gates as the production code in
/// `cp-sim` `managers.rs` (`generate_resources` / `gen_grassland` / `gen_mountain`
/// / `gen_river`):
///  - Farm (grassland): produces iff a `BasicWorker` is on the tile AND the farm
///    has MATURED — i.e. its stored `growth_phase == 4`, so that the engine's
///    `gen_grassland` (`growth_phase + 1 == 5 && has_worker`) pays out THIS turn.
///  - Mine (mountain): produces iff a `BasicWorker` is on the tile.
///  - Hydro (river) / Nuclear (grassland): produces iff an `Expert` is on the tile.
///  - Village / Outpost: produce unconditionally → always producing.
fn is_producing_producer(g: &Game, tile: &cp_sim::Tile) -> bool {
    let Some(b) = &tile.building else {
        return false;
    };
    let has = |kind: UnitType| tile.units.iter().any(|&u| g.units[u.0].kind == kind);
    match b.kind {
        BuildingType::Farm => b.growth_phase == 4 && has(UnitType::BasicWorker),
        BuildingType::Mine => has(UnitType::BasicWorker),
        BuildingType::Hydro | BuildingType::Nuclear => has(UnitType::Expert),
        BuildingType::Village | BuildingType::Outpost => true,
        _ => false,
    }
}

/// Count of `Soldier` units in a unit-id list (owned defenders OR conquering
/// attackers, depending on which list is passed).
fn count_soldiers(g: &Game, ids: &[cp_sim::UnitId]) -> i64 {
    ids.iter()
        .filter(|&&u| g.units[u.0].kind == UnitType::Soldier)
        .count() as i64
}

/// A live enemy's **mobile-soldier budget**: how many soldiers it could deploy
/// onto a staging tile next turn (GAME-MECHANICS.md §4/§5/§6). Faithful definition:
///   free/movable owned soldiers (every owned soldier can teleport in one action,
///   so the whole owned-soldier army is "mobile") + affordable NEW soldiers within
///   the enemy's REMAINING (Device-halved) soldier cap.
/// The Device-halving is already baked into the cached `max_soldier_amount`/
/// `free_soldier_amount` (engine recomputes it on build/end-of-turn, §6), so we
/// read the cached caps (no `&mut` needed).
fn enemy_mobile_budget(g: &Game, enemy: PlayerId) -> i64 {
    // Already-fielded soldiers (all are mobile — no move budget, §1).
    let owned_soldiers = g.current_soldier_amount(enemy);
    // New soldiers it can afford this turn (Money/200, Metal/50), capped by the
    // remaining (Device-aware) soldier slots.
    let res = &g.players[enemy.0].resources;
    let money = res.get(BasicResource::Money).unwrap_or(0);
    let metal = res.get(BasicResource::Metal).unwrap_or(0);
    let affordable = (money / 200).min(metal / 50).max(0);
    let free_slots = g.free_soldier_amount(enemy).max(0);
    owned_soldiers + affordable.min(free_slots)
}

/// Build the `(PLANE_COUNT, H, W)` spatial tensor for `player`. Returns the flat
/// tensor plus `(h, w)`. Board dimensions are derived from the tile grid
/// (`max x + 1`, `max y + 1`).
pub fn board_planes(g: &Game, player: PlayerId) -> (Vec<f64>, usize, usize) {
    let tiles = g.get_tiles();

    // Board dimensions from the tile coordinates.
    let mut max_x = 0i32;
    let mut max_y = 0i32;
    for t in tiles {
        if t.x > max_x {
            max_x = t.x;
        }
        if t.y > max_y {
            max_y = t.y;
        }
    }
    let w = (max_x + 1).max(1) as usize;
    let h = (max_y + 1).max(1) as usize;

    let mut out = vec![0.0f64; PLANE_COUNT * h * w];

    let live: Vec<PlayerId> = g.live_players().to_vec();
    let is_live_enemy = |o: PlayerId| o != player && live.contains(&o);

    // ── Gather coords for the per-cell DISTANCE fields (C_DIST_TO_ENEMY_HQ /
    // C_DIST_TO_ENEMY_DEVICE). `diameter = w + h` is the board-size-agnostic max
    // Manhattan distance (there is no board_diameter const). All-zero when absent,
    // matching the plane invariant convention.
    let diameter = (w + h) as f64;
    // Live-enemy un-conquered HQ coords (same source as C_ENEMY_HQ below).
    let mut enemy_hq_coords: Vec<(i32, i32)> = Vec::new();
    for &op in &live {
        if op == player {
            continue;
        }
        if let Some(hq) = g.get_hq_tile(op) {
            let t = &tiles[hq.0];
            enemy_hq_coords.push((t.x, t.y));
        }
    }
    // Enemy-OWNED standing Strange Device coord (at most one). Only counts when the
    // standing Device's tile is owned by a live enemy (not unowned, not mine).
    let mut enemy_device_coord: Option<(i32, i32)> = None;
    if let Some(dt) = g.find_strange_device_tile() {
        let t = &tiles[dt.0];
        if matches!(t.owner, Some(o) if is_live_enemy(o)) {
            enemy_device_coord = Some((t.x, t.y));
        }
    }

    // Per-tile planes (ownership / terrain / buildings / soldiers / combat).
    for t in tiles {
        // Defensive: skip any tile that would index outside the derived grid.
        if t.x < 0 || t.y < 0 || t.x as usize >= w || t.y as usize >= h {
            continue;
        }
        let (x, y) = (t.x as usize, t.y as usize);
        let cell = |c: usize| idx(c, y, x, h, w);

        // Terrain (one-hot; forest merges Forest + AbundantForest).
        match t.tile_type {
            TileType::Grassland => out[cell(C_T_GRASSLAND)] = 1.0,
            TileType::Forest | TileType::AbundantForest => out[cell(C_T_FOREST)] = 1.0,
            TileType::Mountain => out[cell(C_T_MOUNTAIN)] = 1.0,
            TileType::River => out[cell(C_T_RIVER)] = 1.0,
        }

        // Ownership.
        let owned_by_me = t.owner == Some(player);
        let owned_by_enemy = matches!(t.owner, Some(o) if is_live_enemy(o));
        match t.owner {
            Some(o) if o == player => out[cell(C_MINE)] = 1.0,
            Some(o) if is_live_enemy(o) => out[cell(C_ENEMY)] = 1.0,
            Some(_) => {} // dead/neutralised owner — neither mine nor a live-enemy
            None => out[cell(C_NEUTRAL)] = 1.0,
        }

        // Buildings.
        if let Some(b) = &t.building {
            match b.kind {
                BuildingType::Outpost => out[cell(C_MILITARY)] = 1.0,
                BuildingType::StrangeDevice => {
                    out[cell(C_DEVICE)] = 1.0;
                    // A standing Device holds zero owned defenders → defenseless (§2,§6).
                    out[cell(C_DEVICE_DEFENSELESS)] = 1.0;
                }
                k if is_producer(k) => out[cell(C_PRODUCER)] = 1.0,
                _ => {}
            }
            if owned_by_me && is_producing_producer(g, t) {
                out[cell(C_PRODUCING)] = 1.0;
            }
        }

        // Unbridged owned river = expansion dead-end (§8): owned by ANYONE it is a
        // movement block, but the actionable case for the net is MY own river.
        if owned_by_me && t.tile_type == TileType::River && t.building.is_none() {
            out[cell(C_RIVER_BLOCK)] = 1.0;
        }

        // ── Soldiers, split by OWNED (defenders) vs CONQUERING (staged attackers),
        // and by side (§2, §3). Owned soldiers live on the tile owner's side;
        // conquering soldiers are staged by whoever is attacking.
        let owned_sol = count_soldiers(g, &t.units);

        if owned_sol > 0 {
            if owned_by_me {
                out[cell(C_MY_OWNED_SOLDIERS)] = (owned_sol as f64 / 5.0).min(1.0);
            } else if owned_by_enemy {
                out[cell(C_ENEMY_OWNED_SOLDIERS)] = (owned_sol as f64 / 5.0).min(1.0);
            }
        }

        // Conquering units belong to a player attacking THIS tile. We classify by
        // unit owner (conquering units sit on tiles they do NOT own).
        let (mut my_conq, mut enemy_conq) = (0i64, 0i64);
        for &u in g.tile_conquering_units(t.id) {
            if g.units[u.0].kind != UnitType::Soldier {
                continue;
            }
            match g.units[u.0].owner {
                Some(o) if o == player => my_conq += 1,
                Some(o) if is_live_enemy(o) => enemy_conq += 1,
                _ => {}
            }
        }
        if my_conq > 0 {
            out[cell(C_MY_CONQ_SOLDIERS)] = (my_conq as f64 / 5.0).min(1.0);
        }
        if enemy_conq > 0 {
            out[cell(C_ENEMY_CONQ_SOLDIERS)] = (enemy_conq as f64 / 5.0).min(1.0);
        }

        // attacker − defender at this cell, from MY perspective, signed (§3, strict >).
        // On a tile I own: I am the defender (my owned soldiers) vs enemy conquerors.
        // On an enemy tile: I am the attacker (my conquerors) vs the enemy's owned
        // soldiers. Elsewhere (neutral): my conquerors vs none.
        let att_minus_def: i64 = if owned_by_me {
            owned_sol - enemy_conq // my defenders − enemy attackers (I want this > 0)
        } else if owned_by_enemy {
            my_conq - owned_sol // my attackers − enemy defenders (I need > 0 to take it)
        } else {
            my_conq // neutral claim: any attacker suffices
        };
        if att_minus_def != 0 {
            out[cell(C_ATT_MINUS_DEF)] = (att_minus_def as f64 / 5.0).clamp(-1.0, 1.0);
        }

        // ── Per-cell DISTANCE potential fields: 1 - clamp01(min Manhattan to a
        // target / diameter). 0.0 everywhere when there is no target (invariant).
        if !enemy_hq_coords.is_empty() {
            let dist = enemy_hq_coords
                .iter()
                .map(|&(ex, ey)| ((t.x - ex).abs() + (t.y - ey).abs()) as f64)
                .fold(f64::INFINITY, f64::min);
            out[cell(C_DIST_TO_ENEMY_HQ)] = (1.0 - (dist / diameter)).clamp(0.0, 1.0);
        }
        if let Some((ex, ey)) = enemy_device_coord {
            let dist = ((t.x - ex).abs() + (t.y - ey).abs()) as f64;
            out[cell(C_DIST_TO_ENEMY_DEVICE)] = (1.0 - (dist / diameter)).clamp(0.0, 1.0);
        }
    }

    // ── ENEMY-REACHABILITY (plane 16) and SELF-REACHABILITY (plane 17): each
    // player's getAvailableTiles() staging frontier (GAME-MECHANICS.md §1/§4),
    // computed EXACTLY as the engine does (owned ∪ orthogonal-4 border, unbridged-
    // river block, own-HQ exclusion). The union over live enemies = every cell an
    // enemy could stage an attacker on next turn.
    for tid in g.get_available_tiles_for(player) {
        let t = &tiles[tid.0];
        if t.x >= 0 && t.y >= 0 && (t.x as usize) < w && (t.y as usize) < h {
            out[idx(C_MY_REACH, t.y as usize, t.x as usize, h, w)] = 1.0;
        }
    }
    for &op in &live {
        if op == player {
            continue;
        }
        for tid in g.get_available_tiles_for(op) {
            let t = &tiles[tid.0];
            if t.x >= 0 && t.y >= 0 && (t.x as usize) < w && (t.y as usize) < h {
                out[idx(C_ENEMY_REACH, t.y as usize, t.x as usize, h, w)] = 1.0;
            }
        }
    }

    // ── ENEMY mobile-army budget (plane 23, BROADCAST constant): the strongest
    // live enemy's deployable-soldier budget, normalised. Gates the reachability
    // plane — reachable cells only threaten if the enemy can field (mine there)+1.
    let max_enemy_budget = live
        .iter()
        .filter(|&&o| o != player)
        .map(|&o| enemy_mobile_budget(g, o))
        .max()
        .unwrap_or(0);
    if max_enemy_budget > 0 {
        let v = (max_enemy_budget as f64 / 6.0).min(1.0);
        for y in 0..h {
            for x in 0..w {
                out[idx(C_ENEMY_BUDGET, y, x, h, w)] = v;
            }
        }
    }

    // ── MY mobile-army budget (plane 26, BROADCAST constant): symmetric to
    // C_ENEMY_BUDGET, using the SAME budget definition applied to `player`.
    let my_budget = enemy_mobile_budget(g, player);
    if my_budget > 0 {
        let v = (my_budget as f64 / 6.0).min(1.0);
        for y in 0..h {
            for x in 0..w {
                out[idx(C_MY_BUDGET, y, x, h, w)] = v;
            }
        }
    }

    // My HQ.
    if let Some(hq) = g.get_hq_tile(player) {
        let t = &tiles[hq.0];
        if t.x >= 0 && t.y >= 0 && (t.x as usize) < w && (t.y as usize) < h {
            out[idx(C_MY_HQ, t.y as usize, t.x as usize, h, w)] = 1.0;
        }
    }

    // Any live-enemy HQ.
    for &op in &live {
        if op == player {
            continue;
        }
        if let Some(hq) = g.get_hq_tile(op) {
            let t = &tiles[hq.0];
            if t.x >= 0 && t.y >= 0 && (t.x as usize) < w && (t.y as usize) < h {
                out[idx(C_ENEMY_HQ, t.y as usize, t.x as usize, h, w)] = 1.0;
            }
        }
    }

    // My-HQ-connected mask (engine's own connectivity rule, §7).
    for tid in g.get_hq_connected_tiles(player) {
        let t = &tiles[tid.0];
        if t.x >= 0 && t.y >= 0 && (t.x as usize) < w && (t.y as usize) < h {
            out[idx(C_HQ_CONNECTED, t.y as usize, t.x as usize, h, w)] = 1.0;
        }
    }

    (out, h, w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cp_sim::TileId;

    fn at(g: &Game, x: i32, y: i32) -> TileId {
        let i = g
            .get_tiles()
            .iter()
            .position(|t| t.x == x && t.y == y)
            .expect("tile exists");
        TileId(i)
    }

    #[inline]
    fn cell(out: &[f64], c: usize, x: usize, y: usize, h: usize, w: usize) -> f64 {
        out[idx(c, y, x, h, w)]
    }

    #[test]
    fn planes_basic_ownership_and_hq() {
        let mut g = Game::new(6, 5, &["P0", "P1"]);
        g.generate_map(6, 5, 1);

        let me = PlayerId(0);
        let enemy = PlayerId(1);

        let my_hq = at(&g, 0, 0);
        let my_other = at(&g, 1, 0);
        let enemy_hq = at(&g, 5, 3);

        g.set_tile_owner(my_hq, Some(me));
        g.set_tile_owner(my_other, Some(me));
        g.place_building(my_hq, BuildingType::Headquarters, Some(me));

        g.set_tile_owner(enemy_hq, Some(enemy));
        g.place_building(enemy_hq, BuildingType::Headquarters, Some(enemy));

        g.place_building(my_other, BuildingType::Farm, Some(me));

        let (out, h, w) = board_planes(&g, me);
        assert_eq!(out.len(), PLANE_COUNT * h * w);
        assert_eq!((w, h), (6, 5));

        assert_eq!(cell(&out, C_MINE, 0, 0, h, w), 1.0);
        assert_eq!(cell(&out, C_MINE, 1, 0, h, w), 1.0);
        assert_eq!(cell(&out, C_MINE, 5, 3, h, w), 0.0);

        assert_eq!(cell(&out, C_ENEMY, 5, 3, h, w), 1.0);
        assert_eq!(cell(&out, C_ENEMY, 0, 0, h, w), 0.0);

        assert_eq!(cell(&out, C_MY_HQ, 0, 0, h, w), 1.0);
        assert_eq!(cell(&out, C_MY_HQ, 1, 0, h, w), 0.0);
        assert_eq!(cell(&out, C_MY_HQ, 5, 3, h, w), 0.0);

        assert_eq!(cell(&out, C_ENEMY_HQ, 5, 3, h, w), 1.0);
        assert_eq!(cell(&out, C_ENEMY_HQ, 0, 0, h, w), 0.0);

        assert_eq!(cell(&out, C_PRODUCER, 1, 0, h, w), 1.0);

        assert_eq!(cell(&out, C_HQ_CONNECTED, 0, 0, h, w), 1.0);
        assert_eq!(cell(&out, C_HQ_CONNECTED, 1, 0, h, w), 1.0);
        assert_eq!(cell(&out, C_HQ_CONNECTED, 5, 3, h, w), 0.0);

        // Terrain planes partition the board.
        for ty in 0..h {
            for tx in 0..w {
                let n = cell(&out, C_T_GRASSLAND, tx, ty, h, w)
                    + cell(&out, C_T_FOREST, tx, ty, h, w)
                    + cell(&out, C_T_MOUNTAIN, tx, ty, h, w)
                    + cell(&out, C_T_RIVER, tx, ty, h, w);
                assert_eq!(n, 1.0, "exactly one terrain plane set at ({tx},{ty})");
            }
        }
    }

    #[test]
    fn planes_military_device_defenseless_and_neutral() {
        let mut g = Game::new(5, 3, &["P0", "P1"]);
        g.generate_map(5, 3, 2);
        let me = PlayerId(0);

        let outpost_tile = at(&g, 2, 1);
        let device_tile = at(&g, 3, 1);
        let neutral_tile = at(&g, 4, 2);

        g.set_tile_owner(outpost_tile, Some(me));
        g.place_building(outpost_tile, BuildingType::Outpost, Some(me));

        g.set_tile_owner(device_tile, Some(me));
        g.place_building(device_tile, BuildingType::StrangeDevice, Some(me));

        g.set_tile_owner(neutral_tile, None);

        let (out, h, w) = board_planes(&g, me);
        assert_eq!(out.len(), PLANE_COUNT * h * w);

        // Outpost = impregnable binary (§3).
        assert_eq!(cell(&out, C_MILITARY, 2, 1, h, w), 1.0);
        // Device present + defenseless (§2,§6).
        assert_eq!(cell(&out, C_DEVICE, 3, 1, h, w), 1.0);
        assert_eq!(cell(&out, C_DEVICE_DEFENSELESS, 3, 1, h, w), 1.0);
        assert_eq!(cell(&out, C_DEVICE_DEFENSELESS, 2, 1, h, w), 0.0, "outpost is not defenseless");
        assert_eq!(cell(&out, C_NEUTRAL, 4, 2, h, w), 1.0);
        assert_eq!(cell(&out, C_MINE, 4, 2, h, w), 0.0);
    }

    /// Threat is frontier-REACHABILITY, NOT soldier-cell adjacency: a soldier deep
    /// in territory makes the enemy's WHOLE staging frontier reachable, gated by
    /// the enemy's mobile budget. (GAME-MECHANICS.md §1, §4.)
    #[test]
    fn enemy_reachability_is_frontier_not_adjacency() {
        let mut g = Game::new(6, 5, &["P0", "P1"]);
        g.generate_map(6, 5, 7);
        // Clear all worldgen-assigned ownership so the available sets are controlled.
        let all: Vec<TileId> = g.get_tiles().iter().map(|t| t.id).collect();
        for tid in all {
            g.set_tile_owner(tid, None);
        }
        let me = PlayerId(0);
        let enemy = PlayerId(1);

        // Enemy owns a 2-tile block (2,2)-(3,2); its HQ at (3,2). A single enemy
        // soldier sits on (3,2) — deep, NOT adjacent to most of its own frontier.
        let my_hq = at(&g, 0, 0);
        g.set_tile_owner(my_hq, Some(me));
        g.place_building(my_hq, BuildingType::Headquarters, Some(me));

        let e_a = at(&g, 2, 2);
        let e_hq = at(&g, 3, 2);
        // Force Grassland so neither tile is an unbridged-river expansion dead-end
        // (the available-set logic depends on terrain; pin it for determinism).
        g.tiles[e_a.0].tile_type = TileType::Grassland;
        g.tiles[e_hq.0].tile_type = TileType::Grassland;
        g.tiles[my_hq.0].tile_type = TileType::Grassland;
        g.set_tile_owner(e_a, Some(enemy));
        g.set_tile_owner(e_hq, Some(enemy));
        g.place_building(e_hq, BuildingType::Headquarters, Some(enemy));
        // One soldier deep in enemy territory; give the enemy money so its budget>0.
        g.spawn_unit_on_tile(UnitType::Soldier, enemy, e_a, false);

        let (out, h, w) = board_planes(&g, me);

        // The enemy frontier = e_a/e_hq owned ∪ their orthogonal-4 border (minus the
        // enemy's own un-conquered HQ tile). Reachability must light up the border
        // tiles around the WHOLE block, not merely the 4-neighbours of the soldier.
        // e_a=(2,2) neighbours: (1,2),(2,1),(2,3) and e_hq=(3,2) is owned. The
        // enemy's own un-conquered HQ (3,2) is EXCLUDED from its available set.
        assert_eq!(cell(&out, C_ENEMY_REACH, 1, 2, h, w), 1.0, "border (1,2) reachable");
        assert_eq!(cell(&out, C_ENEMY_REACH, 2, 1, h, w), 1.0, "border (2,1) reachable");
        assert_eq!(cell(&out, C_ENEMY_REACH, 2, 3, h, w), 1.0, "border (2,3) reachable");
        // e_hq's far-side neighbour (4,2) is reachable too — far from the soldier's
        // cell, proving threat is frontier-wide, not soldier-adjacency.
        assert_eq!(cell(&out, C_ENEMY_REACH, 4, 2, h, w), 1.0, "far border (4,2) reachable");
        // The enemy's own un-conquered HQ tile is excluded from its available set.
        assert_eq!(cell(&out, C_ENEMY_REACH, 3, 2, h, w), 0.0, "enemy own HQ excluded");
        // My HQ at (0,0) is far from the enemy block → not on its frontier.
        assert_eq!(cell(&out, C_ENEMY_REACH, 0, 0, h, w), 0.0, "my far HQ not reachable");

        // Budget plane is broadcast & > 0 (enemy has 1 owned soldier).
        assert!(cell(&out, C_ENEMY_BUDGET, 0, 0, h, w) > 0.0, "enemy budget broadcast");
        assert_eq!(
            cell(&out, C_ENEMY_BUDGET, 0, 0, h, w),
            cell(&out, C_ENEMY_BUDGET, 5, 4, h, w),
            "budget plane is constant (broadcast)"
        );
    }

    /// Self-reachability mirrors enemy-reachability for MY available set. (§1)
    #[test]
    fn self_reachability_matches_engine() {
        let mut g = Game::new(6, 5, &["P0", "P1"]);
        g.generate_map(6, 5, 5);
        let me = PlayerId(0);
        // Own a single non-HQ tile so the available set is well-defined.
        let a = at(&g, 2, 2);
        g.set_tile_owner(a, Some(me));
        // Don't place an HQ here so has_opponent_headquarters stays true and the
        // tile + its neighbours are all available.
        let (out, h, w) = board_planes(&g, me);
        assert_eq!(cell(&out, C_MY_REACH, 2, 2, h, w), 1.0, "owned tile reachable");
        for &(x, y) in &[(1usize, 2usize), (3, 2), (2, 1), (2, 3)] {
            assert_eq!(cell(&out, C_MY_REACH, x, y, h, w), 1.0, "neighbour ({x},{y}) reachable");
        }
    }

    /// Owned defenders vs conquering (staged-attacker) soldiers go in SEPARATE
    /// planes per side, and the att−def diff is signed from my perspective. (§2,§3)
    #[test]
    fn owned_vs_conquering_soldiers_split() {
        let mut g = Game::new(6, 5, &["P0", "P1"]);
        g.generate_map(6, 5, 9);
        let me = PlayerId(0);
        let enemy = PlayerId(1);

        // I own (1,1) with a defending soldier; enemy stages a conquering soldier on it.
        let mine = at(&g, 1, 1);
        g.set_tile_owner(mine, Some(me));
        g.spawn_unit_on_tile(UnitType::Soldier, me, mine, false); // owned defender
        g.spawn_unit_on_tile(UnitType::Soldier, enemy, mine, true); // conquering attacker

        // Enemy owns (4,3) with a defender; I stage a conquering soldier on it.
        let his = at(&g, 4, 3);
        g.set_tile_owner(his, Some(enemy));
        g.spawn_unit_on_tile(UnitType::Soldier, enemy, his, false); // enemy owned defender
        g.spawn_unit_on_tile(UnitType::Soldier, me, his, true); // my conquering attacker

        let (out, h, w) = board_planes(&g, me);

        // My tile (1,1): my owned-soldier plane fires; enemy conquering plane fires.
        assert_eq!(cell(&out, C_MY_OWNED_SOLDIERS, 1, 1, h, w), 0.2);
        assert_eq!(cell(&out, C_ENEMY_CONQ_SOLDIERS, 1, 1, h, w), 0.2);
        assert_eq!(cell(&out, C_ENEMY_OWNED_SOLDIERS, 1, 1, h, w), 0.0);
        assert_eq!(cell(&out, C_MY_CONQ_SOLDIERS, 1, 1, h, w), 0.0);
        // att−def on my tile = my defenders(1) − enemy attackers(1) = 0 → tie favours
        // me (defender holds), encoded as 0 (no margin).
        assert_eq!(cell(&out, C_ATT_MINUS_DEF, 1, 1, h, w), 0.0);

        // Enemy tile (4,3): enemy owned-soldier plane fires; my conquering plane fires.
        assert_eq!(cell(&out, C_ENEMY_OWNED_SOLDIERS, 4, 3, h, w), 0.2);
        assert_eq!(cell(&out, C_MY_CONQ_SOLDIERS, 4, 3, h, w), 0.2);
        // att−def on enemy tile = my attackers(1) − enemy defenders(1) = 0 → I do NOT
        // take it (strict >), encoded 0.
        assert_eq!(cell(&out, C_ATT_MINUS_DEF, 4, 3, h, w), 0.0);
    }

    /// Unbridged owned river = expansion dead-end binary plane. (§8)
    #[test]
    fn unbridged_river_block_plane() {
        let mut g = Game::new(6, 5, &["P0", "P1"]);
        g.generate_map(6, 5, 11);
        let me = PlayerId(0);
        // Find a river tile, own it with no building → blocked.
        let river = g
            .get_tiles()
            .iter()
            .find(|t| t.tile_type == TileType::River)
            .map(|t| t.id);
        if let Some(rid) = river {
            g.set_tile_owner(rid, Some(me));
            let (out, h, w) = board_planes(&g, me);
            let t = &g.get_tiles()[rid.0];
            assert_eq!(
                cell(&out, C_RIVER_BLOCK, t.x as usize, t.y as usize, h, w),
                1.0,
                "unbridged owned river is an expansion block"
            );
        }
    }

    #[test]
    fn plane_producing_distinguishes_staffed_from_unstaffed() {
        let mut g = Game::new(6, 5, &["P0", "P1"]);
        g.generate_map(6, 5, 3);
        let me = PlayerId(0);

        let staffed = at(&g, 1, 1);
        let empty = at(&g, 3, 1);
        g.set_tile_owner(staffed, Some(me));
        g.set_tile_owner(empty, Some(me));
        g.place_building(staffed, BuildingType::Farm, Some(me));
        g.place_building(empty, BuildingType::Farm, Some(me));
        g.spawn_unit_on_tile(UnitType::BasicWorker, me, staffed, false);
        g.tiles[staffed.0].building.as_mut().unwrap().growth_phase = 4;

        let (out, h, w) = board_planes(&g, me);

        assert_eq!(cell(&out, C_PRODUCER, 1, 1, h, w), 1.0);
        assert_eq!(cell(&out, C_PRODUCER, 3, 1, h, w), 1.0);
        assert_eq!(cell(&out, C_PRODUCING, 1, 1, h, w), 1.0, "staffed mature farm produces");
        assert_eq!(cell(&out, C_PRODUCING, 3, 1, h, w), 0.0, "empty farm does not");

        g.tiles[staffed.0].building.as_mut().unwrap().growth_phase = 1;
        let (out_im, hi, wi) = board_planes(&g, me);
        assert_eq!(cell(&out_im, C_PRODUCING, 1, 1, hi, wi), 0.0, "immature farm does not produce");
        g.tiles[staffed.0].building.as_mut().unwrap().growth_phase = 4;

        let village = at(&g, 0, 3);
        g.set_tile_owner(village, Some(me));
        g.place_building(village, BuildingType::Village, Some(me));
        let (out2, h2, w2) = board_planes(&g, me);
        assert_eq!(cell(&out2, C_PRODUCING, 0, 3, h2, w2), 1.0, "village always produces");
    }

    /// Per-cell distance-to-enemy-HQ potential: 1 on the HQ, ~1-1/diameter on an
    /// adjacent cell, lower far away, all-zero when there is no live-enemy HQ. (§7)
    #[test]
    fn dist_to_enemy_hq_plane() {
        let mut g = Game::new(6, 5, &["P0", "P1"]);
        g.generate_map(6, 5, 1);
        let me = PlayerId(0);
        let enemy = PlayerId(1);

        // No enemy HQ yet → plane is all-zero.
        let my_hq = at(&g, 0, 0);
        g.set_tile_owner(my_hq, Some(me));
        g.place_building(my_hq, BuildingType::Headquarters, Some(me));
        let (out0, h0, w0) = board_planes(&g, me);
        for y in 0..h0 {
            for x in 0..w0 {
                assert_eq!(
                    cell(&out0, C_DIST_TO_ENEMY_HQ, x, y, h0, w0),
                    0.0,
                    "no enemy HQ → all-zero at ({x},{y})"
                );
            }
        }

        // One enemy HQ at (5,3). diameter = w + h = 6 + 5 = 11.
        let enemy_hq = at(&g, 5, 3);
        g.set_tile_owner(enemy_hq, Some(enemy));
        g.place_building(enemy_hq, BuildingType::Headquarters, Some(enemy));
        let (out, h, w) = board_planes(&g, me);
        let diameter = (w + h) as f64;

        // On the HQ cell: distance 0 → value 1.0.
        assert!(
            (cell(&out, C_DIST_TO_ENEMY_HQ, 5, 3, h, w) - 1.0).abs() < 1e-9,
            "on enemy HQ → 1.0"
        );
        // Adjacent cell (4,3): distance 1 → 1 - 1/diameter.
        let adj = cell(&out, C_DIST_TO_ENEMY_HQ, 4, 3, h, w);
        assert!(
            (adj - (1.0 - 1.0 / diameter)).abs() < 1e-9,
            "adjacent → 1 - 1/diameter, got {adj}"
        );
        // Far cell (0,0): distance 5+3=8 → 1 - 8/diameter, strictly lower than adjacent.
        let far = cell(&out, C_DIST_TO_ENEMY_HQ, 0, 0, h, w);
        assert!(
            (far - (1.0 - 8.0 / diameter)).abs() < 1e-9,
            "far → 1 - 8/diameter, got {far}"
        );
        assert!(far < adj, "far cell ({far}) < adjacent cell ({adj})");
    }

    /// Per-cell distance-to-enemy-device potential: nonzero when an ENEMY-OWNED
    /// standing Device exists; all-zero when the Device is mine or absent. (§6)
    #[test]
    fn dist_to_enemy_device_plane() {
        let mut g = Game::new(6, 5, &["P0", "P1"]);
        g.generate_map(6, 5, 4);
        let me = PlayerId(0);
        let enemy = PlayerId(1);

        // No device → all-zero.
        let (out0, h0, w0) = board_planes(&g, me);
        for y in 0..h0 {
            for x in 0..w0 {
                assert_eq!(cell(&out0, C_DIST_TO_ENEMY_DEVICE, x, y, h0, w0), 0.0);
            }
        }

        // MY device → still all-zero (only an enemy-owned device counts).
        let dev = at(&g, 2, 2);
        g.set_tile_owner(dev, Some(me));
        g.place_building(dev, BuildingType::StrangeDevice, Some(me));
        let (out_mine, hm, wm) = board_planes(&g, me);
        for y in 0..hm {
            for x in 0..wm {
                assert_eq!(
                    cell(&out_mine, C_DIST_TO_ENEMY_DEVICE, x, y, hm, wm),
                    0.0,
                    "my own device → all-zero at ({x},{y})"
                );
            }
        }

        // Hand the device tile to the enemy → plane lights up, 1.0 on the device.
        g.set_tile_owner(dev, Some(enemy));
        let (out, h, w) = board_planes(&g, me);
        assert!(
            (cell(&out, C_DIST_TO_ENEMY_DEVICE, 2, 2, h, w) - 1.0).abs() < 1e-9,
            "on enemy device → 1.0"
        );
        // A distant cell is strictly less than the on-device value.
        assert!(
            cell(&out, C_DIST_TO_ENEMY_DEVICE, 5, 4, h, w)
                < cell(&out, C_DIST_TO_ENEMY_DEVICE, 2, 2, h, w),
            "distant cell decays below on-device value"
        );
    }

    /// MY mobile-army budget broadcast: constant across the board, equal to the
    /// `enemy_mobile_budget` fn applied to the acting seat. (§4,§5,§6)
    #[test]
    fn my_budget_broadcast() {
        use cp_sim::resources::BasicResource;
        let mut g = Game::new(6, 5, &["P0", "P1"]);
        g.generate_map(6, 5, 6);
        let me = PlayerId(0);

        // Give the acting seat an HQ (so it has soldier-cap slots), some money/metal
        // and an owned soldier so its budget is > 0.
        let my_hq = at(&g, 0, 0);
        g.set_tile_owner(my_hq, Some(me));
        g.place_building(my_hq, BuildingType::Headquarters, Some(me));
        let a = at(&g, 1, 0);
        g.set_tile_owner(a, Some(me));
        g.spawn_unit_on_tile(UnitType::Soldier, me, a, false);
        g.players[me.0].resources.set(BasicResource::Money, 600);
        g.players[me.0].resources.set(BasicResource::Metal, 200);

        let (out, h, w) = board_planes(&g, me);

        let budget = enemy_mobile_budget(&g, me);
        assert!(budget > 0, "acting seat budget should be > 0, got {budget}");
        let expected = (budget as f64 / 6.0).min(1.0);

        // Constant across the whole board, equal to the fn's normalized value.
        for y in 0..h {
            for x in 0..w {
                assert!(
                    (cell(&out, C_MY_BUDGET, x, y, h, w) - expected).abs() < 1e-12,
                    "my-budget broadcast constant {expected} at ({x},{y})"
                );
            }
        }
    }
}
