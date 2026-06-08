//! Port of `src/ai/nn/candidates.ts` — the macro-action *intents* the policy
//! chooses among, the 11 builders, `enumerate()`, and each candidate's
//! `execute()`.
//!
//! In the TS each `Candidate` carries an `execute` closure that mutates the
//! engine. Rust's borrow checker won't let us hold such closures (each borrows
//! `&mut Game`) while enumerating over `&Game`. We therefore split the candidate
//! into a *plan* (built under an immutable borrow) and an [`Action`] describing
//! exactly what `execute()` should do; [`execute_action`] performs it under a
//! mutable borrow. The control flow / fallibility matches the TS closures.

use crate::metrics as m;
use crate::safety as s;
use crate::tiers::TierConfig;
use cp_sim::resources::{
    basic_worker_cost, bridge_build_cost, expert_cost, farm_build_cost, hepp_build_cost,
    mine_build_cost, nuclearpp_build_cost, outpost_build_cost, soldier_cost,
    strange_device_build_cost, village_build_cost, BasicResource, ResourceMap,
};
use cp_sim::{BuildingType, Game, PlayerId, TileId, TileType, UnitId, UnitType};

/// Intent enum (integer values 0..=14). Order is fixed and load-bearing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum Intent {
    BuildFarm = 0,
    BuildMine = 1,
    BuildVillage = 2,
    BuildOutpost = 3,
    BuildHydro = 4,
    BuildNuclear = 5,
    Expand = 6,
    HireSoldier = 7,
    Attack = 8,
    StackProducer = 9,
    Pass = 10,
    /// Strange-Device arc. Kept AFTER Pass so the existing values are unchanged;
    /// only INTENT_COUNT grows 11→12 (policy input 63→64, network-breaking).
    BuildStrangeDevice = 11,
    /// Plan-B action-space expansion (DEEP-REDESIGN-MEMO §6.2). Build a Bridge on
    /// an owned River tile to unblock expansion + offensive routing. Affordable
    /// guarded by `bridge_build_cost()`. Parity-locked with the TS mirror.
    BuildBridge = 12,
    /// Plan-B action-space expansion (DEEP-REDESIGN-MEMO §6.2). Attack-on-Device
    /// as a FIRST-CLASS intent so the value head has a distinct signal for the
    /// single biggest loss source (HARD's Device line). Functionally an
    /// `Action::Attack` against the enemy device tile.
    CrackDevice = 13,
    /// Plan-B action-space expansion (Plan-B addendum). Attack-on-HQ as a
    /// FIRST-CLASS intent. Functionally an `Action::Attack` against an
    /// un-conquered enemy Headquarters; the SEPARATE intent label lets the value
    /// head see the defender count it needs to beat (§3 strict-greater conquest).
    CrackHQ = 14,
    /// "Complete the eyes" action-space expansion. Relocate ONE free OWN Soldier in
    /// a single rangeless move (GAME-MECHANICS §1) onto the own-or-neutral reachable
    /// tile that most reduces Manhattan distance to the nearest enemy objective
    /// (enemy-owned Strange Device, else un-conquered enemy HQ). Gives the value head
    /// a non-attacking *manoeuvre* signal — staging an army toward the front without
    /// committing to an assault. NOT a game-rules change (no arc bump). Parity-locked
    /// with the TS mirror.
    MarchSoldier = 15,
}

pub const INTENT_COUNT: usize = 16;
pub const LOCAL_DIM: usize = 16;

/// Max distinct Expand target tiles emitted as candidates per turn (after sort).
pub const EXPAND_CANDIDATE_CAP: usize = 6;
/// Max distinct feasible Attack target tiles emitted as candidates per turn (after sort).
pub const ATTACK_CANDIDATE_CAP: usize = 4;
/// Max distinct MarchSoldier candidates (one per movable own Soldier) per turn (after sort).
pub const MARCH_CANDIDATE_CAP: usize = 4;

/// The concrete action a candidate's `execute()` performs. Mirrors the TS
/// closures exactly (including their internal fallback chains).
#[derive(Debug, Clone)]
pub enum Action {
    Build(&'static str, TileId),
    /// Expand: move idle worker if present (and not already on tile), else hire,
    /// else move a surplus worker.
    Expand {
        tile: TileId,
        idle: Option<(UnitId, TileId)>,
        can_hire: bool,
        surplus: Option<(UnitId, TileId)>,
    },
    BuyUnit(&'static str, TileId),
    /// Attack: bring `needed` soldiers onto `tile`, moving free soldiers first,
    /// buying when `can_buy`.
    Attack {
        tile: TileId,
        needed: i64,
        placed: i64,
        can_buy: bool,
    },
    /// March: relocate a single OWN Soldier `unit` from `from` to `to` in one
    /// rangeless move (the destination is own-or-neutral + reachable + has space).
    March {
        unit: UnitId,
        from: TileId,
        to: TileId,
    },
    Pass,
}

/// A candidate the policy scores. `local` and `intent` are the network features;
/// `action` is replayed by [`execute_action`].
#[derive(Debug, Clone)]
pub struct Candidate {
    pub intent: Intent,
    pub local: Vec<f64>,
    pub action: Action,
    pub label: String,
}

// --- small helpers ---------------------------------------------------------

fn claim_value(g: &Game, tid: TileId) -> i64 {
    if let Some(b) = &g.tiles[tid.0].building {
        if b.kind == BuildingType::Mikontalo {
            return 6;
        }
    }
    match g.tiles[tid.0].tile_type {
        TileType::Mountain => 5,
        TileType::Grassland => 4,
        TileType::Forest => 3,
        TileType::AbundantForest => 2,
        _ => 1,
    }
}

fn money_cost(c: &ResourceMap) -> i64 {
    -(c.get(BasicResource::Money).unwrap_or(0))
}

fn building_type(g: &Game, tid: TileId) -> Option<BuildingType> {
    g.tiles[tid.0].building.as_ref().map(|b| b.kind)
}

fn tile_threatened(g: &Game, tid: TileId, p: PlayerId) -> bool {
    for ntid in g.neighbour_tiles(tid) {
        let o = g.tiles[ntid.0].owner;
        if o.is_some()
            && o != Some(p)
            && g
                .tile_units(ntid)
                .iter()
                .any(|&u| g.units[u.0].kind == UnitType::Soldier)
        {
            return true;
        }
    }
    false
}

fn first_worker_on(g: &Game, tid: TileId) -> Option<UnitId> {
    g.tile_units(tid)
        .iter()
        .copied()
        .find(|&u| g.units[u.0].kind == UnitType::BasicWorker)
}

/// `findIdleWorker(p)` — a worker on an owned plain tile (no building, not forest).
fn find_idle_worker(g: &Game, p: PlayerId) -> Option<(UnitId, TileId)> {
    for tid in m::owned_tiles(g, p) {
        let ty = g.tiles[tid.0].tile_type;
        if g.tiles[tid.0].building.is_some()
            || ty == TileType::Forest
            || ty == TileType::AbundantForest
        {
            continue;
        }
        if let Some(w) = first_worker_on(g, tid) {
            return Some((w, tid));
        }
    }
    None
}

/// `findSurplusProducerWorker(p)`.
fn find_surplus_producer_worker(g: &Game, p: PlayerId) -> Option<(UnitId, TileId)> {
    for tid in m::owned_tiles(g, p) {
        let ty = building_type(g, tid);
        if matches!(
            ty,
            Some(BuildingType::Mine) | Some(BuildingType::Nuclear) | Some(BuildingType::Hydro)
        ) {
            let ws: Vec<UnitId> = g
                .tile_units(tid)
                .iter()
                .copied()
                .filter(|&u| g.units[u.0].kind == UnitType::BasicWorker)
                .collect();
            if ws.len() > 1 {
                return Some((ws[ws.len() - 1], tid));
            }
        }
    }
    if m::wood(g, p) >= 350 {
        for tid in m::owned_tiles(g, p) {
            if g.tiles[tid.0].tile_type != TileType::Forest {
                continue;
            }
            if let Some(w) = first_worker_on(g, tid) {
                return Some((w, tid));
            }
        }
    }
    None
}

/// `findFreeSoldier(p, exclude)`.
fn find_free_soldier(g: &Game, p: PlayerId, exclude: TileId) -> Option<(UnitId, TileId)> {
    for tid in m::owned_tiles(g, p) {
        if tid == exclude {
            continue;
        }
        if let Some(s) = g
            .tile_units(tid)
            .iter()
            .copied()
            .find(|&u| g.units[u.0].kind == UnitType::Soldier)
        {
            return Some((s, tid));
        }
    }
    None
}

#[inline]
fn clamp3(v: f64) -> f64 {
    if v < -3.0 {
        -3.0
    } else if v > 3.0 {
        3.0
    } else {
        v
    }
}

/// clamp(v, 0, 1) — matches the TS `clamp(sp.*, 0, 1)` for the neighbor ratios.
#[inline]
fn clamp01(v: f64) -> f64 {
    if v < 0.0 {
        0.0
    } else if v > 1.0 {
        1.0
    } else {
        v
    }
}

/// clamp(v, 0, 3) — matches the TS `clamp(sp.*, 0, 3)` for distances/frontier.
#[inline]
fn clamp03(v: f64) -> f64 {
    if v < 0.0 {
        0.0
    } else if v > 3.0 {
        3.0
    } else {
        v
    }
}

/// The 6 spatial/positional per-target features (local indices 10–15). Mirror of
/// the TS `SpatialFeatures`. Raw (unclamped) values flow out of [`tile_spatial`];
/// clamps are applied in [`local_vec`]. `frontier` (slot 15) is set per-intent by
/// the caller (Expand frontier flag vs Attack own-soldier-adjacent count / 3).
#[derive(Debug, Clone, Copy, Default)]
struct SpatialFeatures {
    enemy_neighbors: f64,       // 10: (#8-neighbors owned by !in {none,p}) / 8
    own_neighbors: f64,         // 11: (#8-neighbors owned by p) / 8
    neutral_neighbors: f64,     // 12: (#8-neighbors with no owner) / 8
    dist_own_hq: f64,           // 13: Manhattan(tile, HQ; none->99) / 20
    dist_nearest_enemy: f64,    // 14: min Manhattan(tile, enemy tile; none->99) / 20
    frontier: f64,              // 15: set per-intent by caller
}

/// Compute the 6 spatial features for a target tile. Reuses the parity-proven
/// `g.neighbour_tiles(tid)` (8-neighbors), tile `.owner`, `g.get_hq_tile(p)` and
/// tile `.x/.y` — no inline neighbour/coordinate math.
///
/// "enemy" = owner is Some AND != the acting player (neutral/None excluded).
/// Manhattan distance |dx|+|dy| on integer coords; sentinel 99 (missing HQ / no
/// enemy tiles) is applied BEFORE dividing by 20. Slot 15 (frontier) is filled by
/// the caller per intent. Clamps are applied in `local_vec`.
fn tile_spatial(g: &Game, tid: TileId, p: PlayerId, enemy_coords: &[(i32, i32)]) -> SpatialFeatures {
    let mut enemy_n = 0i64;
    let mut own_n = 0i64;
    let mut neutral_n = 0i64;
    for ntid in g.neighbour_tiles(tid) {
        match g.tiles[ntid.0].owner {
            None => neutral_n += 1,
            Some(o) if o == p => own_n += 1,
            Some(_) => enemy_n += 1,
        }
    }
    let tx = g.tiles[tid.0].x;
    let ty = g.tiles[tid.0].y;

    let mut dist_hq = 99i32;
    if let Some(hq) = g.get_hq_tile(p) {
        dist_hq = (tx - g.tiles[hq.0].x).abs() + (ty - g.tiles[hq.0].y).abs();
    }

    let mut dist_enemy = 99i32;
    for &(ex, ey) in enemy_coords {
        let d = (tx - ex).abs() + (ty - ey).abs();
        if d < dist_enemy {
            dist_enemy = d;
        }
    }

    SpatialFeatures {
        enemy_neighbors: enemy_n as f64 / 8.0,
        own_neighbors: own_n as f64 / 8.0,
        neutral_neighbors: neutral_n as f64 / 8.0,
        dist_own_hq: dist_hq as f64 / 20.0,
        dist_nearest_enemy: dist_enemy as f64 / 20.0,
        frontier: 0.0,
    }
}

/// Coordinates of all enemy-owned tiles (owner Some AND != p; neutral excluded).
/// One pass over `g.get_tiles()` order (min-reduction is commutative → order won't
/// affect the value). Precomputed once per `enumerate()` and threaded into
/// expand/attack.
fn enemy_tile_coords(g: &Game, p: PlayerId) -> Vec<(i32, i32)> {
    let mut out = Vec::new();
    for t in g.get_tiles() {
        if let Some(o) = t.owner {
            if o != p {
                out.push((t.x, t.y));
            }
        }
    }
    out
}

/// The (x,y) of the nearest enemy *objective* the army should march on: an
/// enemy-owned standing Strange Device first, else the first un-conquered enemy HQ
/// in canonical (`get_tiles`, column-major) order, else `None`. Mirrors
/// `enemyObjectiveCoord` in `src/ai/nn/candidates.ts` (parity-locked).
fn enemy_objective_coord(g: &Game, p: PlayerId) -> Option<(i32, i32)> {
    // (a) enemy-owned standing Strange Device.
    if let Some(dtid) = g.find_strange_device_tile() {
        if let Some(o) = g.tiles[dtid.0].owner {
            if o != p {
                return Some((g.tiles[dtid.0].x, g.tiles[dtid.0].y));
            }
        }
    }
    // (b) first un-conquered enemy HQ in get_tiles() (column-major) order.
    for t in g.get_tiles() {
        if let Some(b) = &t.building {
            if b.kind == BuildingType::Headquarters
                && !b.conquered
                && matches!(t.owner, Some(o) if o != p)
            {
                return Some((t.x, t.y));
            }
        }
    }
    None
}

/// Options for [`local_vec`]. Mirrors the TS `localVec` opts object.
struct Local {
    cost: Option<ResourceMap>,
    net_delta: f64,
    target_value: f64,
    unit_cap_gain: f64,
    soldier_cap_gain: f64,
    threatened: bool,
    income_staffing: bool,
    /// Spatial/positional per-target features (local indices 10–15). Defaults to
    /// all-zero for non-positional intents (Build*/Hire/Stack/Pass).
    spatial: SpatialFeatures,
}

impl Default for Local {
    fn default() -> Self {
        Local {
            cost: None,
            net_delta: 0.0,
            target_value: 0.0,
            unit_cap_gain: 0.0,
            soldier_cap_gain: 0.0,
            threatened: false,
            income_staffing: false,
            spatial: SpatialFeatures::default(),
        }
    }
}

/// Port of `localVec` — the 10-dim per-candidate feature vector.
fn local_vec(g: &Game, p: PlayerId, o: &Local) -> Vec<f64> {
    let cm = match &o.cost {
        Some(c) => money_cost(c) as f64,
        None => 0.0,
    };
    let wood_need = match &o.cost {
        Some(c) => -(c.get(BasicResource::Wood).unwrap_or(0)) as f64,
        None => 0.0,
    };
    let upkeep = m::wood_upkeep(g, p);
    let buffer = if upkeep > 0.0 {
        100.0_f64.max(upkeep * 5.0)
    } else {
        0.0
    };
    vec![
        clamp3(cm / 1000.0),
        clamp3(o.net_delta / 100.0),
        clamp3(o.target_value / 6.0),
        clamp3(o.unit_cap_gain / 3.0),
        clamp3(o.soldier_cap_gain / 3.0),
        if o.threatened { 1.0 } else { 0.0 },
        clamp3((m::money(g, p) as f64 - 120.0 - m::money_drain_per_round(g, p) * 5.0) / 1000.0),
        if o.income_staffing { 1.0 } else { 0.0 },
        clamp3((m::wood(g, p) as f64 - wood_need - buffer) / 500.0),
        // NN feature scale constant — DELIBERATELY left at 50 (no longer == the soldier
        // metal cost, which was rebalanced 50 → 30 in arc sd3). It is a normalization
        // offset, not a rule; changing it would shift the feature distribution mid-arc.
        clamp3((m::metal(g, p) as f64 - 50.0) / 500.0),
        // --- spatial/positional (indices 10–15) ---
        clamp01(o.spatial.enemy_neighbors),
        clamp01(o.spatial.own_neighbors),
        clamp01(o.spatial.neutral_neighbors),
        clamp03(o.spatial.dist_own_hq),
        clamp03(o.spatial.dist_nearest_enemy),
        clamp03(o.spatial.frontier),
    ]
}

// --- intent builders -------------------------------------------------------

/// Owned empty grassland tiles where a Farm can be built.
fn empty_grassland(g: &Game, p: PlayerId) -> Vec<TileId> {
    m::owned_tiles(g, p)
        .into_iter()
        .filter(|&t| {
            g.tiles[t.0].tile_type == TileType::Grassland
                && g.tiles[t.0].building.is_none()
                && g.buildable_buildings(t).contains(&"Farm")
        })
        .collect()
}

fn build_farm(g: &Game, p: PlayerId, _cfg: &TierConfig) -> Option<Candidate> {
    let spots = empty_grassland(g, p);
    if spots.is_empty() {
        return None;
    }
    let cost = farm_build_cost();
    if !s::affords_income_build(g, p, &cost, 40) || !s::has_wood_buffer(g, p, &cost) {
        return None;
    }
    let staffed = spots
        .iter()
        .copied()
        .find(|&t| m::has_type(g, t, UnitType::BasicWorker));
    let spot = staffed.unwrap_or(spots[0]);
    let local = local_vec(
        g,
        p,
        &Local {
            cost: Some(cost),
            net_delta: 44.0,
            target_value: 4.0,
            income_staffing: staffed.is_some(),
            ..Default::default()
        },
    );
    Some(Candidate {
        intent: Intent::BuildFarm,
        local,
        action: Action::Build("Farm", spot),
        label: "BuildFarm".to_string(),
    })
}

fn build_mine(g: &Game, p: PlayerId, cfg: &TierConfig) -> Option<Candidate> {
    if m::wood(g, p) < 300 {
        return None;
    }
    let mountain = m::owned_tiles(g, p).into_iter().find(|&t| {
        g.tiles[t.0].tile_type == TileType::Mountain && g.tiles[t.0].building.is_none()
    })?;
    let cost = mine_build_cost();
    if !s::affords(g, p, &cost, cfg.reserve) || !s::has_wood_buffer(g, p, &cost) {
        return None;
    }
    let local = local_vec(
        g,
        p,
        &Local {
            cost: Some(cost),
            net_delta: 20.0,
            target_value: 5.0,
            ..Default::default()
        },
    );
    Some(Candidate {
        intent: Intent::BuildMine,
        local,
        action: Action::Build("Mine", mountain),
        label: "BuildMine".to_string(),
    })
}

fn build_village(g: &Game, p: PlayerId, cfg: &TierConfig) -> Option<Candidate> {
    let spot = *empty_grassland(g, p).first()?;
    // Sustainability gates.
    let has_forest = m::owned_tiles(g, p).into_iter().any(|t| {
        g.tiles[t.0].tile_type == TileType::Forest
            && (g.tiles[t.0].building.is_none() || m::has_type(g, t, UnitType::BasicWorker))
    });
    if !has_forest {
        return None;
    }
    // -10 = new village money upkeep (-5, arc sd4) + 5 buffer. Was -15 at -10 upkeep.
    if m::net_money_per_round(g, p) - 10.0 < 0.0 {
        return None;
    }
    let post_upkeep = m::wood_upkeep(g, p) + 10.0;
    if (m::wood(g, p) as f64 - 100.0) < 100.0_f64.max(post_upkeep * 5.0) {
        return None;
    }
    let cost = village_build_cost();
    if !s::affords(g, p, &cost, cfg.reserve) {
        return None;
    }
    let local = local_vec(
        g,
        p,
        &Local {
            cost: Some(cost),
            net_delta: -5.0, // arc sd4 unit-cap rebalance (was -10)
            target_value: 4.0,
            unit_cap_gain: 3.0,
            ..Default::default()
        },
    );
    Some(Candidate {
        intent: Intent::BuildVillage,
        local,
        action: Action::Build("Village", spot),
        label: "BuildVillage".to_string(),
    })
}

fn build_outpost(g: &Game, p: PlayerId, cfg: &TierConfig) -> Option<Candidate> {
    if !cfg.military {
        return None;
    }
    // Tile gate lowered 12→8 to match the HARD bot (hard_ai.rs) — the old ≥12 was an
    // asymmetric handicap that delayed the NN's army past HARD's. Parity-locked with
    // buildOutpost in src/ai/nn/candidates.ts.
    if g.get_tile_count_for_player(p) < 8 {
        return None;
    }
    if m::net_money_per_round(g, p) < 0.0 {
        return None;
    }
    let outposts = m::building_counts(g, p).outpost;
    // Per-Outpost metal upkeep gate. The * 5.0 mirrors the Outpost metal upkeep in
    // resources.rs (rebalanced -15 → -5, arc sd3) — if left at 15 it would RE-CREATE the
    // unreachability bug. Parity-locked with buildOutpost in src/ai/nn/candidates.ts.
    if m::metal_income_per_round(g, p) - (outposts as f64 + 1.0) * 5.0 < 0.0 {
        return None;
    }
    let spot = m::owned_tiles(g, p).into_iter().find(|&t| {
        g.tiles[t.0].tile_type == TileType::Grassland
            && g.tiles[t.0].building.is_none()
            && g.buildable_buildings(t).contains(&"Outpost")
    })?;
    let cost = outpost_build_cost();
    // Affordability: the LIGHT, terminal-style standard (mirrors `build_strange_device`)
    // rather than `affords`' `reserve + 5×drain` buffer. The strict buffer made the
    // Outpost UNREACHABLE for a reinvesting economy: after paying 650 money the residual
    // (~30) fell below `reserve(100) + 5×drain`, so the candidate was never even OFFERED
    // to the net (confirmed by gate-tracing a developed 78-tile state). We instead offer
    // it whenever the player can (a) literally pay the raw cost, (b) carry the Outpost's
    // −50 money/round upkeep WITHOUT going net-negative, and (c) keep a small cash floor.
    // The metal-income + net-income (pre-build) + tiles≥12 gates above still guard
    // sustainability, so bankruptcy risk stays low while the net is free to LEARN the
    // outpost→soldier-cap→army→device chain via MCTS + value. Mirrors `buildOutpost` in
    // src/ai/nn/candidates.ts (parity-locked).
    if !g.players[p.0].has_enough_resources(&cost) {
        return None;
    }
    if m::net_money_per_round(g, p) - 50.0 < 0.0 {
        return None;
    }
    let outpost_money = cost.get(BasicResource::Money).unwrap_or(0); // negative
    if m::money(g, p) + outpost_money < 50 {
        return None;
    }
    let local = local_vec(
        g,
        p,
        &Local {
            cost: Some(cost),
            net_delta: -50.0,
            target_value: 3.0,
            soldier_cap_gain: 3.0,
            ..Default::default()
        },
    );
    Some(Candidate {
        intent: Intent::BuildOutpost,
        local,
        action: Action::Build("Outpost", spot),
        label: "BuildOutpost".to_string(),
    })
}

fn build_hydro(g: &Game, p: PlayerId, cfg: &TierConfig) -> Option<Candidate> {
    if !cfg.experts {
        return None;
    }
    if m::net_money_per_round(g, p) <= 0.0 {
        return None;
    }
    let river = m::owned_tiles(g, p).into_iter().find(|&t| {
        g.tiles[t.0].tile_type == TileType::River
            && g.tiles[t.0].building.is_none()
            && g
                .buildable_buildings(t)
                .contains(&"Hydroelectric Power Plant")
    })?;
    let cost = hepp_build_cost();
    if !s::affords(g, p, &cost, cfg.reserve.min(80)) || !s::has_wood_buffer(g, p, &cost) {
        return None;
    }
    let local = local_vec(
        g,
        p,
        &Local {
            cost: Some(cost),
            net_delta: 80.0,
            target_value: 3.0,
            ..Default::default()
        },
    );
    Some(Candidate {
        intent: Intent::BuildHydro,
        local,
        action: Action::Build("Hydroelectric Power Plant", river),
        label: "BuildHydro".to_string(),
    })
}

fn build_nuclear(g: &Game, p: PlayerId, cfg: &TierConfig) -> Option<Candidate> {
    if !cfg.experts || !cfg.nuclear {
        return None;
    }
    if m::money(g, p) <= 2600 || g.free_unit_amount(p) <= 1 {
        return None;
    }
    let spot = empty_grassland(g, p)
        .into_iter()
        .find(|&t| !m::has_type(g, t, UnitType::BasicWorker))?;
    let cost = nuclearpp_build_cost();
    if !s::affords(g, p, &cost, cfg.reserve) || !s::has_wood_buffer(g, p, &cost) {
        return None;
    }
    let local = local_vec(
        g,
        p,
        &Local {
            cost: Some(cost),
            net_delta: 160.0,
            target_value: 5.0,
            ..Default::default()
        },
    );
    Some(Candidate {
        intent: Intent::BuildNuclear,
        local,
        action: Action::Build("Nuclear Power Plant", spot),
        label: "BuildNuclear".to_string(),
    })
}

/// # of a tile's 8-neighbours owned by an enemy (owner Some AND != p; neutral
/// excluded). Mirror of the TS `enemyBorderCount` / hard_ai `enemy_border_count`.
fn enemy_border_count(g: &Game, tid: TileId, p: PlayerId) -> i64 {
    let mut n = 0i64;
    for nb in g.neighbour_tiles(tid) {
        if matches!(g.tiles[nb.0].owner, Some(o) if o != p) {
            n += 1;
        }
    }
    n
}

/// The Strange Device endgame as a neural intent — mirror of the TS
/// `buildStrangeDevice` (candidates.ts) and the hard bot's gating: build it only
/// when enabled, no Device exists (one per game), the game has matured (≥18
/// rounds), we are NOT losing on tiles, and the economy can carry the
/// one-time cost as a terminal play. Placed on the safest interior grassland
/// (fewest enemy-bordering neighbours), which must be empty (the Device holds no
/// units). No Outpost is required (the game allows building the Device on any
/// buildable grassland) — the net is free to consider it without one.
fn build_strange_device(g: &Game, p: PlayerId, cfg: &TierConfig) -> Option<Candidate> {
    if !cfg.device {
        return None;
    }
    if g.has_strange_device() {
        return None; // one per game (counterplay handles an enemy's)
    }
    if g.get_rounds_played() < 18 {
        return None;
    }
    let my_tiles = g.get_tile_count_for_player(p);
    let not_losing = g
        .live_players()
        .iter()
        .all(|&q| q == p || g.get_tile_count_for_player(q) <= my_tiles);
    if !not_losing {
        return None;
    }
    let cost = strange_device_build_cost();
    if !g.players[p.0].has_enough_resources(&cost) {
        return None;
    }
    if m::net_money_per_round(g, p) < 0.0 {
        return None;
    }
    let device_money = cost.get(BasicResource::Money).unwrap_or(0);
    if m::money(g, p) + device_money < 150 {
        return None;
    }
    // Interior grassland: owned, empty, Device-buildable, fewest enemy-bordering
    // neighbours first (stable sort, matching the TS `.sort`).
    let mut spots: Vec<TileId> = m::owned_tiles(g, p)
        .into_iter()
        .filter(|&t| {
            g.tiles[t.0].tile_type == TileType::Grassland
                && g.tiles[t.0].building.is_none()
                && g.tiles[t.0].units.is_empty()
                && g.buildable_buildings(t).contains(&"Strange Device")
        })
        .collect();
    spots.sort_by_key(|&t| enemy_border_count(g, t, p));
    let spot = *spots.first()?;
    let local = local_vec(
        g,
        p,
        &Local {
            cost: Some(cost),
            net_delta: 0.0,
            target_value: 6.0,
            ..Default::default()
        },
    );
    Some(Candidate {
        intent: Intent::BuildStrangeDevice,
        local,
        action: Action::Build("Strange Device", spot),
        label: "BuildStrangeDevice".to_string(),
    })
}

/// Plan-B `Intent::BuildBridge`. Owned River tile, no building, river orientation
/// allows a Bridge (`buildable_buildings` returns "Bridge"), and the player can
/// afford `bridge_build_cost()`. Local feature `bridge_unblock_count` = how many
/// additional tiles enter `get_available_tiles()` reachability if this Bridge is
/// built (cheap simulation: temporarily mark the river as bridged and recount).
/// Mirrors `buildBridge` in `src/ai/nn/candidates.ts` (parity-locked).
fn build_bridge(g: &Game, p: PlayerId, cfg: &TierConfig) -> Option<Candidate> {
    // Find any owned River tile with no building and a Bridge-allowing orientation.
    let river = m::owned_tiles(g, p).into_iter().find(|&t| {
        g.tiles[t.0].tile_type == TileType::River
            && g.tiles[t.0].building.is_none()
            && g.buildable_buildings(t).contains(&"Bridge")
    })?;
    let cost = bridge_build_cost();
    // Light affordability: must pay the raw cost AND keep a non-negative net money
    // after build (Bridge upkeep is -5 wood/round, no money drain). The TS mirror
    // uses the same gate.
    if !g.players[p.0].has_enough_resources(&cost) {
        return None;
    }
    if !s::has_wood_buffer(g, p, &cost) {
        return None;
    }
    // Keep a small cash floor (mirrors the Outpost/Device terminal-style gate).
    let money_cost = cost.get(BasicResource::Money).unwrap_or(0); // negative
    if m::money(g, p) + money_cost < cfg.reserve {
        return None;
    }
    // bridge_unblock_count: simulate what `get_available_tiles_for(p)` would return
    // if THIS river tile were bridged (no building suffices to satisfy the gate, the
    // candidate's interest is only the new neighbour exposure). Cheap O(neighbours):
    // count orthogonal-4 neighbours of the river that the player does NOT already
    // own AND that are not already in their availability set. This is the additive
    // gain a Bridge here would yield (the river itself becomes a passable connector).
    let pre_avail = g.get_available_tiles_for(p);
    let mut unblock_count: i64 = 0;
    for n in g.neighbour_four_tiles(river) {
        if g.tiles[n.0].owner == Some(p) {
            continue; // already owned
        }
        if pre_avail.contains(&n) {
            continue; // already reachable
        }
        // The river-as-bridge would expose `n` to availability (subject to
        // `has_opponent_headquarters` which is true for non-own-HQ tiles).
        if g.has_opponent_headquarters(n, p) {
            unblock_count += 1;
        }
    }
    let local = local_vec(
        g,
        p,
        &Local {
            cost: Some(cost),
            net_delta: -5.0, // -5 wood/round upkeep ≈ small negative money-equiv
            target_value: unblock_count as f64,
            ..Default::default()
        },
    );
    Some(Candidate {
        intent: Intent::BuildBridge,
        local,
        action: Action::Build("Bridge", river),
        label: "BuildBridge".to_string(),
    })
}

/// Plan-B `Intent::CrackDevice`. Enumerate when ANY enemy owns a standing Strange
/// Device AND the champ can stage at least one soldier onto that tile via
/// `get_available_tiles()`. Action is functionally an `Action::Attack`; the
/// SEPARATE intent label gives the value head a distinct signal for the cracker
/// (the single biggest loss source). Local includes `enemy_device_countdown`.
/// Mirrors `crackDevice` in `src/ai/nn/candidates.ts` (parity-locked).
fn crack_device(
    g: &Game,
    p: PlayerId,
    cfg: &TierConfig,
    enemy_coords: &[(i32, i32)],
) -> Option<Candidate> {
    if !cfg.military {
        return None;
    }
    // Find the (at most one) Strange Device tile NOT owned by us.
    let dtid = g.find_strange_device_tile()?;
    if g.tiles[dtid.0].owner == Some(p) {
        return None;
    }
    // The device tile must be reachable AND have room for a conquering unit (per
    // §3 attack legality). Use `get_available_tiles_for(p)`.
    let avail = g.get_available_tiles_for(p);
    if !avail.contains(&dtid) {
        return None;
    }
    if !g.tiles[dtid.0].has_space_for_conquering_units() {
        return None;
    }
    let defenders = g
        .tile_units(dtid)
        .iter()
        .filter(|&&u| g.units[u.0].kind == UnitType::Soldier)
        .count() as i64;
    if defenders >= 3 {
        return None;
    }
    let needed = defenders + 1;
    let placed = g
        .tile_conquering_units(dtid)
        .iter()
        .filter(|&&u| g.units[u.0].owner == Some(p) && g.units[u.0].kind == UnitType::Soldier)
        .count() as i64;
    let to_add = needed - placed;
    if to_add <= 0 {
        // Already staged enough; the regular Attack candidate covers this.
        return None;
    }
    let can_buy = m::money(g, p) >= cfg.reserve + 250;
    let movable: i64 = m::owned_tiles(g, p)
        .into_iter()
        .map(|tt| {
            if tt == dtid {
                0
            } else {
                g.tile_units(tt)
                    .iter()
                    .filter(|&&u| g.units[u.0].kind == UnitType::Soldier)
                    .count() as i64
            }
        })
        .sum();
    let buyable = if can_buy {
        g.free_soldier_amount(p)
            .min(m::metal(g, p) / 30) // soldier metal cost (rebalanced 50→30, arc sd3; parity-locked)
            .min((m::money(g, p) - cfg.reserve) / 200)
    } else {
        0
    };
    if movable + buyable < to_add {
        return None;
    }
    let countdown = g.tiles[dtid.0]
        .building
        .as_ref()
        .map(|b| b.countdown)
        .unwrap_or(0) as f64;
    let mut spatial = tile_spatial(g, dtid, p, enemy_coords);
    let mut own_soldier_neighbors = 0i64;
    for nb in g.neighbour_tiles(dtid) {
        own_soldier_neighbors += g
            .tile_units(nb)
            .iter()
            .filter(|&&u| g.units[u.0].kind == UnitType::Soldier && g.units[u.0].owner == Some(p))
            .count() as i64;
    }
    spatial.frontier = own_soldier_neighbors as f64 / 3.0;
    let local = local_vec(
        g,
        p,
        &Local {
            net_delta: 0.0,
            // High target_value: cracking a device prevents an imminent loss; encode
            // the countdown urgency (lower countdown = more urgent, capped at 6).
            target_value: (6.0 - countdown.min(6.0)).max(0.0),
            spatial,
            ..Default::default()
        },
    );
    Some(Candidate {
        intent: Intent::CrackDevice,
        local,
        action: Action::Attack {
            tile: dtid,
            needed,
            placed,
            can_buy,
        },
        label: "CrackDevice".to_string(),
    })
}

/// Plan-B `Intent::CrackHQ`. Enumerate when ANY enemy owns an un-conquered
/// Headquarters AND the champ can stage at least one soldier on that tile via
/// `get_available_tiles()`. Action is functionally an `Action::Attack`; SEPARATE
/// intent label so the value head sees the defender count it needs to beat (§3
/// strict-greater conquest). Local includes `enemy_hq_neighbour_soldier_count`.
/// Mirrors `crackHQ` in `src/ai/nn/candidates.ts` (parity-locked).
fn crack_hq(
    g: &Game,
    p: PlayerId,
    cfg: &TierConfig,
    enemy_coords: &[(i32, i32)],
) -> Option<Candidate> {
    if !cfg.military {
        return None;
    }
    // Find an un-conquered enemy HQ that is reachable + has space.
    let avail = g.get_available_tiles_for(p);
    let mut hq_tile: Option<TileId> = None;
    for &t in &avail {
        let tile = &g.tiles[t.0];
        if let Some(b) = &tile.building {
            if b.kind == BuildingType::Headquarters
                && !b.conquered
                && tile.owner.is_some()
                && tile.owner != Some(p)
                && tile.has_space_for_conquering_units()
            {
                hq_tile = Some(t);
                break;
            }
        }
    }
    let hq = hq_tile?;
    let defenders = g
        .tile_units(hq)
        .iter()
        .filter(|&&u| g.units[u.0].kind == UnitType::Soldier)
        .count() as i64;
    if defenders >= 3 {
        return None;
    }
    let needed = defenders + 1;
    let placed = g
        .tile_conquering_units(hq)
        .iter()
        .filter(|&&u| g.units[u.0].owner == Some(p) && g.units[u.0].kind == UnitType::Soldier)
        .count() as i64;
    let to_add = needed - placed;
    if to_add <= 0 {
        return None;
    }
    let can_buy = m::money(g, p) >= cfg.reserve + 250;
    let movable: i64 = m::owned_tiles(g, p)
        .into_iter()
        .map(|tt| {
            if tt == hq {
                0
            } else {
                g.tile_units(tt)
                    .iter()
                    .filter(|&&u| g.units[u.0].kind == UnitType::Soldier)
                    .count() as i64
            }
        })
        .sum();
    let buyable = if can_buy {
        g.free_soldier_amount(p)
            .min(m::metal(g, p) / 30) // soldier metal cost (rebalanced 50→30, arc sd3; parity-locked)
            .min((m::money(g, p) - cfg.reserve) / 200)
    } else {
        0
    };
    if movable + buyable < to_add {
        return None;
    }
    let mut spatial = tile_spatial(g, hq, p, enemy_coords);
    let mut own_soldier_neighbors = 0i64;
    for nb in g.neighbour_tiles(hq) {
        own_soldier_neighbors += g
            .tile_units(nb)
            .iter()
            .filter(|&&u| g.units[u.0].kind == UnitType::Soldier && g.units[u.0].owner == Some(p))
            .count() as i64;
    }
    spatial.frontier = own_soldier_neighbors as f64 / 3.0;
    let local = local_vec(
        g,
        p,
        &Local {
            net_delta: 0.0,
            // Maximum target value (an HQ crack collapses an opponent via the
            // connectivity rule); the value head sees the defender count via spatial
            // + own_soldier_neighbors frontier and the local soldier-cap slot.
            target_value: 6.0,
            spatial,
            ..Default::default()
        },
    );
    Some(Candidate {
        intent: Intent::CrackHQ,
        local,
        action: Action::Attack {
            tile: hq,
            needed,
            placed,
            can_buy,
        },
        label: "CrackHQ".to_string(),
    })
}

/// Multi-candidate Expand: one candidate per plausible neutral target tile.
fn expand(g: &Game, p: PlayerId, cfg: &TierConfig, enemy_coords: &[(i32, i32)]) -> Vec<Candidate> {
    // neutral, unowned, has room, not threatened.
    let mut neutral: Vec<TileId> = g
        .get_available_tiles()
        .into_iter()
        .filter(|&t| {
            g.tiles[t.0].owner.is_none()
                && g.tiles[t.0].has_space_for_units()
                && !tile_threatened(g, t, p)
        })
        .collect();
    if neutral.is_empty() {
        return Vec::new();
    }

    // Reachability is a per-turn property (idle/hire/surplus), independent of
    // which neutral tile we target — computed once. Bail if no worker deliverable.
    let idle = find_idle_worker(g, p);
    let can_hire = g.free_unit_amount(p) > 0
        && s::affords(g, p, &basic_worker_cost(), cfg.reserve)
        && m::net_money_per_round(g, p) - 5.0 >= 0.0;
    let surplus = if idle.is_none() && !can_hire {
        find_surplus_producer_worker(g, p)
    } else {
        None
    };
    if idle.is_none() && !can_hire && surplus.is_none() {
        return Vec::new();
    }

    // Total order: claimValue DESC, then tile-index ASC (TileId.0 is the
    // column-major generation index). Cap AFTER sorting.
    neutral.sort_by(|&a, &b| {
        claim_value(g, b)
            .cmp(&claim_value(g, a))
            .then(a.0.cmp(&b.0))
    });
    neutral.truncate(EXPAND_CANDIDATE_CAP);

    neutral
        .into_iter()
        .map(|tile| {
            let cap_gain = if building_type(g, tile) == Some(BuildingType::Mikontalo) {
                2.0
            } else {
                0.0
            };
            let cost = if can_hire && idle.is_none() {
                Some(basic_worker_cost())
            } else {
                None
            };
            let mut spatial = tile_spatial(g, tile, p, enemy_coords);
            // Expand: slot-15 frontier flag = 1 if any enemy neighbor else 0.
            spatial.frontier = if spatial.enemy_neighbors > 0.0 { 1.0 } else { 0.0 };
            let local = local_vec(
                g,
                p,
                &Local {
                    cost,
                    net_delta: if idle.is_some() { 0.0 } else { -5.0 },
                    target_value: claim_value(g, tile) as f64,
                    unit_cap_gain: cap_gain,
                    spatial,
                    ..Default::default()
                },
            );
            Candidate {
                intent: Intent::Expand,
                local,
                action: Action::Expand {
                    tile,
                    idle,
                    can_hire,
                    surplus,
                },
                label: "Expand".to_string(),
            }
        })
        .collect()
}

fn hire_soldier(g: &Game, p: PlayerId, cfg: &TierConfig) -> Option<Candidate> {
    if !cfg.military {
        return None;
    }
    if g.free_soldier_amount(p) <= 0 {
        return None;
    }
    if m::metal(g, p) < 30 { // soldier metal cost (rebalanced 50→30, arc sd3; parity-locked)
        return None;
    }
    let cost = soldier_cost();
    if !s::affords(g, p, &cost, cfg.reserve) || !s::can_afford_upkeep(g, p, 30.0) {
        return None;
    }
    let hq = g.get_hq_tile(p);
    let threatened: Vec<TileId> = m::owned_tiles(g, p)
        .into_iter()
        .filter(|&t| Some(t) != hq && tile_threatened(g, t, p))
        .collect();
    let tile = threatened
        .first()
        .copied()
        .or(hq)
        .or_else(|| {
            m::owned_tiles(g, p)
                .into_iter()
                .find(|&t| g.tiles[t.0].has_space_for_units())
        });
    let tile = match tile {
        Some(t) if g.tiles[t.0].has_space_for_units() => t,
        _ => return None,
    };
    let local = local_vec(
        g,
        p,
        &Local {
            cost: Some(cost),
            net_delta: -30.0,
            soldier_cap_gain: 0.0,
            threatened: !threatened.is_empty(),
            ..Default::default()
        },
    );
    Some(Candidate {
        intent: Intent::HireSoldier,
        local,
        action: Action::BuyUnit("Soldier", tile),
        label: "HireSoldier".to_string(),
    })
}

/// Multi-candidate Attack: one candidate per feasible enemy target tile.
fn attack(g: &Game, p: PlayerId, cfg: &TierConfig, enemy_coords: &[(i32, i32)]) -> Vec<Candidate> {
    if !cfg.military {
        return Vec::new();
    }
    let can_buy = m::money(g, p) >= cfg.reserve + 250;

    struct Target {
        tile: TileId,
        defenders: i64,
        is_hq: bool,
        is_outpost: bool,
    }
    let mut targets: Vec<Target> = g
        .get_available_tiles()
        .into_iter()
        .filter(|&t| {
            let o = g.tiles[t.0].owner;
            o.is_some() && o != Some(p) && g.tiles[t.0].has_space_for_conquering_units()
        })
        .map(|t| {
            let defenders = g
                .tile_units(t)
                .iter()
                .filter(|&&u| g.units[u.0].kind == UnitType::Soldier)
                .count() as i64;
            Target {
                tile: t,
                defenders,
                is_hq: building_type(g, t) == Some(BuildingType::Headquarters),
                is_outpost: building_type(g, t) == Some(BuildingType::Outpost),
            }
        })
        .filter(|t| !t.is_outpost && t.defenders < 3)
        .collect();
    // Total order: HQ-first, then fewest defenders, then tile-index ASC (TileId.0
    // is the column-major generation index). Cap (counting FEASIBLE emitted
    // candidates) applied after sorting in the loop below.
    targets.sort_by(|a, b| {
        (b.is_hq as i64)
            .cmp(&(a.is_hq as i64))
            .then(a.defenders.cmp(&b.defenders))
            .then(a.tile.0.cmp(&b.tile.0))
    });

    let mut out: Vec<Candidate> = Vec::new();
    for t in &targets {
        if out.len() >= ATTACK_CANDIDATE_CAP {
            break;
        }
        let needed = t.defenders + 1;
        let placed = g
            .tile_conquering_units(t.tile)
            .iter()
            .filter(|&&u| g.units[u.0].owner == Some(p) && g.units[u.0].kind == UnitType::Soldier)
            .count() as i64;
        let to_add = needed - placed;
        if to_add <= 0 {
            continue;
        }
        let movable: i64 = m::owned_tiles(g, p)
            .into_iter()
            .map(|tt| {
                if tt == t.tile {
                    0
                } else {
                    g.tile_units(tt)
                        .iter()
                        .filter(|&&u| g.units[u.0].kind == UnitType::Soldier)
                        .count() as i64
                }
            })
            .sum();
        let buyable = if can_buy {
            g.free_soldier_amount(p)
                .min(m::metal(g, p) / 30) // soldier metal cost (rebalanced 50→30, arc sd3; parity-locked)
                .min((m::money(g, p) - cfg.reserve) / 200)
        } else {
            0
        };
        if movable + buyable < to_add {
            continue;
        }
        let mut spatial = tile_spatial(g, t.tile, p, enemy_coords);
        // Attack frontier (slot 15): my Soldiers on the target's 8-neighbours / 3
        // (counted by soldier owner == p, regardless of the neighbour tile's owner).
        let mut own_soldier_neighbors = 0i64;
        for nb in g.neighbour_tiles(t.tile) {
            own_soldier_neighbors += g
                .tile_units(nb)
                .iter()
                .filter(|&&u| {
                    g.units[u.0].kind == UnitType::Soldier && g.units[u.0].owner == Some(p)
                })
                .count() as i64;
        }
        spatial.frontier = own_soldier_neighbors as f64 / 3.0;
        let local = local_vec(
            g,
            p,
            &Local {
                net_delta: 0.0,
                target_value: if t.is_hq { 6.0 } else { (4 - t.defenders) as f64 },
                soldier_cap_gain: 0.0,
                spatial,
                ..Default::default()
            },
        );
        out.push(Candidate {
            intent: Intent::Attack,
            local,
            action: Action::Attack {
                tile: t.tile,
                needed,
                placed,
                can_buy,
            },
            label: format!("Attack{}", if t.is_hq { ":HQ" } else { "" }),
        });
    }
    out
}

/// `Intent::MarchSoldier` — relocate ONE free OWN Soldier, in one rangeless move
/// (GAME-MECHANICS §1), to the own-or-neutral reachable tile that most reduces the
/// Manhattan distance to the nearest enemy objective (enemy-owned Strange Device,
/// else un-conquered enemy HQ). Enemy-owned tiles are excluded (those are
/// Attack/CrackHQ/CrackDevice). One candidate per movable soldier, capped at
/// `MARCH_CANDIDATE_CAP`; only emitted when the best destination STRICTLY reduces
/// distance. Mirrors `marchSoldierCandidates` in `src/ai/nn/candidates.ts`
/// (parity-locked).
fn march_soldier(
    g: &Game,
    p: PlayerId,
    cfg: &TierConfig,
    enemy_coords: &[(i32, i32)],
) -> Vec<Candidate> {
    if !cfg.military {
        return Vec::new();
    }
    let (ox, oy) = match enemy_objective_coord(g, p) {
        Some(c) => c,
        None => return Vec::new(),
    };

    // Reachable own-or-neutral destinations with room for a unit.
    let dests: Vec<TileId> = g
        .get_available_tiles_for(p)
        .into_iter()
        .filter(|&t| {
            let o = g.tiles[t.0].owner;
            (o == Some(p) || o.is_none()) && g.tiles[t.0].has_space_for_units()
        })
        .collect();

    // (d_best, from_tile_id, candidate) collected, then totally ordered.
    struct March {
        d_best: i64,
        from: TileId,
        cand: Candidate,
    }
    let mut marches: Vec<March> = Vec::new();

    // One candidate per movable own Soldier (find_free_soldier scan pattern: iterate
    // owned tiles, find a Soldier on each).
    for from in m::owned_tiles(g, p) {
        let soldier = g
            .tile_units(from)
            .iter()
            .copied()
            .find(|&u| g.units[u.0].kind == UnitType::Soldier);
        let unit = match soldier {
            Some(u) => u,
            None => continue,
        };
        let sx = g.tiles[from.0].x;
        let sy = g.tiles[from.0].y;
        let d0 = ((sx - ox).abs() + (sy - oy).abs()) as i64;

        // Best destination != from minimising Manhattan distance to the objective;
        // tie-break on destination tile-id ASC.
        let mut best: Option<(i64, TileId)> = None;
        for &dest in &dests {
            if dest == from {
                continue;
            }
            let dx = g.tiles[dest.0].x;
            let dy = g.tiles[dest.0].y;
            let d = ((dx - ox).abs() + (dy - oy).abs()) as i64;
            match best {
                Some((bd, bt)) if d > bd || (d == bd && dest.0 >= bt.0) => {}
                _ => best = Some((d, dest)),
            }
        }
        let (d_best, dest) = match best {
            Some(v) => v,
            None => continue,
        };
        if d_best >= d0 {
            continue; // must STRICTLY reduce distance
        }

        let mut spatial = tile_spatial(g, dest, p, enemy_coords);
        spatial.frontier = if spatial.enemy_neighbors > 0.0 { 1.0 } else { 0.0 };
        let local = local_vec(
            g,
            p,
            &Local {
                target_value: (d0 - d_best) as f64,
                spatial,
                ..Default::default()
            },
        );
        marches.push(March {
            d_best,
            from,
            cand: Candidate {
                intent: Intent::MarchSoldier,
                local,
                action: Action::March {
                    unit,
                    from,
                    to: dest,
                },
                label: "MarchSoldier".to_string(),
            },
        });
    }

    // Total order: d_best ASC, then from tile-id ASC. Cap AFTER sorting.
    marches.sort_by(|a, b| a.d_best.cmp(&b.d_best).then(a.from.0.cmp(&b.from.0)));
    marches.truncate(MARCH_CANDIDATE_CAP);
    marches.into_iter().map(|x| x.cand).collect()
}

fn stack_producer(g: &Game, p: PlayerId, cfg: &TierConfig) -> Option<Candidate> {
    if g.free_unit_amount(p) <= 0 {
        return None;
    }
    let tile = m::owned_tiles(g, p).into_iter().find(|&t| {
        let ty = building_type(g, t);
        matches!(
            ty,
            Some(BuildingType::Mine) | Some(BuildingType::Nuclear) | Some(BuildingType::Hydro)
        ) && g.tiles[t.0].has_space_for_units()
    })?;
    let want_expert = cfg.experts
        && building_type(g, tile) != Some(BuildingType::Hydro)
        && !m::has_type(g, tile, UnitType::Expert)
        && g.free_unit_amount(p) > 1;
    let cost = if want_expert {
        expert_cost()
    } else {
        basic_worker_cost()
    };
    let reserve = if want_expert {
        cfg.reserve
    } else {
        s::STAFF_RESERVE
    };
    if !s::affords(g, p, &cost, reserve) {
        return None;
    }
    let local = local_vec(
        g,
        p,
        &Local {
            cost: Some(cost),
            net_delta: 20.0,
            target_value: 3.0,
            income_staffing: true,
            ..Default::default()
        },
    );
    let unit = if want_expert { "Expert" } else { "BasicWorker" };
    Some(Candidate {
        intent: Intent::StackProducer,
        local,
        action: Action::BuyUnit(unit, tile),
        label: format!("StackProducer{}", if want_expert { ":Expert" } else { "" }),
    })
}

fn pass_candidate() -> Candidate {
    Candidate {
        intent: Intent::Pass,
        local: vec![0.0; LOCAL_DIM],
        action: Action::Pass,
        label: "Pass".to_string(),
    }
}

/// `enumerate(ctx)` — builders in fixed order, then PASS appended.
///
/// Build* / HireSoldier / StackProducer are single-candidate (0 or 1). Expand
/// and Attack are MULTI-candidate: they emit one candidate per plausible target
/// tile, spread into the list in their builders' total-sorted order. Pass last.
pub fn enumerate(g: &Game, p: PlayerId, cfg: &TierConfig) -> Vec<Candidate> {
    type Single = fn(&Game, PlayerId, &TierConfig) -> Option<Candidate>;
    let mut out = Vec::new();
    let push_single = |out: &mut Vec<Candidate>, b: Single| {
        if let Some(c) = b(g, p, cfg) {
            out.push(c);
        }
    };
    push_single(&mut out, build_farm);
    push_single(&mut out, build_mine);
    push_single(&mut out, build_village);
    push_single(&mut out, build_outpost);
    push_single(&mut out, build_hydro);
    push_single(&mut out, build_nuclear);
    push_single(&mut out, build_strange_device);
    push_single(&mut out, build_bridge);
    let enemy_coords = enemy_tile_coords(g, p);
    out.extend(expand(g, p, cfg, &enemy_coords));
    push_single(&mut out, hire_soldier);
    out.extend(attack(g, p, cfg, &enemy_coords));
    // Plan-B Crack candidates: piggy-back on `Action::Attack` against the device/HQ
    // tile, but emit with a DISTINCT intent label so the value head sees them.
    if let Some(c) = crack_device(g, p, cfg, &enemy_coords) {
        out.push(c);
    }
    if let Some(c) = crack_hq(g, p, cfg, &enemy_coords) {
        out.push(c);
    }
    // MarchSoldier: rangeless manoeuvre of a free Soldier toward the nearest enemy
    // objective. Position is load-bearing for parity (after crack_hq, before
    // stack_producer).
    out.extend(march_soldier(g, p, cfg, &enemy_coords));
    push_single(&mut out, stack_producer);
    out.push(pass_candidate());
    out
}

/// Perform a candidate's action (the TS `execute()` closure). Returns whether it
/// did anything (the controller treats `false` as a failed execute).
pub fn execute_action(g: &mut Game, p: PlayerId, cfg: &TierConfig, action: &Action) -> bool {
    match action {
        Action::Build(name, tid) => g.ai_build_building(name, *tid),
        Action::BuyUnit(name, tid) => g.ai_buy_and_place_unit(name, *tid),
        Action::Pass => true,
        Action::Expand {
            tile,
            idle,
            can_hire,
            surplus,
        } => {
            if let Some((unit, from)) = idle {
                if *from != *tile {
                    return g.ai_move_unit(*unit, *from, *tile);
                }
            }
            if *can_hire {
                return g.ai_buy_and_place_unit("BasicWorker", *tile);
            }
            if let Some((unit, from)) = surplus {
                if *from != *tile {
                    return g.ai_move_unit(*unit, *from, *tile);
                }
            }
            false
        }
        Action::Attack {
            tile,
            needed,
            placed,
            can_buy,
        } => {
            let mut cur = *placed;
            let mut did = false;
            while cur < *needed {
                let spare = find_free_soldier(g, p, *tile);
                let mut step = false;
                if let Some((unit, from)) = spare {
                    step = g.ai_move_unit(unit, from, *tile);
                } else if *can_buy
                    && g.free_soldier_amount(p) > 0
                    && m::metal(g, p) >= 30 // soldier metal cost (rebalanced 50→30, arc sd3)
                    && s::affords(g, p, &soldier_cost(), cfg.reserve)
                {
                    step = g.ai_buy_and_place_unit("Soldier", *tile);
                }
                if !step {
                    break;
                }
                did = true;
                cur += 1;
            }
            did
        }
        Action::March { unit, from, to } => g.ai_move_unit(*unit, *from, *to),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cp_sim::Game;

    /// Hand-built board test for `tile_spatial` + `enemy_tile_coords`, guarding the
    /// neighbor/distance/owner logic independent of full parity replay.
    ///
    /// Layout (12x12, column-major idx = x*12 + y). Acting player = P1 (PlayerId 0),
    /// enemy = P2 (PlayerId 1). Target = (5,5), idx 65. We set its 8-neighbors:
    ///   enemy (P2):  (4,4), (5,4), (6,4)            -> 3 enemy neighbors
    ///   own   (P1):  (4,5), (6,5)                   -> 2 own neighbors
    ///   neutral:     (4,6), (5,6), (6,6)            -> 3 neutral neighbors
    /// P1 HQ at (5,8): Manhattan(5,5 -> 5,8) = 3.
    /// Closest enemy-owned tile is a neighbor at distance 1 -> distNearestEnemy = 1.
    #[test]
    fn tile_spatial_hand_computed() {
        let mut g = Game::new(12, 12, &["P1", "P2"]);
        g.generate_map(12, 12, 1);
        let p1 = PlayerId(0);
        let p2 = PlayerId(1);

        let id = |x: i32, y: i32| TileId((x * 12 + y) as usize);
        let target = id(5, 5);

        // Enemy neighbors.
        for c in [(4, 4), (5, 4), (6, 4)] {
            g.set_tile_owner(id(c.0, c.1), Some(p2));
        }
        // Own neighbors.
        for c in [(4, 5), (6, 5)] {
            g.set_tile_owner(id(c.0, c.1), Some(p1));
        }
        // Remaining neighbors (4,6),(5,6),(6,6) stay None (neutral).

        // P1 HQ at (5,8): owner P1 + Headquarters building.
        let hq = id(5, 8);
        g.set_tile_owner(hq, Some(p1));
        g.place_building(hq, BuildingType::Headquarters, Some(p1));

        let enemy_coords = enemy_tile_coords(&g, p1);
        // Exactly the 3 enemy neighbors are enemy-owned.
        assert_eq!(enemy_coords.len(), 3);

        let sp = tile_spatial(&g, target, p1, &enemy_coords);
        assert!((sp.enemy_neighbors - 3.0 / 8.0).abs() < 1e-12);
        assert!((sp.own_neighbors - 2.0 / 8.0).abs() < 1e-12);
        assert!((sp.neutral_neighbors - 3.0 / 8.0).abs() < 1e-12);
        assert!((sp.dist_own_hq - 3.0 / 20.0).abs() < 1e-12);
        assert!((sp.dist_nearest_enemy - 1.0 / 20.0).abs() < 1e-12);
        assert_eq!(sp.frontier, 0.0); // set per-intent by caller

        // Attack slot-15: place 2 of P1's soldiers on neighbor tiles + 1 of P2's
        // (the enemy soldier must NOT be counted; counting is by soldier owner==p1).
        g.spawn_unit_on_tile(UnitType::Soldier, p1, id(4, 5), false);
        g.spawn_unit_on_tile(UnitType::Soldier, p1, id(6, 5), false);
        g.spawn_unit_on_tile(UnitType::Soldier, p2, id(5, 4), false);
        let mut own_soldier_neighbors = 0i64;
        for nb in g.neighbour_tiles(target) {
            own_soldier_neighbors += g
                .tile_units(nb)
                .iter()
                .filter(|&&u| {
                    g.units[u.0].kind == UnitType::Soldier && g.units[u.0].owner == Some(p1)
                })
                .count() as i64;
        }
        assert_eq!(own_soldier_neighbors, 2);
        assert!(((own_soldier_neighbors as f64 / 3.0) - 2.0 / 3.0).abs() < 1e-12);
    }

    /// Plan-B `Intent::BuildBridge` must enumerate when the player owns a River
    /// tile (bridge-allowing orientation), there is no building on it, and they
    /// can afford the cost. Mirrors `buildBridge` in
    /// `src/ai/nn/candidates.ts` — parity-locked.
    #[test]
    fn bridge_candidate_emits_on_owned_river_no_building() {
        let mut g = Game::new(12, 12, &["P1", "P2"]);
        g.generate_map(12, 12, 1);
        let p1 = PlayerId(0);

        // Force a river tile we own with no building + orientation 0.
        let id = |x: i32, y: i32| TileId((x * 12 + y) as usize);
        let river = id(5, 5);
        g.tiles[river.0].tile_type = TileType::River;
        g.tiles[river.0].river_orientation = 0;
        g.tiles[river.0].building = None;
        g.set_tile_owner(river, Some(p1));

        // Treasury — generous so affordability gates pass.
        g.set_player_resources(p1, 1500, 1500, 1500, 1500);

        // Sanity: the engine offers Bridge for this tile.
        assert!(g.buildable_buildings(river).contains(&"Bridge"));

        let cfg = crate::tiers::TRAINING_CONFIG;
        let cand = build_bridge(&g, p1, &cfg).expect("Bridge candidate must emit");
        assert_eq!(cand.intent, Intent::BuildBridge);
        match cand.action {
            Action::Build(name, tid) => {
                assert_eq!(name, "Bridge");
                assert_eq!(tid, river);
            }
            _ => panic!("BuildBridge candidate must use Action::Build"),
        }
    }

    /// Plan-B `Intent::BuildBridge` must NOT enumerate when the river tile already
    /// has a building (e.g. an existing Bridge or a Hydroelectric Power Plant). The
    /// candidate gate filters on `g.tiles[t.0].building.is_none()` — covered here.
    #[test]
    fn bridge_candidate_does_not_emit_when_river_has_building() {
        let mut g = Game::new(12, 12, &["P1", "P2"]);
        g.generate_map(12, 12, 1);
        let p1 = PlayerId(0);

        let id = |x: i32, y: i32| TileId((x * 12 + y) as usize);
        let river = id(5, 5);
        g.tiles[river.0].tile_type = TileType::River;
        g.tiles[river.0].river_orientation = 0;
        // PRE-place a Bridge on this river tile.
        g.place_building(river, BuildingType::Bridge, Some(p1));
        g.set_tile_owner(river, Some(p1));
        g.set_player_resources(p1, 1500, 1500, 1500, 1500);

        let cfg = crate::tiers::TRAINING_CONFIG;
        assert!(
            build_bridge(&g, p1, &cfg).is_none(),
            "Bridge candidate must not emit when the river already has a building"
        );
    }

    /// Plan-B `Intent::CrackDevice` must enumerate when an enemy owns a standing
    /// Strange Device tile AND the champ can stage ≥1 soldier on it. Hand-built
    /// fixture: enemy device on a tile adjacent to a player-owned tile holding a
    /// soldier (so the device tile enters `get_available_tiles_for(p)` and the
    /// soldier can be moved onto it). Mirrors `crackDevice` in
    /// `src/ai/nn/candidates.ts` — parity-locked.
    #[test]
    fn crack_device_candidate_emits_when_enemy_device_present_and_reachable() {
        let mut g = Game::new(12, 12, &["P1", "P2"]);
        g.generate_map(12, 12, 1);
        let p1 = PlayerId(0);
        let p2 = PlayerId(1);

        let id = |x: i32, y: i32| TileId((x * 12 + y) as usize);
        // Champion HQ at (5,8) so availability flows from there.
        let hq = id(5, 8);
        g.tiles[hq.0].tile_type = TileType::Grassland;
        g.tiles[hq.0].building = None;
        g.set_tile_owner(hq, Some(p1));
        g.place_building(hq, BuildingType::Headquarters, Some(p1));
        // Own a tile adjacent to the device tile so it enters availability.
        let stage = id(5, 6);
        g.tiles[stage.0].tile_type = TileType::Grassland;
        g.tiles[stage.0].building = None;
        g.set_tile_owner(stage, Some(p1));
        // Enemy device at (5,5) — adjacent to (5,6) under 4-neighbour BFS.
        let dev = id(5, 5);
        g.tiles[dev.0].tile_type = TileType::Grassland;
        g.set_tile_owner(dev, Some(p2));
        g.place_building(dev, BuildingType::StrangeDevice, Some(p2));
        // A soldier we can move off the staging tile (not on the device tile itself).
        g.spawn_unit_on_tile(UnitType::Soldier, p1, stage, false);
        g.set_player_resources(p1, 2000, 2000, 2000, 2000);
        // Force a positive soldier cap so `free_soldier_amount` doesn't go negative
        // (no end_turn was run, so the cached cap is the default 0).
        g.update_unit_amounts(p1);
        // HQ + extra soldier headroom for the test (just so free_soldier_amount > 0).
        g.players[p1.0].max_soldier_amount = g.players[p1.0].max_soldier_amount.max(3);

        let cfg = crate::tiers::TRAINING_CONFIG;
        let enemy_coords = enemy_tile_coords(&g, p1);
        let cand =
            crack_device(&g, p1, &cfg, &enemy_coords).expect("CrackDevice candidate must emit");
        assert_eq!(cand.intent, Intent::CrackDevice);
        match cand.action {
            Action::Attack { tile, .. } => assert_eq!(tile, dev),
            _ => panic!("CrackDevice candidate must use Action::Attack"),
        }
    }

    /// Plan-B `Intent::CrackHQ` must enumerate when an enemy owns an un-conquered
    /// Headquarters AND the champ can stage ≥1 soldier on it (the tile enters
    /// `get_available_tiles_for(p)` and a soldier can move onto it). Mirrors
    /// `crackHQ` in `src/ai/nn/candidates.ts` — parity-locked.
    #[test]
    fn crack_hq_candidate_emits_when_enemy_hq_present_and_reachable() {
        let mut g = Game::new(12, 12, &["P1", "P2"]);
        g.generate_map(12, 12, 1);
        let p1 = PlayerId(0);
        let p2 = PlayerId(1);

        let id = |x: i32, y: i32| TileId((x * 12 + y) as usize);
        // Champion HQ at (5,9) so availability flows from there.
        let my_hq = id(5, 9);
        g.tiles[my_hq.0].tile_type = TileType::Grassland;
        g.tiles[my_hq.0].building = None;
        g.set_tile_owner(my_hq, Some(p1));
        g.place_building(my_hq, BuildingType::Headquarters, Some(p1));
        // Own a tile adjacent to the enemy HQ so the HQ enters availability.
        let stage = id(5, 7);
        g.tiles[stage.0].tile_type = TileType::Grassland;
        g.tiles[stage.0].building = None;
        g.set_tile_owner(stage, Some(p1));
        // Enemy HQ at (5,6) — un-conquered.
        let enemy_hq = id(5, 6);
        g.tiles[enemy_hq.0].tile_type = TileType::Grassland;
        g.set_tile_owner(enemy_hq, Some(p2));
        g.place_building(enemy_hq, BuildingType::Headquarters, Some(p2));
        // A movable soldier.
        g.spawn_unit_on_tile(UnitType::Soldier, p1, stage, false);
        g.set_player_resources(p1, 2000, 2000, 2000, 2000);
        g.update_unit_amounts(p1);
        g.players[p1.0].max_soldier_amount = g.players[p1.0].max_soldier_amount.max(3);

        let cfg = crate::tiers::TRAINING_CONFIG;
        let enemy_coords = enemy_tile_coords(&g, p1);
        let cand = crack_hq(&g, p1, &cfg, &enemy_coords).expect("CrackHQ candidate must emit");
        assert_eq!(cand.intent, Intent::CrackHQ);
        match cand.action {
            Action::Attack { tile, .. } => assert_eq!(tile, enemy_hq),
            _ => panic!("CrackHQ candidate must use Action::Attack"),
        }
    }

    /// `Intent::MarchSoldier` must emit a candidate moving an own Soldier strictly
    /// closer (Manhattan) to an enemy objective. Fixture: own soldier at distance d0
    /// from an enemy HQ + a reachable own/neutral tile at d0-1. Mirrors
    /// `marchSoldierCandidates` in `src/ai/nn/candidates.ts` (parity-locked).
    #[test]
    fn march_soldier_emits_and_reduces_distance() {
        let mut g = Game::new(12, 12, &["P1", "P2"]);
        g.generate_map(12, 12, 1);
        let p1 = PlayerId(0);
        let p2 = PlayerId(1);

        let id = |x: i32, y: i32| TileId((x * 12 + y) as usize);
        // Champion HQ at (5,9) so availability flows from there.
        let my_hq = id(5, 9);
        g.tiles[my_hq.0].tile_type = TileType::Grassland;
        g.tiles[my_hq.0].building = None;
        g.set_tile_owner(my_hq, Some(p1));
        g.place_building(my_hq, BuildingType::Headquarters, Some(p1));
        // A soldier sits on an owned tile at (5,8): distance to enemy HQ (5,5) = 3.
        let from = id(5, 8);
        g.tiles[from.0].tile_type = TileType::Grassland;
        g.tiles[from.0].building = None;
        g.set_tile_owner(from, Some(p1));
        g.spawn_unit_on_tile(UnitType::Soldier, p1, from, false);
        // A reachable own destination at (5,7): distance to enemy HQ = 2 (< 3).
        let dest = id(5, 7);
        g.tiles[dest.0].tile_type = TileType::Grassland;
        g.tiles[dest.0].building = None;
        g.set_tile_owner(dest, Some(p1));
        // Enemy HQ at (5,5), un-conquered.
        let enemy_hq = id(5, 5);
        g.tiles[enemy_hq.0].tile_type = TileType::Grassland;
        g.set_tile_owner(enemy_hq, Some(p2));
        g.place_building(enemy_hq, BuildingType::Headquarters, Some(p2));

        let cfg = crate::tiers::TRAINING_CONFIG;
        let enemy_coords = enemy_tile_coords(&g, p1);
        let cands = march_soldier(&g, p1, &cfg, &enemy_coords);
        assert!(!cands.is_empty(), "MarchSoldier candidate must emit");
        let c = &cands[0];
        assert_eq!(c.intent, Intent::MarchSoldier);
        match c.action {
            Action::March { from: f, to, .. } => {
                assert_eq!(f, from);
                let d0 = (g.tiles[f.0].x - g.tiles[enemy_hq.0].x).abs()
                    + (g.tiles[f.0].y - g.tiles[enemy_hq.0].y).abs();
                let d_to = (g.tiles[to.0].x - g.tiles[enemy_hq.0].x).abs()
                    + (g.tiles[to.0].y - g.tiles[enemy_hq.0].y).abs();
                assert!(d_to < d0, "March destination must strictly reduce distance");
            }
            _ => panic!("MarchSoldier candidate must use Action::March"),
        }
    }

    /// No enemy device and no un-conquered enemy HQ → no objective → no MarchSoldier
    /// candidate.
    #[test]
    fn march_soldier_no_candidate_without_objective() {
        let mut g = Game::new(12, 12, &["P1", "P2"]);
        g.generate_map(12, 12, 1);
        let p1 = PlayerId(0);

        let id = |x: i32, y: i32| TileId((x * 12 + y) as usize);
        let my_hq = id(5, 9);
        g.tiles[my_hq.0].tile_type = TileType::Grassland;
        g.tiles[my_hq.0].building = None;
        g.set_tile_owner(my_hq, Some(p1));
        g.place_building(my_hq, BuildingType::Headquarters, Some(p1));
        let from = id(5, 8);
        g.tiles[from.0].tile_type = TileType::Grassland;
        g.set_tile_owner(from, Some(p1));
        g.spawn_unit_on_tile(UnitType::Soldier, p1, from, false);

        let cfg = crate::tiers::TRAINING_CONFIG;
        let enemy_coords = enemy_tile_coords(&g, p1);
        assert!(enemy_objective_coord(&g, p1).is_none());
        assert!(
            march_soldier(&g, p1, &cfg, &enemy_coords).is_empty(),
            "MarchSoldier must not emit without an enemy objective"
        );
    }

    /// MarchSoldier destinations exclude enemy-owned tiles: a closer ENEMY tile must
    /// not be chosen — only own/neutral reachable tiles are valid destinations.
    #[test]
    fn march_soldier_excludes_enemy_tiles() {
        let mut g = Game::new(12, 12, &["P1", "P2"]);
        g.generate_map(12, 12, 1);
        let p1 = PlayerId(0);
        let p2 = PlayerId(1);

        let id = |x: i32, y: i32| TileId((x * 12 + y) as usize);
        let my_hq = id(5, 9);
        g.tiles[my_hq.0].tile_type = TileType::Grassland;
        g.tiles[my_hq.0].building = None;
        g.set_tile_owner(my_hq, Some(p1));
        g.place_building(my_hq, BuildingType::Headquarters, Some(p1));
        // Soldier at (5,8), objective enemy HQ at (5,5).
        let from = id(5, 8);
        g.tiles[from.0].tile_type = TileType::Grassland;
        g.tiles[from.0].building = None;
        g.set_tile_owner(from, Some(p1));
        g.spawn_unit_on_tile(UnitType::Soldier, p1, from, false);
        // A CLOSER enemy-owned tile at (5,6) (distance 1 to the HQ) — must be EXCLUDED.
        let enemy_tile = id(5, 6);
        g.tiles[enemy_tile.0].tile_type = TileType::Grassland;
        g.tiles[enemy_tile.0].building = None;
        g.set_tile_owner(enemy_tile, Some(p2));
        // An own destination at (5,7) (distance 2) — the only valid closer tile.
        let dest = id(5, 7);
        g.tiles[dest.0].tile_type = TileType::Grassland;
        g.tiles[dest.0].building = None;
        g.set_tile_owner(dest, Some(p1));
        // Enemy HQ at (5,5).
        let enemy_hq = id(5, 5);
        g.tiles[enemy_hq.0].tile_type = TileType::Grassland;
        g.set_tile_owner(enemy_hq, Some(p2));
        g.place_building(enemy_hq, BuildingType::Headquarters, Some(p2));

        let cfg = crate::tiers::TRAINING_CONFIG;
        let enemy_coords = enemy_tile_coords(&g, p1);
        let cands = march_soldier(&g, p1, &cfg, &enemy_coords);
        assert!(!cands.is_empty(), "MarchSoldier candidate must emit");
        match cands[0].action {
            Action::March { to, .. } => {
                assert_ne!(to, enemy_tile, "must not march onto an enemy-owned tile");
                assert_eq!(to, dest, "must pick the own/neutral closer tile");
            }
            _ => panic!("MarchSoldier candidate must use Action::March"),
        }
    }

    /// A target with no enemy tiles anywhere yields the 99-sentinel distance, and
    /// missing HQ also yields 99 (both before the /20).
    #[test]
    fn tile_spatial_sentinels() {
        let mut g = Game::new(12, 12, &["P1", "P2"]);
        g.generate_map(12, 12, 1);
        let p1 = PlayerId(0);
        let target = TileId((5 * 12 + 5) as usize);
        // No enemy tiles, no HQ owned by p1 (generate_map doesn't place HQs).
        let enemy_coords = enemy_tile_coords(&g, p1);
        assert!(enemy_coords.is_empty());
        let sp = tile_spatial(&g, target, p1, &enemy_coords);
        assert!((sp.dist_nearest_enemy - 99.0 / 20.0).abs() < 1e-12);
        assert!((sp.dist_own_hq - 99.0 / 20.0).abs() < 1e-12);
    }
}
