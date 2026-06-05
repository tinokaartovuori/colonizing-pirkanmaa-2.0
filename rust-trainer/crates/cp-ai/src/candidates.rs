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
    basic_worker_cost, expert_cost, farm_build_cost, hepp_build_cost, mine_build_cost,
    nuclearpp_build_cost, outpost_build_cost, soldier_cost, strange_device_build_cost,
    village_build_cost, BasicResource, ResourceMap,
};
use cp_sim::{BuildingType, Game, PlayerId, TileId, TileType, UnitId, UnitType};

/// Intent enum (integer values 0..=11). Order is fixed and load-bearing.
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
}

pub const INTENT_COUNT: usize = 12;
pub const LOCAL_DIM: usize = 16;

/// Max distinct Expand target tiles emitted as candidates per turn (after sort).
pub const EXPAND_CANDIDATE_CAP: usize = 6;
/// Max distinct feasible Attack target tiles emitted as candidates per turn (after sort).
pub const ATTACK_CANDIDATE_CAP: usize = 4;

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
    if m::net_money_per_round(g, p) - 15.0 < 0.0 {
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
            net_delta: -10.0,
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
    if m::metal_income_per_round(g, p) - (outposts as f64 + 1.0) * 15.0 < 0.0 {
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
    if m::metal(g, p) < 50 {
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
                .min(m::metal(g, p) / 50)
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
    let enemy_coords = enemy_tile_coords(g, p);
    out.extend(expand(g, p, cfg, &enemy_coords));
    push_single(&mut out, hire_soldier);
    out.extend(attack(g, p, cfg, &enemy_coords));
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
                    && m::metal(g, p) >= 50
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
