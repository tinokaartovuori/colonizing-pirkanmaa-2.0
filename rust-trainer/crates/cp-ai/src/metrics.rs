//! Port of `src/ai/nn/metrics.ts` — Phaser-free economy/military metrics.
//!
//! Every formula mirrors the TS exactly. Resource state is `i64` in cp-sim;
//! we convert to `f64` precisely where the TS does its (floating-point) math so
//! the feature/score values reproduce bit-for-bit.

use cp_sim::resources::BasicResource;
use cp_sim::{BuildingType, Game, ObjId, PlayerId, TileId, TileType, UnitType};

/// `M.ownedTiles(p)` — owned tiles in `objects_` order.
pub fn owned_tiles(g: &Game, p: PlayerId) -> Vec<TileId> {
    g.owned_tiles(p)
}

pub fn money(g: &Game, p: PlayerId) -> i64 {
    g.players[p.0].resources.get(BasicResource::Money).unwrap_or(0)
}
pub fn wood(g: &Game, p: PlayerId) -> i64 {
    g.players[p.0].resources.get(BasicResource::Wood).unwrap_or(0)
}
pub fn stone(g: &Game, p: PlayerId) -> i64 {
    g.players[p.0].resources.get(BasicResource::Stone).unwrap_or(0)
}
pub fn metal(g: &Game, p: PlayerId) -> i64 {
    g.players[p.0].resources.get(BasicResource::Metal).unwrap_or(0)
}

/// "Total wealth" of a player, in money-equivalent units. This is a RELATIVE
/// telemetry metric (not game logic) used by the v3 reward to reward having
/// more accumulated value than the opponent. It sums, exactly:
///   1. Liquid resources: money + wood + stone + metal (raw i64 counts).
///   2. For every building the player OWNS (one per owned tile that has a
///      building, HQ/Mikontalo included): the MONEY component of that
///      building's `build_cost()`, as a positive amount (build costs store
///      money as a negative delta, so we negate). HQ/Mikontalo contribute 0
///      because their build cost is empty.
///   3. For every UNIT the player owns (BasicWorker/Expert/Soldier): the MONEY
///      component of that unit's `cost()`, as a positive amount (again negated
///      from the stored negative purchase delta).
/// Metal in unit/building costs is intentionally NOT counted — only the money
/// component, per the v3 spec.
pub fn total_wealth(g: &Game, p: PlayerId) -> f64 {
    let mut w = (money(g, p) + wood(g, p) + stone(g, p) + metal(g, p)) as f64;
    for obj in &g.players[p.0].objects {
        match obj {
            ObjId::Tile(tid) => {
                if let Some(bt) = building_type(g, *tid) {
                    let cost = bt.build_cost().get(BasicResource::Money).unwrap_or(0);
                    w += (-cost) as f64; // stored negative; spend = positive value
                }
            }
            ObjId::Unit(uid) => {
                let cost = g.units[uid.0].kind.cost().get(BasicResource::Money).unwrap_or(0);
                w += (-cost) as f64;
            }
        }
    }
    w
}

/// True if a tile holds a unit of `kind`.
pub fn has_type(g: &Game, tid: TileId, kind: UnitType) -> bool {
    g.tile_units(tid)
        .iter()
        .any(|&u| g.units[u.0].kind == kind)
}

/// `countWorkers(tile)` — BasicWorkers on a tile.
pub fn count_workers(g: &Game, tid: TileId) -> i64 {
    g.tile_units(tid)
        .iter()
        .filter(|&&u| g.units[u.0].kind == UnitType::BasicWorker)
        .count() as i64
}

fn building_type(g: &Game, tid: TileId) -> Option<BuildingType> {
    g.tiles[tid.0].building.as_ref().map(|b| b.kind)
}

/// `salaryPerRound(p)`.
pub fn salary_per_round(g: &Game, p: PlayerId) -> f64 {
    (g.current_basic_worker_amount(p) * 5
        + g.current_expert_amount(p) * 25
        + g.current_soldier_amount(p) * 30) as f64
}

/// `moneyDrainPerRound(p)` — wages + Village/Outpost upkeep.
pub fn money_drain_per_round(g: &Game, p: PlayerId) -> f64 {
    let mut upkeep = 0.0f64;
    for t in owned_tiles(g, p) {
        match building_type(g, t) {
            Some(BuildingType::Village) => upkeep += 10.0,
            Some(BuildingType::Outpost) => upkeep += 50.0,
            _ => {}
        }
    }
    salary_per_round(g, p) + upkeep
}

/// `netMoneyPerRound(p)` — amortised money income minus salaries.
pub fn net_money_per_round(g: &Game, p: PlayerId) -> f64 {
    let mut income = 0.0f64;
    for tid in owned_tiles(g, p) {
        let ty = building_type(g, tid);
        let workers = count_workers(g, tid);
        let expert = has_type(g, tid, UnitType::Expert);
        match ty {
            Some(BuildingType::Farm) if workers > 0 => income += 175.0 / 4.0,
            Some(BuildingType::Mine) if workers > 0 => {
                income += 20.0 * workers as f64 * if expert { 2.0 } else { 1.0 }
            }
            Some(BuildingType::Nuclear) if workers > 0 && expert => {
                income += 160.0 * workers as f64
            }
            Some(BuildingType::Hydro) if workers > 0 && expert => {
                income += 80.0 * workers as f64
            }
            _ => {
                if g.tiles[tid.0].tile_type == TileType::AbundantForest && workers > 0 {
                    income += 15.0;
                }
            }
        }
        if ty == Some(BuildingType::Village) {
            income -= 10.0;
        }
        if ty == Some(BuildingType::Outpost) {
            income -= 50.0;
        }
    }
    income - salary_per_round(g, p)
}

/// `metalIncomePerRound(p)`.
pub fn metal_income_per_round(g: &Game, p: PlayerId) -> f64 {
    let mut m = 0.0f64;
    for tid in owned_tiles(g, p) {
        if building_type(g, tid) != Some(BuildingType::Mine) {
            continue;
        }
        m += 20.0 * count_workers(g, tid) as f64 * if has_type(g, tid, UnitType::Expert) {
            2.0
        } else {
            1.0
        };
    }
    m
}

/// `woodUpkeep(p)` — Villages -10, Bridges -5.
pub fn wood_upkeep(g: &Game, p: PlayerId) -> f64 {
    let mut w = 0.0f64;
    for tid in owned_tiles(g, p) {
        match building_type(g, tid) {
            Some(BuildingType::Village) => w += 10.0,
            Some(BuildingType::Bridge) => w += 5.0,
            _ => {}
        }
    }
    w
}

/// `woodIncomePerRound(p)` — staffed forests producing ~100/worker.
pub fn wood_income_per_round(g: &Game, p: PlayerId) -> f64 {
    let mut w = 0.0f64;
    for tid in owned_tiles(g, p) {
        if g.tiles[tid.0].tile_type != TileType::Forest {
            continue;
        }
        w += count_workers(g, tid) as f64 * 100.0;
    }
    w
}

/// `BuildingCounts` for `buildingCounts(p)`.
#[derive(Debug, Default, Clone, Copy)]
pub struct BuildingCounts {
    pub farm: i64,
    pub mine: i64,
    pub village: i64,
    pub outpost: i64,
    pub nuclear: i64,
    pub hydro: i64,
    pub bridge: i64,
    pub staffed_farms: i64,
    pub forest_harvesters: i64,
    pub free_mountains: i64,
    pub free_grassland: i64,
    pub free_rivers: i64,
}

/// `buildingCounts(p)`.
pub fn building_counts(g: &Game, p: PlayerId) -> BuildingCounts {
    let mut c = BuildingCounts::default();
    for tid in owned_tiles(g, p) {
        let ty = building_type(g, tid);
        match ty {
            Some(BuildingType::Farm) => {
                c.farm += 1;
                if has_type(g, tid, UnitType::BasicWorker) {
                    c.staffed_farms += 1;
                }
            }
            Some(BuildingType::Mine) => c.mine += 1,
            Some(BuildingType::Village) => c.village += 1,
            Some(BuildingType::Outpost) => c.outpost += 1,
            Some(BuildingType::Nuclear) => c.nuclear += 1,
            Some(BuildingType::Hydro) => c.hydro += 1,
            Some(BuildingType::Bridge) => c.bridge += 1,
            _ => {}
        }
        let tile_type = g.tiles[tid.0].tile_type;
        let no_building = g.tiles[tid.0].building.is_none();
        if tile_type == TileType::Forest && no_building && has_type(g, tid, UnitType::BasicWorker) {
            c.forest_harvesters += 1;
        }
        if tile_type == TileType::Mountain && no_building {
            c.free_mountains += 1;
        }
        if tile_type == TileType::Grassland && no_building {
            c.free_grassland += 1;
        }
        if tile_type == TileType::River && no_building && !g.buildable_buildings(tid).is_empty() {
            c.free_rivers += 1;
        }
    }
    c
}

/// `enemyThreat(p)` — opposing soldiers invading our tiles or massed adjacent.
pub fn enemy_threat(g: &Game, p: PlayerId) -> f64 {
    let mut threat = 0i64;
    for tid in owned_tiles(g, p) {
        threat += g
            .tile_conquering_units(tid)
            .iter()
            .filter(|&&u| g.units[u.0].owner != Some(p) && g.units[u.0].kind == UnitType::Soldier)
            .count() as i64;
        for ntid in g.neighbour_tiles(tid) {
            let o = g.tiles[ntid.0].owner;
            if o.is_some() && o != Some(p) {
                threat += g
                    .tile_units(ntid)
                    .iter()
                    .filter(|&&u| g.units[u.0].kind == UnitType::Soldier)
                    .count() as i64;
            }
        }
    }
    threat as f64
}

/// `OpponentSummary`.
#[derive(Debug, Default, Clone, Copy)]
pub struct OpponentSummary {
    pub alive: i64,
    pub max_tiles: i64,
    pub total_tiles: i64,
    pub total_soldiers: i64,
}

/// `opponentSummary(p, om, pm)` — over the live players.
pub fn opponent_summary(g: &Game, p: PlayerId) -> OpponentSummary {
    let mut s = OpponentSummary::default();
    for &other in g.live_players() {
        if other == p {
            continue;
        }
        s.alive += 1;
        let t = g.get_tile_count_for_player(other);
        s.total_tiles += t;
        if t > s.max_tiles {
            s.max_tiles = t;
        }
        s.total_soldiers += g.current_soldier_amount(other);
    }
    s
}

/// `hasReachableEnemy(p, om)` — an enemy-owned tile is on our border (or threat).
pub fn has_reachable_enemy(g: &Game, p: PlayerId) -> bool {
    if enemy_threat(g, p) > 0.0 {
        return true;
    }
    g.get_available_tiles().iter().any(|&tid| {
        let o = g.tiles[tid.0].owner;
        o.is_some() && o != Some(p)
    })
}
