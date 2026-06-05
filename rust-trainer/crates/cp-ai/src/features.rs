//! Port of `src/ai/nn/features.ts` — board-size-invariant global features.
//!
//! 36-dim vector of global aggregates (fractions, per-round rates, ratios to
//! caps), all clamped to ~[-3,3]. Index→name table is locked in
//! `rust-trainer/golden/SCHEMA.md`. Index 35 is the constant `1` bias.
//!
//! `getMaxUnitAmount`/`getMaxSoldierAmount` refresh the unit caps before reading
//! (the TS does too), so this takes `&mut Game` to preserve that ordering.

use crate::metrics as m;
use crate::spatial;
use cp_sim::{Game, PlayerId};

pub const GLOBAL_DIM: usize = 36;

/// Names of the global features, in order (also documented in SCHEMA.md).
pub const GLOBAL_FEATURE_NAMES: [&str; GLOBAL_DIM] = [
    "money", "wood", "stone", "metal",
    "netMoney", "metalIncome", "netWood", "moneyDrain",
    "tileFraction", "tileAbs",
    "maxUnit", "freeUnit", "workers", "experts",
    "maxSoldier", "freeSoldier", "soldiers",
    "staffedFarms", "mines", "villages", "outposts", "powerplants", "harvesters",
    "freeGrass", "freeMountain", "freeRiver",
    "round", "threat",
    "oppMaxFraction", "leadMargin", "oppSoldiers", "oppAlive",
    "dominationProgress", "neutralFraction", "reachableEnemy",
    "bias",
];

#[inline]
fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

#[inline]
fn c3(v: f64) -> f64 {
    clamp(v, -3.0, 3.0)
}

/// Build the global feature vector for `player`. `round` is the rounds-played
/// counter. Mirrors `globalFeatures()` exactly, including evaluation order.
pub fn global_features(g: &mut Game, player: PlayerId, round: i64) -> Vec<f64> {
    let total_tiles = g.get_tile_count().max(1) as f64;
    let my_tiles = g.get_tile_count_for_player(player) as f64;
    let bc = m::building_counts(g, player);
    let opp = m::opponent_summary(g, player);
    let neutral = g.get_neutral_tiles() as f64;

    // getMaxUnitAmount()/getMaxSoldierAmount() refresh the caps (mutating), then
    // the free/current reads use the (now-current) cached values.
    let max_unit = g.max_unit_amount(player) as f64;
    let max_soldier = g.max_soldier_amount(player) as f64;
    let free_unit = g.free_unit_amount(player) as f64;
    let free_soldier = g.free_soldier_amount(player) as f64;
    let workers = g.current_basic_worker_amount(player) as f64;
    let experts = g.current_expert_amount(player) as f64;
    let soldiers = g.current_soldier_amount(player) as f64;

    let f = vec![
        c3(m::money(g, player) as f64 / 1000.0),
        c3(m::wood(g, player) as f64 / 1000.0),
        c3(m::stone(g, player) as f64 / 1000.0),
        c3(m::metal(g, player) as f64 / 500.0),
        c3(m::net_money_per_round(g, player) / 100.0),
        c3(m::metal_income_per_round(g, player) / 100.0),
        c3((m::wood_income_per_round(g, player) - m::wood_upkeep(g, player)) / 300.0),
        c3(m::money_drain_per_round(g, player) / 200.0),
        clamp(my_tiles / total_tiles, 0.0, 1.0),
        c3(my_tiles / 40.0),
        c3(max_unit / 20.0),
        c3(free_unit / 10.0),
        c3(workers / 20.0),
        c3(experts / 10.0),
        c3(max_soldier / 15.0),
        c3(free_soldier / 10.0),
        c3(soldiers / 15.0),
        c3(bc.staffed_farms as f64 / 15.0),
        c3(bc.mine as f64 / 8.0),
        c3(bc.village as f64 / 6.0),
        c3(bc.outpost as f64 / 4.0),
        c3((bc.nuclear + bc.hydro) as f64 / 4.0),
        c3(bc.forest_harvesters as f64 / 4.0),
        c3(bc.free_grassland as f64 / 15.0),
        c3(bc.free_mountains as f64 / 6.0),
        c3(bc.free_rivers as f64 / 6.0),
        clamp(round as f64 / 60.0, 0.0, 3.0),
        c3(m::enemy_threat(g, player) / 8.0),
        clamp(opp.max_tiles as f64 / total_tiles, 0.0, 1.0),
        c3((my_tiles - opp.max_tiles as f64) / total_tiles),
        c3(opp.total_soldiers as f64 / 15.0),
        clamp(opp.alive as f64 / 3.0, 0.0, 1.0),
        clamp(my_tiles / (0.7 * total_tiles), 0.0, 2.0),
        clamp(neutral / total_tiles, 0.0, 1.0),
        if m::has_reachable_enemy(g, player) { 1.0 } else { 0.0 },
        1.0, // bias
    ];
    f
}

/// Number of board-invariant spatial summary features appended for the value net.
pub const SPATIAL_GLOBAL_DIM: usize = 5;
/// Input width of the SPATIAL value net: 36 global + 5 spatial summaries = 41.
pub const VALUE_SPATIAL_DIM: usize = GLOBAL_DIM + SPATIAL_GLOBAL_DIM;

/// Enriched VALUE-net input: the 36 global features + 5 board-invariant spatial
/// summaries (HQ-to-HQ distance, frontier length, enemy-HQ push, own dispersion,
/// HQ-connectivity cut-risk). Targets the "reach parity but can't convert"
/// ceiling by giving the leaf evaluator positional awareness. Used ONLY by a
/// spatial value net — the policy/candidates/parity path never calls this.
pub fn value_features_spatial(g: &mut Game, player: PlayerId, round: i64) -> Vec<f64> {
    let mut v = global_features(g, player, round);
    // `global_features` took &mut (refreshes caps); spatial reads are immutable.
    v.push(clamp(spatial::hq_to_hq_dist(g, player) as f64 / 20.0, 0.0, 3.0));
    v.push(spatial::frontier_fraction(g, player)); // [0,1]
    v.push(spatial::enemy_hq_push(g, player, 4)); // [0,1]
    v.push(clamp(spatial::own_dispersion(g, player), 0.0, 3.0));
    v.push(spatial::mean_cut_risk(g, player)); // [0,1]
    v
}
