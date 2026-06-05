//! Port of `src/ai/nn/safety.ts` — affordability / solvency guards.
//!
//! The policy only chooses among candidates that pass these guards, so a neural
//! CPU can never bankrupt itself or make an illegal spend.

use crate::metrics as m;
use cp_sim::resources::{BasicResource, ResourceMap};
use cp_sim::{Game, PlayerId};

/// Buffer kept when hiring a worker for a net-positive building.
pub const STAFF_RESERVE: i64 = 20;

fn has_enough_resources(g: &Game, p: PlayerId, cost: &ResourceMap) -> bool {
    g.players[p.0].has_enough_resources(cost)
}

fn cost_money(cost: &ResourceMap) -> i64 {
    cost.get(BasicResource::Money).unwrap_or(0)
}

/// `affords` — keep `reserve` + ~5 rounds of drain buffered, no resource negative.
pub fn affords(g: &Game, p: PlayerId, cost: &ResourceMap, reserve: i64) -> bool {
    if !has_enough_resources(g, p, cost) {
        return false;
    }
    let buffer = reserve as f64 + m::money_drain_per_round(g, p) * 5.0;
    (m::money(g, p) + cost_money(cost)) as f64 >= buffer
}

/// `affordsIncomeBuild` — income builds only need raw resources + a money floor.
pub fn affords_income_build(g: &Game, p: PlayerId, cost: &ResourceMap, floor: i64) -> bool {
    if !has_enough_resources(g, p, cost) {
        return false;
    }
    m::money(g, p) + cost_money(cost) >= floor
}

/// `canAffordUpkeep` — taking on one more salaried unit stays net-non-negative.
pub fn can_afford_upkeep(g: &Game, p: PlayerId, salary: f64) -> bool {
    m::net_money_per_round(g, p) - salary >= 0.0
}

/// `hasWoodBuffer` — spending the cost's wood doesn't risk a wood death.
pub fn has_wood_buffer(g: &Game, p: PlayerId, cost: &ResourceMap) -> bool {
    let need = -(cost.get(BasicResource::Wood).unwrap_or(0));
    if need <= 0 {
        return true;
    }
    let upkeep = m::wood_upkeep(g, p);
    let buffer = if upkeep > 0.0 {
        100.0_f64.max(upkeep * 5.0)
    } else {
        0.0
    };
    (m::wood(g, p) - need) as f64 >= buffer
}
