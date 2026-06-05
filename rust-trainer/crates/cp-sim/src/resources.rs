//! Port of `src/core/resources.ts` (which itself ports `Course/basicresources.*`
//! and `Core/resourcemaps.h`).
//!
//! NUMERIC TYPE DECISION — resources are stored as `i64`.
//! The TS uses plain JS `number`s, but inspection of the model/managers shows
//! resource amounts are only ever produced by integer constants multiplied by
//! integer worker counts and combined with integer add/subtract (`merge`,
//! salaries, build costs). The only divisions in the codebase live in AI
//! heuristics and UI percentage bars — never in stored resource state. So
//! resources are integers in practice; `i64` is exact and avoids float drift,
//! which matters for deterministic parity with the TS golden traces.
//!
//! RESOURCEMAP SEMANTICS — `ResourceMap` mirrors the TS `Map<BasicResource,
//! number>` (which itself mirrors C++ `std::map<BasicResource,int>`). Several
//! algorithms (`merge_resource_maps`, `has_enough_resources`) rely on the fact
//! that iteration visits **only present keys**, and JS `Map` iterates in
//! insertion order. We therefore back the map with an insertion-ordered
//! `Vec<(BasicResource, i64)>` rather than a `HashMap`, so iteration order is
//! deterministic and matches the TS exactly.

/// `Course::BasicResource`. Discriminants match the TS enum verbatim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum BasicResource {
    None = 0,
    Money = 1,
    Wood = 2,
    Stone = 3,
    Metal = 4,
}

/// Insertion-ordered map of resource -> amount.
///
/// Mirrors the TS `ResourceMap = Map<BasicResource, number>`: keys are unique,
/// iteration is in insertion order, and only present keys are visited.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResourceMap {
    entries: Vec<(BasicResource, i64)>,
}

impl ResourceMap {
    /// An empty map (TS `new Map()` / `EMPTY`).
    pub fn new() -> Self {
        ResourceMap { entries: Vec::new() }
    }

    /// Build from an ordered list of `(resource, amount)` pairs.
    ///
    /// Equivalent to the TS `rmap({...})` helper. Later duplicate keys
    /// overwrite earlier ones (matching `Map.set`), keeping the first
    /// position.
    pub fn from_pairs(pairs: &[(BasicResource, i64)]) -> Self {
        let mut m = ResourceMap::new();
        for &(k, v) in pairs {
            m.set(k, v);
        }
        m
    }

    /// `Map.get` — returns `None` when the key is absent.
    pub fn get(&self, key: BasicResource) -> Option<i64> {
        self.entries
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, v)| *v)
    }

    /// `Map.has`.
    pub fn has(&self, key: BasicResource) -> bool {
        self.entries.iter().any(|(k, _)| *k == key)
    }

    /// `Map.set` — insert or overwrite, preserving insertion order for
    /// pre-existing keys.
    pub fn set(&mut self, key: BasicResource, value: i64) {
        if let Some(slot) = self.entries.iter_mut().find(|(k, _)| *k == key) {
            slot.1 = value;
        } else {
            self.entries.push((key, value));
        }
    }

    /// Number of present keys.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the map has no keys.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate `(resource, amount)` pairs in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (BasicResource, i64)> + '_ {
        self.entries.iter().copied()
    }
}

/// `cloneResourceMap` — deep copy. (Rust `Clone` already does this; provided
/// for name-parity with the TS API.)
pub fn clone_resource_map(map: &ResourceMap) -> ResourceMap {
    map.clone()
}

/// `mergeResourceMaps`: start from `right`, then for every key in `left` either
/// insert it (if absent) or add to the existing value. Keys present in only one
/// map survive. (`Course::mergeResourceMaps`)
pub fn merge_resource_maps(left: &ResourceMap, right: &ResourceMap) -> ResourceMap {
    let mut result = right.clone();
    for (key, value) in left.iter() {
        match result.get(key) {
            None => result.set(key, value),
            Some(existing) => result.set(key, existing + value),
        }
    }
    result
}

/// `reverseResourceMap`: negate every value.
pub fn reverse_resource_map(map: &ResourceMap) -> ResourceMap {
    let mut result = ResourceMap::new();
    for (key, value) in map.iter() {
        result.set(key, value * -1);
    }
    result
}

/// `getNegativesMap`: keep negatives, zero out the rest.
pub fn get_negatives_map(map: &ResourceMap) -> ResourceMap {
    let mut result = ResourceMap::new();
    for (key, value) in map.iter() {
        result.set(key, if value < 0 { value } else { 0 });
    }
    result
}

/// `getPositivesMap`: keep `>= 0` values, zero out negatives.
pub fn get_positives_map(map: &ResourceMap) -> ResourceMap {
    let mut result = ResourceMap::new();
    for (key, value) in map.iter() {
        result.set(key, if value >= 0 { value } else { 0 });
    }
    result
}

// ---------------------------------------------------------------------------
// ConstResourceMaps and scalar economy constants (verbatim from resources.ts).
//
// The Mine / Hydroelectric / Nuclear values are a DELIBERATE rebalance away
// from the C++ original (see CLAUDE.md). The TS file is the source of truth for
// these three buildings — copied verbatim below.
// ---------------------------------------------------------------------------

use BasicResource::{Metal, Money, Stone, Wood};

/// Construct a const-style `ResourceMap` from pairs (helper for the builders
/// below; these are functions because `ResourceMap` owns a `Vec`).
fn rmap(pairs: &[(BasicResource, i64)]) -> ResourceMap {
    ResourceMap::from_pairs(pairs)
}

/// `EMPTY` — no keys at all.
pub fn empty() -> ResourceMap {
    ResourceMap::new()
}

/// `NO_RESOURCES` — all four resources present at 0 (insertion order: money,
/// wood, metal, stone — matching the TS literal order).
pub fn no_resources() -> ResourceMap {
    rmap(&[(Money, 0), (Wood, 0), (Metal, 0), (Stone, 0)])
}

/// `RESOURCE_LIMITS`.
pub fn resource_limits() -> ResourceMap {
    rmap(&[
        (Money, 9_999_999),
        (Wood, 9_999_999),
        (Stone, 9_999_999),
        (Metal, 9_999_999),
    ])
}

pub const UNIT_LIMITS: i64 = 999;

/// `STARTING_RESOURCES` — 400 money / 200 wood / 100 stone / 25 metal.
pub fn starting_resources() -> ResourceMap {
    rmap(&[(Money, 400), (Wood, 200), (Stone, 100), (Metal, 25)])
}

// Tile - Forest
pub fn forest_production() -> ResourceMap {
    rmap(&[(Wood, 100), (Stone, 10)])
}
pub fn forest_capacity() -> ResourceMap {
    rmap(&[(Wood, 600), (Stone, 60)])
}

// Tile - Abundant Forest
pub fn abundant_forest_production() -> ResourceMap {
    rmap(&[(Money, 15)])
}

pub const FOREST_GROW_TIME: i64 = 5;

// Building - Farm
pub fn farm_build_cost() -> ResourceMap {
    rmap(&[(Money, -100), (Wood, -100), (Metal, -5)])
}
pub fn farm_production() -> ResourceMap {
    rmap(&[(Money, 175)])
}
pub const FARM_GROW_TIME: i64 = 4;

// Building - Mine (REBALANCED — diverges from C++; TS is source of truth)
pub fn mine_build_cost() -> ResourceMap {
    rmap(&[(Money, -200), (Wood, -200), (Stone, 200)])
}
pub fn mine_production() -> ResourceMap {
    rmap(&[(Money, 20), (Stone, 30), (Metal, 20)])
}

// Building - Hydroelectric Power Plant (REBALANCED — TS is source of truth)
pub fn hepp_build_cost() -> ResourceMap {
    rmap(&[(Money, -280), (Wood, -150), (Stone, -120), (Metal, -60)])
}
pub fn hepp_production() -> ResourceMap {
    rmap(&[(Money, 80)])
}

// Building - Nuclear Power Plant (REBALANCED — TS is source of truth)
pub fn nuclearpp_build_cost() -> ResourceMap {
    rmap(&[(Money, -2000), (Wood, -200), (Stone, -250), (Metal, -250)])
}
pub fn nuclearpp_production() -> ResourceMap {
    rmap(&[(Money, 160)])
}

// Building - Outpost
pub fn outpost_build_cost() -> ResourceMap {
    rmap(&[(Money, -650), (Wood, -300), (Stone, -300), (Metal, -300)])
}
pub fn outpost_production() -> ResourceMap {
    rmap(&[(Money, -50), (Metal, -15)])
}
pub const OUTPOST_SOLDIER_VALUE: i64 = 3;

// Building - Bridge
pub fn bridge_build_cost() -> ResourceMap {
    rmap(&[(Money, -100), (Wood, -300), (Stone, -150)])
}
pub fn bridge_production() -> ResourceMap {
    rmap(&[(Wood, -5)])
}

// Building - Village (Neighborhood)
pub fn village_build_cost() -> ResourceMap {
    rmap(&[(Money, -200), (Wood, -200), (Stone, -100), (Metal, -25)])
}
pub fn village_production() -> ResourceMap {
    rmap(&[(Money, -10), (Wood, -10), (Stone, -10)])
}
pub const VILLAGE_UNIT_VALUE: i64 = 3;

// Building - Strange Device (DELIBERATE divergence — the original had no Device;
// see STRANGE-DEVICE-DESIGN.md / CLAUDE.md). One-time nuclear-tier build cost, NO
// per-turn drain; the balancer is the soldier-cap halving (see managers.rs
// update_unit_amounts). Mirrors STRANGE_DEVICE_* in src/core/resources.ts.
pub fn strange_device_build_cost() -> ResourceMap {
    // Tuned (2026-06-03): 1300 money (down from a 1800 nuclear-tier first pass). The
    // builder was 56% of bankruptcies at 1800 — 1300 keeps the Device a real commitment
    // while letting the (halved-army) builder stay solvent enough to DEFEND it, which
    // dropped hard-vs-hard bankruptcies 27%->~21%, raised the Device-win share to ~39%
    // of games, and put Device-survival near a coin-flip (~53%). Stays in lockstep with
    // STRANGE_DEVICE_BUILD_COST in src/core/resources.ts.
    rmap(&[(Money, -1300), (Stone, -200), (Metal, -200)])
}
pub const STRANGE_DEVICE_COUNTDOWN_BASE: i64 = 18;
pub const STRANGE_DEVICE_COUNTDOWN_PER_TILE: f64 = 0.12;
/// `strangeDeviceCountdown(tileCount)` — rounds before a standing Device wins.
/// `round(BASE + PER_TILE × tileCount)`. JS `Math.round` rounds half toward +∞;
/// Rust `f64::round` rounds half away from zero. They agree for the always-positive
/// values here (≥ 18), so `.round()` is the faithful port — do not "fix" it unless
/// BASE/PER_TILE ever go negative.
pub fn strange_device_countdown(tile_count: i64) -> i64 {
    (STRANGE_DEVICE_COUNTDOWN_BASE as f64 + STRANGE_DEVICE_COUNTDOWN_PER_TILE * tile_count as f64)
        .round() as i64
}

// Building - Mikontalo
pub const MIKONTALO_UNIT_VALUE: i64 = 2;

// Building - HQ
pub const HQ_UNIT_VALUE: i64 = 3;
pub const HQ_SOLDIER_VALUE: i64 = 1;

// Worker
pub fn basic_worker_cost() -> ResourceMap {
    rmap(&[(Money, -50)])
}
pub fn basic_worker_salary() -> ResourceMap {
    rmap(&[(Money, -5)])
}

// Expert
pub fn expert_cost() -> ResourceMap {
    rmap(&[(Money, -250)])
}
pub fn expert_salary() -> ResourceMap {
    rmap(&[(Money, -25)])
}

// Soldier
pub fn soldier_cost() -> ResourceMap {
    rmap(&[(Money, -200), (Metal, -50)])
}
pub fn soldier_salary() -> ResourceMap {
    rmap(&[(Money, -30)])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_get_has() {
        let mut m = ResourceMap::new();
        assert!(!m.has(Money));
        assert_eq!(m.get(Money), None);
        m.set(Money, 10);
        assert!(m.has(Money));
        assert_eq!(m.get(Money), Some(10));
        // overwrite keeps single entry
        m.set(Money, 20);
        assert_eq!(m.get(Money), Some(20));
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn insertion_order_preserved() {
        let m = rmap(&[(Stone, 1), (Money, 2), (Metal, 3)]);
        let keys: Vec<BasicResource> = m.iter().map(|(k, _)| k).collect();
        assert_eq!(keys, vec![Stone, Money, Metal]);
    }

    #[test]
    fn merge_adds_shared_and_keeps_unique() {
        // right has money+wood, left has money+metal -> money sums, others kept
        let left = rmap(&[(Money, 100), (Metal, 5)]);
        let right = rmap(&[(Money, 1), (Wood, 7)]);
        let merged = merge_resource_maps(&left, &right);
        assert_eq!(merged.get(Money), Some(101));
        assert_eq!(merged.get(Wood), Some(7));
        assert_eq!(merged.get(Metal), Some(5));
        // result starts from `right`, so money & wood come first (right order),
        // then metal (new from left).
        let keys: Vec<BasicResource> = merged.iter().map(|(k, _)| k).collect();
        assert_eq!(keys, vec![Money, Wood, Metal]);
    }

    #[test]
    fn reverse_negates() {
        let m = rmap(&[(Money, 5), (Wood, -3)]);
        let r = reverse_resource_map(&m);
        assert_eq!(r.get(Money), Some(-5));
        assert_eq!(r.get(Wood), Some(3));
    }

    #[test]
    fn negatives_and_positives() {
        let m = rmap(&[(Money, 5), (Wood, -3), (Stone, 0)]);
        let neg = get_negatives_map(&m);
        assert_eq!(neg.get(Money), Some(0));
        assert_eq!(neg.get(Wood), Some(-3));
        assert_eq!(neg.get(Stone), Some(0));
        let pos = get_positives_map(&m);
        assert_eq!(pos.get(Money), Some(5));
        assert_eq!(pos.get(Wood), Some(0));
        // 0 counts as positive (>= 0)
        assert_eq!(pos.get(Stone), Some(0));
    }

    #[test]
    fn starting_resources_locked() {
        let s = starting_resources();
        assert_eq!(s.get(Money), Some(400));
        assert_eq!(s.get(Wood), Some(200));
        assert_eq!(s.get(Stone), Some(100));
        assert_eq!(s.get(Metal), Some(25));
    }

    #[test]
    fn rebalanced_industry_values_locked() {
        // These are the deliberate divergences from C++ — guard against drift.
        assert_eq!(nuclearpp_build_cost().get(Money), Some(-2000));
        assert_eq!(nuclearpp_production().get(Money), Some(160));
        assert_eq!(hepp_production().get(Money), Some(80));
        assert_eq!(mine_production().get(Metal), Some(20));
        assert_eq!(mine_build_cost().get(Stone), Some(200));
    }
}
