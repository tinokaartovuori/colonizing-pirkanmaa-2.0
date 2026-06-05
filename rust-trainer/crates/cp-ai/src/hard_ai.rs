//! Faithful Rust port of `src/managers/ai.ts` — the HELD-OUT hard heuristic.
//!
//! This is NOT on the parity / policy-net path. It is a standalone benchmark
//! opponent: a behaviourally-faithful port of the TypeScript `AiController`
//! (the ~1300-line bounded heuristic that drives a CPU through HQ placement and
//! a full turn). It mirrors the TS decision logic, ordering, affordability rules
//! and target selection. It does NOT need to be bit-for-bit; it must be
//! strategically equivalent — same priorities and same strength.
//!
//! It drives a `cp_sim::Game` for the *current* seat, reusing the same engine
//! primitives a human/the NN controller would (`ai_build_building`,
//! `ai_buy_and_place_unit`, `ai_move_unit`). All of `get_available_tiles`,
//! `free_unit_amount`, etc. operate on `current_player()`, so HardAi must be
//! called only when its seat is current (the harness ensures this).
//!
//! The TS turn is a generator that yields per action; headless we run straight
//! through. The action budget is decremented exactly as the TS `doAction` does:
//! only on a *successful* action.

use cp_sim::resources::{
    self, basic_worker_cost, expert_cost, soldier_cost, BasicResource, ResourceMap,
};
use cp_sim::{BuildingType, Game, PlayerId, TileId, TileType, UnitId, UnitType};

/// Per-difficulty parameters (`PARAMS` in ai.ts). Only `hard` is used as the
/// benchmark, but `easy`/`medium` are provided for completeness/parity of the
/// strategy surface.
#[derive(Debug, Clone, Copy)]
pub struct AiParams {
    pub reserve: i64,
    pub max_actions: i64,
    pub experts: bool,
    pub military: bool,
    pub garrison: i64,
    pub expand: i64,
    pub attack: bool,
    pub nuclear: bool,
    pub max_outposts: i64,
    pub strike_force: i64,
    pub assaults_per_turn: i64,
    pub warmonger: bool,
    /// EXPERIMENTAL (non-shipped, ceiling probe only): when true, the `attack`
    /// phase orders targets by `spatial::offensive_cut_value` (the fraction of
    /// enemy territory that disconnects if the tile is taken) instead of the
    /// shipped "HQ-first, then fewest-defenders" order. Default false → byte-
    /// identical to the ported TS bot (parity-safe).
    pub cut_priority: bool,
    /// Use the Strange Device WIN strategy: when clearly leading, build the Device
    /// to force a decisive finish. Gates only the BUILD decision — the counterplay
    /// (massing soldiers + assaulting an enemy Device on sight) is always on, so a
    /// `device: false` AI still races to crack one. Mirrors `AiParams.device`.
    pub device: bool,
}

/// `PARAMS.hard` — the held-out benchmark difficulty.
pub const HARD_PARAMS: AiParams = AiParams {
    reserve: 140,
    max_actions: 28,
    experts: true,
    military: true,
    garrison: 3,
    expand: 5,
    attack: true,
    nuclear: true,
    max_outposts: 5,
    strike_force: 7,
    assaults_per_turn: 7,
    warmonger: false,
    cut_priority: false,
    device: true,
};

/// `PARAMS.medium`.
pub const MEDIUM_PARAMS: AiParams = AiParams {
    reserve: 110,
    max_actions: 14,
    experts: true,
    military: true,
    garrison: 2,
    expand: 3,
    attack: true,
    nuclear: false,
    max_outposts: 2,
    strike_force: 3,
    assaults_per_turn: 4,
    warmonger: false,
    cut_priority: false,
    device: true,
};

/// `PARAMS.easy`.
pub const EASY_PARAMS: AiParams = AiParams {
    reserve: 80,
    max_actions: 5,
    experts: false,
    military: true,
    garrison: 1,
    expand: 2,
    attack: true,
    nuclear: false,
    max_outposts: 1,
    strike_force: 1,
    assaults_per_turn: 1,
    warmonger: false,
    cut_priority: false,
    device: false,
};

/// Scripted DEVICE-RUSHER strategy opponent (Lever C, TRAINING-ONLY). A HardAi
/// variant biased to bank a minimal economy and build the Strange Device as early
/// as the gate allows (the engine gate already enforces `rounds >= 18`, not-losing,
/// affordable — `build_strange_device`), then DEFEND it (the existing `military`
/// phase already rings the Device's approaches; `attack` is left ON so it can still
/// crack an enemy Device, but with a tiny strike force it does not go on the
/// offensive). Faithful to GAME-MECHANICS §6: the device tile holds zero defenders
/// and the build halves the soldier cap, so this opponent is a *defensively fragile*
/// rush the learner should be able to punish if it over-extends — and must learn to
/// out-race / raid otherwise. This is NOT a new agent or rule: it is HardAi with
/// skewed `AiParams`, so it stays legal and parity-irrelevant.
pub const DEVICE_RUSH_PARAMS: AiParams = AiParams {
    reserve: 120,            // bank toward the Device cost rather than over-spending
    max_actions: 24,
    experts: true,           // efficient economy to afford the Device fast
    military: true,          // keep soldiers so it can DEFEND its device
    garrison: 2,
    expand: 3,               // a small economy, not a sprawling empire
    attack: true,            // counterplay stays on (crack an enemy device on sight)
    nuclear: false,          // the Device, not Nuclear, is the win plan
    max_outposts: 1,         // exactly the +3-cap Outpost the Device precursor needs
    strike_force: 1,         // turtle: minimal offensive army
    assaults_per_turn: 1,
    warmonger: false,
    cut_priority: false,
    device: true,            // THE point: race the Strange Device
};

/// Scripted ARMY-RUSHER strategy opponent (Lever C, TRAINING-ONLY). A HardAi variant
/// biased to max soldier capacity (Outposts give +3 cap each — GAME-MECHANICS §5),
/// expand, hire soldiers and assault early with soldier-superiority (the `attack`
/// phase only targets non-Outpost tiles where it out-numbers the defender — §3).
/// Faithful to the mechanics: no new actions, just HardAi with priorities skewed
/// toward military capacity + aggression and the Device turned OFF (it commits to the
/// army win, not the Device race). The learner must build defensive capacity (the
/// exact capacity-blindness gap) to survive it.
pub const ARMY_RUSH_PARAMS: AiParams = AiParams {
    reserve: 100,
    max_actions: 30,
    experts: true,
    military: true,
    garrison: 3,
    expand: 6,               // grab tiles to feed Outposts + frontier pressure
    attack: true,
    nuclear: false,
    max_outposts: 7,         // MAX soldier cap (each Outpost = +3 cap)
    strike_force: 10,        // field a big offensive army
    assaults_per_turn: 10,   // press the assault every turn
    warmonger: true,         // gear up for war as soon as any enemy exists
    cut_priority: false,
    device: false,           // commit to the army win, not the Device
};

/// `AiController.STAFF_RESERVE`.
const STAFF_RESERVE: i64 = 20;

/// The heuristic CPU controller. Stateless except for the per-turn action
/// budget, which is reset at the start of each `plan_turn`.
pub struct HardAi {
    params: AiParams,
    budget: i64,
}

impl HardAi {
    pub fn new(params: AiParams) -> Self {
        HardAi { params, budget: 0 }
    }

    pub fn hard() -> Self {
        HardAi::new(HARD_PARAMS)
    }

    /// HARD bot with the experimental cut-priority attack ordering (ceiling
    /// probe only — see `AiParams::cut_priority`).
    pub fn hard_cut() -> Self {
        let mut p = HARD_PARAMS;
        p.cut_priority = true;
        HardAi::new(p)
    }

    /// Scripted DEVICE-RUSHER strategy opponent (Lever C, training-only).
    pub fn device_rush() -> Self {
        HardAi::new(DEVICE_RUSH_PARAMS)
    }

    /// Scripted ARMY-RUSHER strategy opponent (Lever C, training-only).
    pub fn army_rush() -> Self {
        HardAi::new(ARMY_RUSH_PARAMS)
    }

    // --- first round --------------------------------------------------------

    /// `placeHeadquarters` — choose and claim a starting tile. Identical scoring
    /// to the NN controller's port and the TS heuristic.
    pub fn place_headquarters(&self, g: &mut Game, player: PlayerId) {
        debug_assert_eq!(g.current_player(), player);
        // Candidates must be BUILDABLE: unowned AND empty (first-round HQ placement
        // is refused on a tile that already holds a building, e.g. an unowned
        // Mikontalo — picking one left the player with 0 tiles → instant loss).
        // Prefer grassland, then any non-river land, then any tile.
        let mut candidates: Vec<TileId> = g
            .get_tiles()
            .iter()
            .filter(|t| t.tile_type == TileType::Grassland && t.owner.is_none() && t.building.is_none())
            .map(|t| t.id)
            .collect();
        if candidates.is_empty() {
            candidates = g.get_tiles().iter().filter(|t| t.owner.is_none() && t.building.is_none() && t.tile_type != TileType::River).map(|t| t.id).collect();
        }
        if candidates.is_empty() {
            candidates = g.get_tiles().iter().filter(|t| t.owner.is_none() && t.building.is_none()).map(|t| t.id).collect();
        }
        if candidates.is_empty() {
            return;
        }
        let mut best = candidates[0];
        let mut best_score = f64::NEG_INFINITY;
        for &tid in &candidates {
            let ns = g.neighbour_tiles(tid);
            let free = ns.iter().filter(|&&n| g.tiles[n.0].owner.is_none()).count() as i64;
            let forests = ns
                .iter()
                .filter(|&&n| g.tiles[n.0].tile_type == TileType::Forest)
                .count() as i64;
            let mountains = ns
                .iter()
                .filter(|&&n| g.tiles[n.0].tile_type == TileType::Mountain)
                .count() as i64;
            let grass = ns
                .iter()
                .filter(|&&n| g.tiles[n.0].tile_type == TileType::Grassland)
                .count() as i64;
            let distance = self.distance_to_nearest_owned(g, tid).min(8);
            let score = (free * 3 + grass * 2 + forests * 2 + mountains * 3 + distance) as f64;
            if score > best_score {
                best_score = score;
                best = tid;
            }
        }
        g.first_round_actions(best);
    }

    fn distance_to_nearest_owned(&self, g: &Game, tid: TileId) -> i64 {
        let mut min = i64::MAX;
        let (cx, cy) = (g.tiles[tid.0].x, g.tiles[tid.0].y);
        for other in g.get_tiles() {
            if other.owner.is_none() {
                continue;
            }
            let d = (other.x - cx).abs() as i64 + (other.y - cy).abs() as i64;
            if d < min {
                min = d;
            }
        }
        if min == i64::MAX {
            99
        } else {
            min
        }
    }

    // --- turn ---------------------------------------------------------------

    /// `planTurn` (drained synchronously, as `playTurn` does). Wrapped so a
    /// panic inside (the TS `try { ... } catch {}`) can never crash the game.
    pub fn plan_turn(&mut self, g: &mut Game, player: PlayerId) {
        self.budget = self.params.max_actions;
        // PANIC MODE: an enemy who builds a Device halves their own soldier cap — that
        // is the window to strike. Go all-in for the turn: spend the reserve, press
        // every front, and field a real army (the Device is attacked first in `attack`).
        // Restored after the turn (HardAi is reused across turns/games). The per-buy
        // upkeep guard in `garrison` still prevents literal bankruptcy.
        let saved = self.params;
        if self.enemy_has_device(g, player) {
            self.params.reserve = (self.params.reserve / 4).max(40);
            self.params.assaults_per_turn = self.params.assaults_per_turn.max(12);
            self.budget += 12;
        }
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.run_turn(g, player);
        }));
        let _ = r; // swallow any panic, matching the TS catch-all.
        self.params = saved;
    }

    fn run_turn(&mut self, g: &mut Game, player: PlayerId) {
        self.ensure_wood_income(g, player);
        self.staff_buildings(g, player);
        self.secure_wood(g, player);

        let saving_for_mine = self.staffed_farm_count(g, player) >= 2
            && self.owned_tiles(g, player).iter().any(|&t| {
                g.tiles[t.0].tile_type == TileType::Mountain && g.tiles[t.0].building.is_none()
            })
            && self.wood(g, player) < 270;

        if !saving_for_mine {
            self.build_farms(g, player);
            self.staff_buildings(g, player);
        }
        self.build_mines(g, player);
        self.staff_buildings(g, player);
        self.boost_mines(g, player);
        self.build_power_plants(g, player);
        self.invest_nuclear(g, player);
        self.build_outposts(g, player);
        self.raise_unit_cap(g, player);
        self.expand(g, player);
        self.build_strange_device(g, player); // when leading: race the Device to a decisive win
        self.military(g, player);
        self.attack(g, player);
        self.stack_producers(g, player);
        self.fill_spare_slots(g, player);
    }

    /// `doAction` — run `fn`; on success count it against the budget. Returns
    /// the success bool. Refuses when the budget is exhausted.
    fn do_action(&mut self, ok: bool) -> bool {
        // The TS guards `budget <= 0` BEFORE calling fn; callers here mirror
        // that by checking `self.budget > 0` in their loops. We still guard.
        if self.budget <= 0 {
            return false;
        }
        if ok {
            self.budget -= 1;
        }
        ok
    }

    // --- resource helpers ---------------------------------------------------

    fn res(&self, g: &Game, p: PlayerId, r: BasicResource) -> i64 {
        g.players[p.0].resources.get(r).unwrap_or(0)
    }
    fn money(&self, g: &Game, p: PlayerId) -> i64 {
        self.res(g, p, BasicResource::Money)
    }
    fn wood(&self, g: &Game, p: PlayerId) -> i64 {
        self.res(g, p, BasicResource::Wood)
    }
    fn stone(&self, g: &Game, p: PlayerId) -> i64 {
        self.res(g, p, BasicResource::Stone)
    }
    fn metal(&self, g: &Game, p: PlayerId) -> i64 {
        self.res(g, p, BasicResource::Metal)
    }

    fn owned_tiles(&self, g: &Game, p: PlayerId) -> Vec<TileId> {
        g.owned_tiles(p)
    }
    fn building_of(&self, g: &Game, tid: TileId) -> Option<BuildingType> {
        g.tiles[tid.0].building.as_ref().map(|b| b.kind)
    }
    fn has_type(&self, g: &Game, tid: TileId, kind: UnitType) -> bool {
        g.tile_units(tid).iter().any(|&u| g.units[u.0].kind == kind)
    }
    fn workers_on(&self, g: &Game, tid: TileId) -> i64 {
        g.tile_units(tid)
            .iter()
            .filter(|&&u| g.units[u.0].kind == UnitType::BasicWorker)
            .count() as i64
    }

    fn salary_per_round(&self, g: &Game, p: PlayerId) -> f64 {
        (g.current_basic_worker_amount(p) * 5
            + g.current_expert_amount(p) * 25
            + g.current_soldier_amount(p) * 30) as f64
    }

    fn money_drain_per_round(&self, g: &Game, p: PlayerId) -> f64 {
        let mut upkeep = 0.0;
        for t in self.owned_tiles(g, p) {
            match self.building_of(g, t) {
                Some(BuildingType::Village) => upkeep += 10.0,
                Some(BuildingType::Outpost) => upkeep += 50.0,
                _ => {}
            }
        }
        self.salary_per_round(g, p) + upkeep
    }

    fn staffed_farm_count(&self, g: &Game, p: PlayerId) -> i64 {
        self.owned_tiles(g, p)
            .iter()
            .filter(|&&t| {
                self.building_of(g, t) == Some(BuildingType::Farm)
                    && self.has_type(g, t, UnitType::BasicWorker)
            })
            .count() as i64
    }

    fn net_money_per_round(&self, g: &Game, p: PlayerId) -> f64 {
        let mut income = 0.0;
        for tid in self.owned_tiles(g, p) {
            let ty = self.building_of(g, tid);
            let workers = self.workers_on(g, tid);
            let has_expert = self.has_type(g, tid, UnitType::Expert);
            match ty {
                Some(BuildingType::Farm) if workers > 0 => income += 175.0 / 4.0,
                Some(BuildingType::Mine) if workers > 0 => {
                    income += 20.0 * workers as f64 * if has_expert { 2.0 } else { 1.0 }
                }
                Some(BuildingType::Nuclear) if workers > 0 && has_expert => {
                    income += 160.0 * workers as f64
                }
                Some(BuildingType::Hydro) if workers > 0 && has_expert => {
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
        income - self.salary_per_round(g, p)
    }

    fn metal_income_per_round(&self, g: &Game, p: PlayerId) -> f64 {
        let mut metal = 0.0;
        for tid in self.owned_tiles(g, p) {
            if self.building_of(g, tid) != Some(BuildingType::Mine) {
                continue;
            }
            metal += 20.0
                * self.workers_on(g, tid) as f64
                * if self.has_type(g, tid, UnitType::Expert) {
                    2.0
                } else {
                    1.0
                };
        }
        metal
    }

    fn stone_income_per_round(&self, g: &Game, p: PlayerId) -> f64 {
        let mut stone = 0.0;
        for tid in self.owned_tiles(g, p) {
            if self.building_of(g, tid) != Some(BuildingType::Mine) {
                continue;
            }
            stone += 30.0
                * self.workers_on(g, tid) as f64
                * if self.has_type(g, tid, UnitType::Expert) {
                    2.0
                } else {
                    1.0
                };
        }
        stone
    }

    fn can_afford_upkeep(&self, g: &Game, p: PlayerId, salary: f64) -> bool {
        self.net_money_per_round(g, p) - salary >= 0.0
    }

    fn affords(&self, g: &Game, p: PlayerId, cost: &ResourceMap, reserve: i64) -> bool {
        if !g.players[p.0].has_enough_resources(cost) {
            return false;
        }
        let buffer = reserve as f64 + self.money_drain_per_round(g, p) * 5.0;
        (self.money(g, p) + cost.get(BasicResource::Money).unwrap_or(0)) as f64 >= buffer
    }

    #[allow(dead_code)]
    fn affords_income_build(&self, g: &Game, p: PlayerId, cost: &ResourceMap, floor: i64) -> bool {
        if !g.players[p.0].has_enough_resources(cost) {
            return false;
        }
        self.money(g, p) + cost.get(BasicResource::Money).unwrap_or(0) >= floor
    }

    fn affords_farm(&self, g: &Game, p: PlayerId, farm_count: i64) -> bool {
        let cost = resources::farm_build_cost();
        if !g.players[p.0].has_enough_resources(&cost) {
            return false;
        }
        let money_after = self.money(g, p) + cost.get(BasicResource::Money).unwrap_or(0);
        // A farm pays out only every ~4 rounds, so keep enough cash to cover ~4
        // rounds of drain (salary + upkeep) after the build — otherwise the bot
        // spends its last cash on farms/staffing and salary bankrupts it BEFORE the
        // farms produce (the grassland-poor self-bankruptcy bug). Early game drain is
        // tiny, so the bootstrap opening stays unblocked.
        let cushion = self.money_drain_per_round(g, p) * 4.0;
        if farm_count < 3 {
            return money_after as f64 >= 40.0_f64.max(cushion);
        }
        money_after as f64 >= 80.0_f64.max(cushion)
    }

    fn add_worker(&mut self, g: &mut Game, player: PlayerId, tid: TileId) -> bool {
        if g.free_unit_amount(player) <= 0 {
            return false;
        }
        if !self.affords(g, player, &basic_worker_cost(), STAFF_RESERVE) {
            return false;
        }
        g.ai_buy_and_place_unit("BasicWorker", tid)
    }

    fn add_expert(&mut self, g: &mut Game, player: PlayerId, tid: TileId) -> bool {
        if !self.affords(g, player, &expert_cost(), self.params.reserve) {
            return false;
        }
        g.ai_buy_and_place_unit("Expert", tid)
    }

    // --- staffing -----------------------------------------------------------

    fn staff_buildings(&mut self, g: &mut Game, player: PlayerId) {
        for tid in self.owned_tiles(g, player) {
            match self.building_of(g, tid) {
                Some(BuildingType::Farm) => {
                    if self.budget > 0 && !self.has_type(g, tid, UnitType::BasicWorker) {
                        let ok = self.add_worker(g, player, tid);
                        self.do_action(ok);
                    }
                }
                Some(BuildingType::Mine) => self.ensure_worker(g, player, tid),
                Some(BuildingType::Nuclear) | Some(BuildingType::Hydro) => {
                    self.staff_plant(g, player, tid)
                }
                _ => {
                    if g.tiles[tid.0].tile_type == TileType::AbundantForest
                        && !self.has_type(g, tid, UnitType::BasicWorker)
                        && self.budget > 0
                    {
                        let ok = self.add_worker(g, player, tid);
                        self.do_action(ok);
                    }
                }
            }
        }
    }

    fn staff_plant(&mut self, g: &mut Game, player: PlayerId, tid: TileId) {
        if !self.params.experts {
            return;
        }
        let has_expert = |g: &Game| self.has_type(g, tid, UnitType::Expert);
        let workers = |g: &Game| self.workers_on(g, tid);
        if has_expert(g) && workers(g) >= 1 {
            return;
        }
        let reloc = |s: &Self, g: &Game| {
            s.find_idle_on_plain(g, player)
                .or_else(|| s.find_surplus_producer_worker(g, player))
        };
        let need_worker = workers(g) < 1;
        let need_expert = !has_expert(g);
        let slots_needed = (need_expert as i64) + (need_worker as i64);
        let reloc_for_worker = if need_worker && reloc(self, g).is_some() {
            1
        } else {
            0
        };
        if g.free_unit_amount(player) + reloc_for_worker < slots_needed {
            return;
        }
        if need_worker {
            if g.free_unit_amount(player) > 0 {
                if self.budget > 0 {
                    let ok = self.add_worker(g, player, tid);
                    self.do_action(ok);
                }
            } else if let Some((unit, from)) = reloc(self, g) {
                if from != tid && self.budget > 0 {
                    let ok = g.ai_move_unit(unit, from, tid);
                    self.do_action(ok);
                }
            }
        }
        if !self.has_type(g, tid, UnitType::Expert)
            && self.workers_on(g, tid) >= 1
            && g.free_unit_amount(player) > 0
            && self.budget > 0
        {
            let ok = self.add_expert(g, player, tid);
            self.do_action(ok);
        }
    }

    fn can_staff_new_plant(&self, g: &Game, player: PlayerId) -> bool {
        let free = g.free_unit_amount(player);
        if free >= 2 {
            return true;
        }
        free >= 1
            && (self.find_idle_on_plain(g, player).is_some()
                || self.find_surplus_producer_worker(g, player).is_some())
    }

    fn ensure_worker(&mut self, g: &mut Game, player: PlayerId, tid: TileId) {
        if self.has_type(g, tid, UnitType::BasicWorker) {
            return;
        }
        if g.free_unit_amount(player) > 0 {
            if self.budget > 0 {
                let ok = self.add_worker(g, player, tid);
                self.do_action(ok);
            }
            return;
        }
        let spare = self
            .find_idle_on_plain(g, player)
            .or_else(|| self.find_spare_worker(g, player, tid));
        if let Some((unit, from)) = spare {
            if from != tid && self.budget > 0 {
                let ok = g.ai_move_unit(unit, from, tid);
                self.do_action(ok);
            }
        }
    }

    fn find_spare_worker(&self, g: &Game, player: PlayerId, exclude: TileId) -> Option<(UnitId, TileId)> {
        for tid in self.owned_tiles(g, player) {
            if tid == exclude {
                continue;
            }
            if matches!(
                self.building_of(g, tid),
                Some(BuildingType::Farm)
                    | Some(BuildingType::Mine)
                    | Some(BuildingType::Nuclear)
                    | Some(BuildingType::Hydro)
            ) {
                continue;
            }
            if let Some(u) = self.first_worker(g, tid) {
                return Some((u, tid));
            }
        }
        None
    }

    fn first_worker(&self, g: &Game, tid: TileId) -> Option<UnitId> {
        g.tile_units(tid)
            .iter()
            .copied()
            .find(|&u| g.units[u.0].kind == UnitType::BasicWorker)
    }

    // --- building -----------------------------------------------------------

    fn empty_grassland(&self, g: &Game, player: PlayerId) -> Vec<TileId> {
        self.owned_tiles(g, player)
            .into_iter()
            .filter(|&t| {
                g.tiles[t.0].tile_type == TileType::Grassland
                    && g.tiles[t.0].building.is_none()
                    && g.buildable_buildings(t).contains(&"Farm")
            })
            .collect()
    }

    fn build_mines(&mut self, g: &mut Game, player: PlayerId) {
        if self.wood(g, player) < 300 {
            return;
        }
        let mines = self
            .owned_tiles(g, player)
            .iter()
            .filter(|&&t| self.building_of(g, t) == Some(BuildingType::Mine))
            .count() as i64;
        let villages = self
            .owned_tiles(g, player)
            .iter()
            .filter(|&&t| self.building_of(g, t) == Some(BuildingType::Village))
            .count() as i64;
        let max_mines = 1 + villages;
        if mines >= max_mines {
            return;
        }
        let mountains: Vec<TileId> = self
            .owned_tiles(g, player)
            .into_iter()
            .filter(|&t| {
                g.tiles[t.0].tile_type == TileType::Mountain && g.tiles[t.0].building.is_none()
            })
            .collect();
        for m in mountains {
            if self.affords(g, player, &resources::mine_build_cost(), self.params.reserve)
                && self.has_wood_buffer(g, player, &resources::mine_build_cost())
                && self.budget > 0
            {
                let ok = g.ai_build_building("Mine", m);
                if self.do_action(ok) {
                    return; // one per turn
                }
            }
        }
    }

    fn boost_mines(&mut self, g: &mut Game, player: PlayerId) {
        if !self.params.experts {
            return;
        }
        for tid in self.owned_tiles(g, player) {
            if self.building_of(g, tid) != Some(BuildingType::Mine) {
                continue;
            }
            if !self.has_type(g, tid, UnitType::BasicWorker)
                || self.has_type(g, tid, UnitType::Expert)
            {
                continue;
            }
            if g.free_unit_amount(player) <= 0 {
                continue;
            }
            if self.budget > 0 {
                let ok = self.add_expert(g, player, tid);
                self.do_action(ok);
            }
        }
    }

    fn build_farms(&mut self, g: &mut Game, player: PlayerId) {
        let spots = self.empty_grassland(g, player);
        let mut farm_count = self
            .owned_tiles(g, player)
            .iter()
            .filter(|&&t| self.building_of(g, t) == Some(BuildingType::Farm))
            .count() as i64;
        let mine_count = self
            .owned_tiles(g, player)
            .iter()
            .filter(|&&t| self.building_of(g, t) == Some(BuildingType::Mine))
            .count() as i64;
        let max_farms = 1i64.max(g.max_unit_amount(player) - 2 - mine_count);

        // First: grassland already holding an idle worker (free staffing).
        let with_worker: Vec<TileId> = spots
            .iter()
            .copied()
            .filter(|&t| self.has_type(g, t, UnitType::BasicWorker))
            .collect();
        for gtid in with_worker {
            if farm_count >= max_farms {
                break;
            }
            if self.affords_farm(g, player, farm_count)
                && self.has_wood_buffer(g, player, &resources::farm_build_cost())
                && self.budget > 0
            {
                let ok = g.ai_build_building("Farm", gtid);
                if self.do_action(ok) {
                    farm_count += 1;
                }
            }
        }
        // Then empty grasslands, if we have a free slot to staff the new farm.
        let slot_floor = if self.wood(g, player) < 200 { 1 } else { 0 };
        let without_worker: Vec<TileId> = spots
            .iter()
            .copied()
            .filter(|&t| !self.has_type(g, t, UnitType::BasicWorker))
            .collect();
        for gtid in without_worker {
            if farm_count >= max_farms {
                break;
            }
            if g.free_unit_amount(player) <= slot_floor {
                break;
            }
            if self.affords_farm(g, player, farm_count)
                && self.has_wood_buffer(g, player, &resources::farm_build_cost())
                && self.budget > 0
            {
                let ok = g.ai_build_building("Farm", gtid);
                if self.do_action(ok) {
                    farm_count += 1;
                }
            }
        }
    }

    fn build_power_plants(&mut self, g: &mut Game, player: PlayerId) {
        if !self.params.experts {
            return;
        }
        if self.net_money_per_round(g, player) <= 0.0 {
            return;
        }
        let hydros: Vec<TileId> = self
            .owned_tiles(g, player)
            .into_iter()
            .filter(|&t| self.building_of(g, t) == Some(BuildingType::Hydro))
            .collect();
        if hydros.iter().any(|&t| {
            !self.has_type(g, t, UnitType::Expert) || !self.has_type(g, t, UnitType::BasicWorker)
        }) {
            return;
        }
        if !self.can_staff_new_plant(g, player) {
            return;
        }
        let rivers: Vec<TileId> = self
            .owned_tiles(g, player)
            .into_iter()
            .filter(|&t| {
                g.tiles[t.0].tile_type == TileType::River
                    && g.tiles[t.0].building.is_none()
                    && g.buildable_buildings(t)
                        .contains(&"Hydroelectric Power Plant")
            })
            .collect();
        for r in rivers {
            if self.affords(
                g,
                player,
                &resources::hepp_build_cost(),
                self.params.reserve.min(80),
            ) && self.has_wood_buffer(g, player, &resources::hepp_build_cost())
                && self.budget > 0
            {
                let ok = g.ai_build_building("Hydroelectric Power Plant", r);
                if self.do_action(ok) {
                    break;
                }
            }
        }
    }

    fn invest_nuclear(&mut self, g: &mut Game, player: PlayerId) {
        if !self.params.nuclear || !self.params.experts {
            return;
        }
        if self.money(g, player) <= 2400 {
            return;
        }
        let nukes = |s: &Self, g: &Game| -> Vec<TileId> {
            s.owned_tiles(g, player)
                .into_iter()
                .filter(|&t| s.building_of(g, t) == Some(BuildingType::Nuclear))
                .collect()
        };
        // 1. Staff existing plants first.
        for plant in nukes(self, g) {
            self.staff_nuclear(g, player, plant);
        }
        let fully_staffed = |s: &Self, g: &Game, t: TileId| {
            s.has_type(g, t, UnitType::Expert) && s.workers_on(g, t) >= 1
        };
        let want_count = 1 + ((self.money(g, player) - 2400) / 3000);
        let cur = nukes(self, g);
        if cur.len() as i64 >= want_count || !cur.iter().all(|&t| fully_staffed(self, g, t)) {
            return;
        }
        let empty_grass: Vec<TileId> = self
            .empty_grassland(g, player)
            .into_iter()
            .filter(|&t| !self.has_type(g, t, UnitType::BasicWorker))
            .collect();
        if g.free_unit_amount(player) < 1
            && !(self.can_raise_cap(g, player) && empty_grass.len() >= 2)
        {
            return;
        }
        if let Some(&spot) = empty_grass.first() {
            if self.affords(g, player, &resources::nuclearpp_build_cost(), self.params.reserve)
                && self.has_wood_buffer(g, player, &resources::nuclearpp_build_cost())
                && self.budget > 0
            {
                let ok = g.ai_build_building("Nuclear Power Plant", spot);
                if self.do_action(ok) {
                    self.staff_nuclear(g, player, spot);
                }
            }
        }
    }

    fn staff_nuclear(&mut self, g: &mut Game, player: PlayerId, plant: TileId) {
        if !self.has_type(g, plant, UnitType::Expert) {
            if g.free_unit_amount(player) < 1 {
                self.raise_unit_cap(g, player);
            }
            if g.free_unit_amount(player) > 0 && self.budget > 0 {
                let ok = self.add_expert(g, player, plant);
                self.do_action(ok);
            }
        }
        while self.has_type(g, plant, UnitType::Expert)
            && self.workers_on(g, plant) < 2
            && g.tiles[plant.0].has_space_for_units()
            && self.budget > 0
        {
            if g.free_unit_amount(player) > 0 {
                let ok = self.add_worker(g, player, plant);
                if !self.do_action(ok) {
                    break;
                }
            } else {
                match self.find_expendable_worker(g, player) {
                    Some((unit, from)) if from != plant => {
                        let ok = g.ai_move_unit(unit, from, plant);
                        if !self.do_action(ok) {
                            break;
                        }
                    }
                    _ => break,
                }
            }
        }
    }

    fn can_raise_cap(&self, g: &Game, player: PlayerId) -> bool {
        let villages = self
            .owned_tiles(g, player)
            .iter()
            .filter(|&&t| self.building_of(g, t) == Some(BuildingType::Village))
            .count() as i64;
        if villages >= 5 {
            return false;
        }
        if !self
            .empty_grassland(g, player)
            .iter()
            .any(|&t| !self.has_type(g, t, UnitType::BasicWorker))
        {
            return false;
        }
        self.owned_tiles(g, player).iter().any(|&t| {
            g.tiles[t.0].tile_type == TileType::Forest
                && (g.tiles[t.0].building.is_none() || self.has_type(g, t, UnitType::BasicWorker))
        })
    }

    fn enemy_exists(&self, g: &Game, player: PlayerId) -> bool {
        g.live_players().iter().any(|&p| p != player)
    }

    /// True while an opponent owns a standing Strange Device — we must crack it before
    /// its countdown wins them the game. Always checked (NOT gated on `params.device`),
    /// so even a non-building AI mounts the counterplay.
    fn enemy_has_device(&self, g: &Game, player: PlayerId) -> bool {
        match g.find_strange_device_tile() {
            Some(dt) => {
                let o = g.tiles[dt.0].owner;
                o.is_some() && o != Some(player)
            }
            None => false,
        }
    }

    fn should_militarise(&self, g: &Game, player: PlayerId) -> bool {
        // A standing enemy Device is an existential threat (its countdown wins the
        // game), so gear up for war regardless of the normal trigger.
        if self.enemy_has_device(g, player) {
            return true;
        }
        if self.params.warmonger {
            self.enemy_exists(g, player)
        } else {
            self.has_reachable_enemy(g, player)
        }
    }

    fn has_reachable_enemy(&self, g: &Game, player: PlayerId) -> bool {
        if self.enemy_threat(g, player) > 0 {
            return true;
        }
        g.get_available_tiles().iter().any(|&t| {
            let o = g.tiles[t.0].owner;
            o.is_some() && o != Some(player)
        })
    }

    fn reachable_enemy_max_defenders(&self, g: &Game, player: PlayerId) -> i64 {
        let mut max = 0;
        for t in g.get_available_tiles() {
            let o = g.tiles[t.0].owner;
            if o.is_none() || o == Some(player) {
                continue;
            }
            if self.building_of(g, t) == Some(BuildingType::Outpost) {
                continue;
            }
            let def = g
                .tile_units(t)
                .iter()
                .filter(|&&u| g.units[u.0].kind == UnitType::Soldier)
                .count() as i64;
            if def > max {
                max = def;
            }
        }
        max
    }

    fn military_need(&self, g: &Game, player: PlayerId) -> bool {
        self.enemy_threat(g, player) > 0
            || self.reachable_enemy_max_defenders(g, player) > 0
            || self.enemy_has_device(g, player)
    }

    /// Owned grassland with no building where `what` is buildable, sorted by fewest
    /// enemy-bordering neighbours first (the "interior"/safest tiles). Mirrors the TS
    /// `buildableGrass` helper. `sort_by_key` is stable, matching JS `.sort`.
    fn buildable_grass_for(&self, g: &Game, player: PlayerId, what: &str) -> Vec<TileId> {
        let mut v: Vec<TileId> = self
            .owned_tiles(g, player)
            .into_iter()
            .filter(|&t| {
                g.tiles[t.0].tile_type == TileType::Grassland
                    && g.tiles[t.0].building.is_none()
                    && g.tiles[t.0].units.is_empty() // Device can't be built on an occupied tile
                    && g.buildable_buildings(t).iter().any(|&s| s == what)
            })
            .collect();
        v.sort_by_key(|&t| self.enemy_border_count(g, t, player));
        v
    }

    /// `buildStrangeDevice` — the Device endgame. When we are the clear leader, building
    /// it forces a decisive finish (a countdown win), at the cost of a halved soldier cap.
    /// We commit only when the strategy is enabled, no Device exists, the game has matured,
    /// we are not losing on tiles, we already hold >= 1 Outpost (so the halved cap leaves
    /// real defenders), and the economy can carry the one-time cost. Mirrors ai.ts 785-833.
    fn build_strange_device(&mut self, g: &mut Game, player: PlayerId) {
        if !self.params.device {
            return;
        }
        if g.has_strange_device() {
            return; // one per game — counterplay (attack) handles an enemy's
        }
        if g.get_rounds_played() < 18 {
            return; // let the game develop first
        }
        // Pursue the Device when we are NOT losing on territory.
        let my_tiles = g.get_tile_count_for_player(player);
        let not_losing = g
            .live_players()
            .iter()
            .all(|&p| p == player || g.get_tile_count_for_player(p) <= my_tiles);
        if !not_losing {
            return;
        }
        // Affordability for a TERMINAL play: raw resources + non-negative money net + a
        // small cash floor after the one-time cost (the lighter standard the TS uses; the
        // fat reserve helper almost never fired in a settled late-game economy).
        let device_cost = resources::strange_device_build_cost();
        if !g.players[player.0].has_enough_resources(&device_cost) {
            return;
        }
        if self.net_money_per_round(g, player) < 0.0 {
            return;
        }
        let device_money = device_cost.get(BasicResource::Money).unwrap_or(0);
        if self.money(g, player) + device_money < 150 {
            return;
        }
        let outposts = self
            .owned_tiles(g, player)
            .iter()
            .filter(|&&t| self.building_of(g, t) == Some(BuildingType::Outpost))
            .count() as i64;
        if outposts < 1 {
            // Precursor: lay the gating Outpost now (the Device halves the cap, so an
            // Outpost's +3 keeps the halved cap above zero). The Device follows next turn.
            if let Some(ospot) = self.buildable_grass_for(g, player, "Outpost").first().copied() {
                let outpost_cost = resources::outpost_build_cost();
                let outpost_money = outpost_cost.get(BasicResource::Money).unwrap_or(0);
                let can_afford = g.players[player.0].has_enough_resources(&outpost_cost)
                    && self.net_money_per_round(g, player) - 50.0 >= 0.0
                    && self.money(g, player) + outpost_money >= 100;
                if can_afford && self.budget > 0 {
                    let ok = g.ai_build_building("Outpost", ospot);
                    self.do_action(ok);
                }
            }
            return;
        }
        if let Some(spot) = self
            .buildable_grass_for(g, player, "Strange Device")
            .first()
            .copied()
        {
            if self.budget > 0 {
                let ok = g.ai_build_building("Strange Device", spot);
                self.do_action(ok);
            }
        }
    }

    fn build_outposts(&mut self, g: &mut Game, player: PlayerId) {
        if self.params.max_outposts <= 0 || !self.params.attack {
            return;
        }
        if !self.should_militarise(g, player) {
            return;
        }
        if !self.military_need(g, player) {
            return;
        }
        let outposts = self
            .owned_tiles(g, player)
            .iter()
            .filter(|&&t| self.building_of(g, t) == Some(BuildingType::Outpost))
            .count() as i64;
        if outposts >= self.params.max_outposts {
            return;
        }
        if g.get_tile_count_for_player(player) < 8 {
            return;
        }
        if self.net_money_per_round(g, player) - 50.0 < 10.0 {
            return;
        }
        if self.metal_income_per_round(g, player) - (outposts + 1) as f64 * 15.0 < 0.0 {
            return;
        }
        let buildable: Vec<TileId> = self
            .owned_tiles(g, player)
            .into_iter()
            .filter(|&t| {
                g.tiles[t.0].tile_type == TileType::Grassland
                    && g.tiles[t.0].building.is_none()
                    && g.buildable_buildings(t).contains(&"Outpost")
            })
            .collect();
        let frontline = buildable.iter().copied().find(|&t| {
            self.tile_threatened(g, t, player)
                || g.neighbour_tiles(t)
                    .iter()
                    .any(|&n| g.tiles[n.0].owner.is_some() && g.tiles[n.0].owner != Some(player))
        });
        let spot = frontline.or_else(|| buildable.first().copied());
        if let Some(spot) = spot {
            if self.affords(g, player, &resources::outpost_build_cost(), self.params.reserve.min(100))
                && self.budget > 0
            {
                let ok = g.ai_build_building("Outpost", spot);
                self.do_action(ok);
            }
        }
    }

    fn raise_unit_cap(&mut self, g: &mut Game, player: PlayerId) {
        if g.free_unit_amount(player) > 1 {
            return;
        }
        let spot = match self.empty_grassland(g, player).first().copied() {
            Some(s) => s,
            None => return,
        };
        let villages = self
            .owned_tiles(g, player)
            .iter()
            .filter(|&&t| self.building_of(g, t) == Some(BuildingType::Village))
            .count() as i64;
        let harvesters = self
            .owned_tiles(g, player)
            .iter()
            .filter(|&&t| {
                g.tiles[t.0].tile_type == TileType::Forest
                    && self.has_type(g, t, UnitType::BasicWorker)
            })
            .count() as i64;
        if villages >= 5i64.min(1 + harvesters * 3) {
            return;
        }
        if !self.owned_tiles(g, player).iter().any(|&t| {
            g.tiles[t.0].tile_type == TileType::Forest
                && (g.tiles[t.0].building.is_none() || self.has_type(g, t, UnitType::BasicWorker))
        }) {
            return;
        }
        if self.net_money_per_round(g, player) - 25.0 < 10.0 {
            return;
        }
        let post_upkeep = self.wood_upkeep(g, player) + 10.0;
        if ((self.wood(g, player) - 200) as f64) < 100.0_f64.max(post_upkeep * 5.0) {
            return;
        }
        let stone_upkeep = (villages + 1) as f64 * 10.0;
        if self.stone_income_per_round(g, player) < stone_upkeep
            && ((self.stone(g, player) - 100) as f64) < stone_upkeep * 8.0
        {
            return;
        }
        if self.affords(g, player, &resources::village_build_cost(), self.params.reserve)
            && self.budget > 0
        {
            let ok = g.ai_build_building("Village", spot);
            self.do_action(ok);
        }
    }

    // --- wood ---------------------------------------------------------------

    fn wood_upkeep(&self, g: &Game, p: PlayerId) -> f64 {
        let mut w = 0.0;
        for t in self.owned_tiles(g, p) {
            match self.building_of(g, t) {
                Some(BuildingType::Village) => w += 10.0,
                Some(BuildingType::Bridge) => w += 5.0,
                _ => {}
            }
        }
        w
    }

    fn has_wood_buffer(&self, g: &Game, p: PlayerId, cost: &ResourceMap) -> bool {
        let need = -(cost.get(BasicResource::Wood).unwrap_or(0));
        if need <= 0 {
            return true;
        }
        let upkeep = self.wood_upkeep(g, p);
        let buffer = if upkeep > 0.0 {
            100.0_f64.max(upkeep * 5.0)
        } else {
            0.0
        };
        (self.wood(g, p) - need) as f64 >= buffer
    }

    fn find_expendable_worker(&self, g: &Game, player: PlayerId) -> Option<(UnitId, TileId)> {
        if let Some(idle) = self.find_idle_on_plain(g, player) {
            return Some(idle);
        }
        if let Some(surplus) = self.find_surplus_producer_worker(g, player) {
            return Some(surplus);
        }
        let farms: Vec<TileId> = self
            .owned_tiles(g, player)
            .into_iter()
            .filter(|&t| {
                self.building_of(g, t) == Some(BuildingType::Farm)
                    && self.has_type(g, t, UnitType::BasicWorker)
            })
            .collect();
        if farms.len() >= 2 {
            let tid = farms[farms.len() - 1];
            if let Some(u) = self.first_worker(g, tid) {
                return Some((u, tid));
            }
        }
        None
    }

    fn ensure_wood_income(&mut self, g: &mut Game, player: PlayerId) {
        let upkeep = self.wood_upkeep(g, player);
        if upkeep <= 0.0 {
            return;
        }
        let harvesters = |s: &Self, g: &Game| -> i64 {
            s.owned_tiles(g, player)
                .iter()
                .filter(|&&t| {
                    g.tiles[t.0].tile_type == TileType::Forest
                        && g.tiles[t.0].building.is_none()
                        && s.has_type(g, t, UnitType::BasicWorker)
                })
                .count() as i64
        };
        let mut need = 1i64.max((upkeep / 40.0).ceil() as i64);
        if (self.wood(g, player) as f64) < upkeep * 4.0 {
            need += 1;
        }
        while harvesters(self, g) < need && self.budget > 0 {
            let f = self.owned_tiles(g, player).into_iter().find(|&t| {
                g.tiles[t.0].tile_type == TileType::Forest
                    && g.tiles[t.0].building.is_none()
                    && g.tiles[t.0].has_space_for_units()
                    && !self.has_type(g, t, UnitType::BasicWorker)
            });
            let f = match f {
                Some(t) => t,
                None => break,
            };
            let mut did = false;
            if g.free_unit_amount(player) > 0
                && self.affords(g, player, &basic_worker_cost(), STAFF_RESERVE)
            {
                let ok = self.add_worker(g, player, f);
                did = self.do_action(ok);
            } else if let Some((unit, from)) = self.find_expendable_worker(g, player) {
                if from != f {
                    let ok = g.ai_move_unit(unit, from, f);
                    did = self.do_action(ok);
                }
            }
            if !did {
                break;
            }
        }
    }

    fn anticipated_wood_need(&self, g: &Game, player: PlayerId) -> i64 {
        let mountains_no_mine = self
            .owned_tiles(g, player)
            .iter()
            .filter(|&&t| {
                g.tiles[t.0].tile_type == TileType::Mountain && g.tiles[t.0].building.is_none()
            })
            .count() as i64;
        let empty_grass = self.empty_grassland(g, player).len() as i64;
        mountains_no_mine * 250 + empty_grass.min(4) * 100
    }

    fn secure_wood(&mut self, g: &mut Game, player: PlayerId) {
        let stock_target = 700i64.min(150i64.max(self.anticipated_wood_need(g, player)));
        if self.wood(g, player) >= stock_target + 100 {
            return;
        }
        let staffed = |s: &Self, g: &Game| -> i64 {
            s.owned_tiles(g, player)
                .iter()
                .filter(|&&t| {
                    g.tiles[t.0].tile_type == TileType::Forest
                        && g.tiles[t.0].building.is_none()
                        && s.has_type(g, t, UnitType::BasicWorker)
                })
                .count() as i64
        };
        let target = if self.wood(g, player) < stock_target
            && self.anticipated_wood_need(g, player) > 200
            && g.max_unit_amount(player) > 6
        {
            2
        } else {
            1
        };
        while staffed(self, g) < target && self.budget > 0 {
            let f = self.owned_tiles(g, player).into_iter().find(|&t| {
                g.tiles[t.0].tile_type == TileType::Forest
                    && g.tiles[t.0].building.is_none()
                    && g.tiles[t.0].has_space_for_units()
                    && !self.has_type(g, t, UnitType::BasicWorker)
            });
            let f = match f {
                Some(t) => t,
                None => break,
            };
            let mut did = false;
            if g.free_unit_amount(player) > 0 && self.can_afford_upkeep(g, player, 5.0) {
                let ok = self.add_worker(g, player, f);
                did = self.do_action(ok);
            } else if let Some((unit, from)) = self.find_idle_on_plain(g, player) {
                let ok = g.ai_move_unit(unit, from, f);
                did = self.do_action(ok);
            }
            if !did {
                break;
            }
        }
    }

    fn find_idle_on_plain(&self, g: &Game, player: PlayerId) -> Option<(UnitId, TileId)> {
        for tid in self.owned_tiles(g, player) {
            let ty = g.tiles[tid.0].tile_type;
            if g.tiles[tid.0].building.is_some()
                || ty == TileType::Forest
                || ty == TileType::AbundantForest
            {
                continue;
            }
            if let Some(u) = self.first_worker(g, tid) {
                return Some((u, tid));
            }
        }
        None
    }

    // --- expansion ----------------------------------------------------------

    fn claim_value(&self, g: &Game, tid: TileId) -> i64 {
        if let Some(b) = self.building_of(g, tid) {
            if b == BuildingType::Mikontalo {
                return 6;
            }
        }
        match g.tiles[tid.0].tile_type {
            TileType::Mountain => 5,
            TileType::Grassland => 4,
            TileType::Forest => 3,
            TileType::AbundantForest => 2,
            TileType::River => {
                if g.buildable_buildings(tid)
                    .contains(&"Hydroelectric Power Plant")
                {
                    4
                } else {
                    1
                }
            }
        }
    }

    fn expand(&mut self, g: &mut Game, player: PlayerId) {
        if self.params.expand <= 0 {
            return;
        }
        let mut claimed = 0;
        while claimed < self.params.expand && self.budget > 0 {
            let mut neutral: Vec<TileId> = g
                .get_available_tiles()
                .into_iter()
                .filter(|&t| {
                    g.tiles[t.0].owner.is_none()
                        && g.tiles[t.0].has_space_for_units()
                        && !self.tile_threatened(g, t, player)
                        && !g
                            .tile_conquering_units(t)
                            .iter()
                            .any(|&u| g.units[u.0].owner == Some(player))
                })
                .collect();
            // sort by claim value descending (stable, matching JS Array.sort
            // which is stable in V8 for the engine's ordering).
            neutral.sort_by(|&a, &b| self.claim_value(g, b).cmp(&self.claim_value(g, a)));
            if neutral.is_empty() {
                return;
            }
            let tile = neutral[0];
            let mut did = false;
            // 1. Leap-frog a genuinely idle worker.
            if let Some((unit, from)) = self.find_idle_worker(g, player) {
                if from != tile {
                    let ok = g.ai_move_unit(unit, from, tile);
                    did = self.do_action(ok);
                }
            }
            // 2. Hire a fresh scout into a free slot.
            if !did
                && g.free_unit_amount(player) > 0
                && self.affords(g, player, &basic_worker_cost(), self.params.reserve)
                && self.can_afford_upkeep_cushion(g, player, 5.0)
            {
                if self.budget > 0 {
                    let ok = g.ai_buy_and_place_unit("BasicWorker", tile);
                    did = self.do_action(ok);
                }
            }
            // 3. Peel a surplus producer worker off to scout.
            if !did {
                if let Some((unit, from)) = self.find_surplus_producer_worker(g, player) {
                    if from != tile {
                        let ok = g.ai_move_unit(unit, from, tile);
                        did = self.do_action(ok);
                    }
                }
            }
            if !did {
                return;
            }
            claimed += 1;
        }
    }

    /// `canAffordUpkeep` for the scout-hire path. The TS passes a large cushion
    /// argument that the implementation ignores (it only checks the net), so a
    /// plain `can_afford_upkeep` is faithful.
    fn can_afford_upkeep_cushion(&self, g: &Game, p: PlayerId, salary: f64) -> bool {
        self.can_afford_upkeep(g, p, salary)
    }

    fn find_surplus_producer_worker(&self, g: &Game, player: PlayerId) -> Option<(UnitId, TileId)> {
        for tid in self.owned_tiles(g, player) {
            let stackable = matches!(
                self.building_of(g, tid),
                Some(BuildingType::Mine) | Some(BuildingType::Nuclear) | Some(BuildingType::Hydro)
            );
            if !stackable {
                continue;
            }
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
        if self.wood(g, player) >= 350 {
            for tid in self.owned_tiles(g, player) {
                if g.tiles[tid.0].tile_type != TileType::Forest {
                    continue;
                }
                if let Some(u) = self.first_worker(g, tid) {
                    return Some((u, tid));
                }
            }
        }
        None
    }

    fn find_idle_worker(&self, g: &Game, player: PlayerId) -> Option<(UnitId, TileId)> {
        let needs_wood = self.wood(g, player) < 350
            || self.owned_tiles(g, player).iter().any(|&t| {
                g.tiles[t.0].tile_type == TileType::Mountain && g.tiles[t.0].building.is_none()
            });
        // Pass 1: genuinely idle workers.
        for tid in self.owned_tiles(g, player) {
            let ty = g.tiles[tid.0].tile_type;
            if g.tiles[tid.0].building.is_some()
                || ty == TileType::Forest
                || ty == TileType::AbundantForest
            {
                continue;
            }
            if let Some(u) = self.first_worker(g, tid) {
                return Some((u, tid));
            }
        }
        if needs_wood {
            return None;
        }
        // Pass 2: forest harvesters when wood is no longer needed.
        for tid in self.owned_tiles(g, player) {
            let ty = g.tiles[tid.0].tile_type;
            if ty != TileType::Forest && ty != TileType::AbundantForest {
                continue;
            }
            if let Some(u) = self.first_worker(g, tid) {
                return Some((u, tid));
            }
        }
        None
    }

    // --- spare workers ------------------------------------------------------

    fn stack_producers(&mut self, g: &mut Game, player: PlayerId) {
        let producers = |s: &Self, g: &Game| -> Vec<TileId> {
            s.owned_tiles(g, player)
                .into_iter()
                .filter(|&t| {
                    matches!(
                        s.building_of(g, t),
                        Some(BuildingType::Mine)
                            | Some(BuildingType::Nuclear)
                            | Some(BuildingType::Hydro)
                    ) && g.tiles[t.0].has_space_for_units()
                })
                .collect()
        };
        while g.free_unit_amount(player) > 0 && self.budget > 0 {
            let tile = match producers(self, g).first().copied() {
                Some(t) => t,
                None => break,
            };
            let want_expert =
                self.params.experts && self.building_of(g, tile) != Some(BuildingType::Hydro);
            if want_expert
                && !self.has_type(g, tile, UnitType::Expert)
                && g.free_unit_amount(player) > 1
            {
                let ok = self.add_expert(g, player, tile);
                if self.do_action(ok) {
                    continue;
                }
            }
            let ok = self.add_worker(g, player, tile);
            if !self.do_action(ok) {
                break;
            }
        }
    }

    fn fill_spare_slots(&mut self, g: &mut Game, player: PlayerId) {
        let forests = |s: &Self, g: &Game| -> Vec<TileId> {
            s.owned_tiles(g, player)
                .into_iter()
                .filter(|&t| {
                    g.tiles[t.0].tile_type == TileType::Forest
                        && g.tiles[t.0].building.is_none()
                        && g.tiles[t.0].has_space_for_units()
                })
                .collect()
        };
        while g.free_unit_amount(player) > 0
            && self.budget > 0
            && self.can_afford_upkeep(g, player, 5.0)
        {
            let f = match forests(self, g).first().copied() {
                Some(t) => t,
                None => break,
            };
            let ok = self.add_worker(g, player, f);
            if !self.do_action(ok) {
                break;
            }
        }
    }

    // --- military -----------------------------------------------------------

    fn enemy_threat(&self, g: &Game, player: PlayerId) -> i64 {
        let mut threat = 0;
        for tid in self.owned_tiles(g, player) {
            threat += g
                .tile_conquering_units(tid)
                .iter()
                .filter(|&&u| {
                    g.units[u.0].owner != Some(player) && g.units[u.0].kind == UnitType::Soldier
                })
                .count() as i64;
            for n in g.neighbour_tiles(tid) {
                let o = g.tiles[n.0].owner;
                if o.is_some() && o != Some(player) {
                    threat += g
                        .tile_units(n)
                        .iter()
                        .filter(|&&u| g.units[u.0].kind == UnitType::Soldier)
                        .count() as i64;
                }
            }
        }
        threat
    }

    fn tile_threatened(&self, g: &Game, tid: TileId, player: PlayerId) -> bool {
        for n in g.neighbour_tiles(tid) {
            let o = g.tiles[n.0].owner;
            if o.is_some()
                && o != Some(player)
                && g.tile_units(n)
                    .iter()
                    .any(|&u| g.units[u.0].kind == UnitType::Soldier)
            {
                return true;
            }
        }
        false
    }

    fn adjacent_enemy_soldiers(&self, g: &Game, tid: TileId, player: PlayerId) -> i64 {
        let mut n = 0;
        for nb in g.neighbour_tiles(tid) {
            let o = g.tiles[nb.0].owner;
            if o.is_some() && o != Some(player) {
                n += g
                    .tile_units(nb)
                    .iter()
                    .filter(|&&u| g.units[u.0].kind == UnitType::Soldier)
                    .count() as i64;
            }
        }
        n
    }

    fn invaders_on(&self, g: &Game, tid: TileId, player: PlayerId) -> i64 {
        g.tile_conquering_units(tid)
            .iter()
            .filter(|&&u| g.units[u.0].owner != Some(player) && g.units[u.0].kind == UnitType::Soldier)
            .count() as i64
    }

    fn soldiers_on(&self, g: &Game, tid: TileId, player: PlayerId) -> i64 {
        g.tile_units(tid)
            .iter()
            .filter(|&&u| g.units[u.0].owner == Some(player) && g.units[u.0].kind == UnitType::Soldier)
            .count() as i64
    }

    fn enemy_border_count(&self, g: &Game, tid: TileId, player: PlayerId) -> i64 {
        let mut n = 0;
        for nb in g.neighbour_tiles(tid) {
            let o = g.tiles[nb.0].owner;
            if o.is_some() && o != Some(player) {
                n += 1;
            }
        }
        n
    }

    fn find_rear_soldier(&self, g: &Game, player: PlayerId, exclude: TileId) -> Option<(UnitId, TileId)> {
        for tid in self.owned_tiles(g, player) {
            if tid == exclude {
                continue;
            }
            if self.adjacent_enemy_soldiers(g, tid, player) + self.invaders_on(g, tid, player) > 0 {
                continue;
            }
            if self.enemy_border_count(g, tid, player) > 0 {
                continue;
            }
            if let Some(&u) = g
                .tile_units(tid)
                .iter()
                .find(|&&u| g.units[u.0].owner == Some(player) && g.units[u.0].kind == UnitType::Soldier)
            {
                return Some((u, tid));
            }
        }
        None
    }

    fn garrison(&mut self, g: &mut Game, player: PlayerId, tid: TileId, want: i64) {
        while self.soldiers_on(g, tid, player) < want
            && g.tiles[tid.0].has_space_for_units()
            && self.budget > 0
        {
            if let Some((unit, from)) = self.find_rear_soldier(g, player, tid) {
                let ok = g.ai_move_unit(unit, from, tid);
                if !self.do_action(ok) {
                    break;
                }
                continue;
            }
            if g.free_soldier_amount(player) > 0
                && self.metal(g, player) >= 50
                && self.affords(g, player, &soldier_cost(), self.params.reserve)
                && self.can_afford_upkeep(g, player, 30.0)
            {
                let ok = g.ai_buy_and_place_unit("Soldier", tid);
                if !self.do_action(ok) {
                    break;
                }
                continue;
            }
            break;
        }
    }

    fn military(&mut self, g: &mut Game, player: PlayerId) {
        if !self.params.military {
            return;
        }
        let cap = g.max_soldier_amount(player);
        if cap <= 0 {
            return;
        }
        let hq = g.get_hq_tile(player);
        let at_war = self.should_militarise(g, player);

        // 1. DEFENCE.
        struct Defend {
            tile: TileId,
            want: i64,
            pressure: i64,
        }
        let mut defend: Vec<Defend> = Vec::new();
        if let Some(hq) = hq {
            let threat = self.adjacent_enemy_soldiers(g, hq, player) + self.invaders_on(g, hq, player);
            let want = if at_war {
                3i64.min(self.params.garrison.max(threat + 1))
            } else {
                3i64.min(threat + 1)
            };
            if want > 0 {
                defend.push(Defend {
                    tile: hq,
                    want,
                    pressure: threat,
                });
            }
        }
        for tid in self.owned_tiles(g, player) {
            if Some(tid) == hq {
                continue;
            }
            if self.building_of(g, tid) == Some(BuildingType::Outpost) {
                continue;
            }
            let threat = self.adjacent_enemy_soldiers(g, tid, player) + self.invaders_on(g, tid, player);
            if threat > 0 {
                defend.push(Defend {
                    tile: tid,
                    want: 3i64.min(threat + 1),
                    pressure: threat,
                });
            }
        }
        // DEFEND OUR OWN DEVICE: the Device tile itself can hold no units, so it is
        // defended by garrisoning its APPROACHES — the owned tiles next to it — to the
        // cap, so the enemy can't get adjacent and stage a conquering unit on it. These
        // are forced to the top (high synthetic pressure): leaving the Device undefended
        // is an instant loss, so our halved army's first job is to ring it.
        if let Some(dt) = g.find_strange_device_tile() {
            if g.tiles[dt.0].owner == Some(player) {
                for ntid in g.neighbour_tiles(dt) {
                    if g.tiles[ntid.0].owner != Some(player) {
                        continue;
                    }
                    if self.building_of(g, ntid) == Some(BuildingType::Outpost) {
                        continue; // outposts can't hold soldiers anyway / are impregnable
                    }
                    let threat =
                        self.adjacent_enemy_soldiers(g, ntid, player) + self.invaders_on(g, ntid, player);
                    defend.push(Defend {
                        tile: ntid,
                        want: 3,
                        pressure: threat + 100, // outrank ordinary defence — the Device is existential
                    });
                }
            }
        }
        // Reinforce the most-pressed shortfalls first (Device approaches carry a +100
        // synthetic pressure, so they win the tiebreak among max-shortfall tiles).
        defend.sort_by(|a, b| {
            let sa = b.want - self.soldiers_on(g, b.tile, player);
            let sb = a.want - self.soldiers_on(g, a.tile, player);
            sa.cmp(&sb).then(b.pressure.cmp(&a.pressure))
        });
        for d in defend {
            self.garrison(g, player, d.tile, d.want);
        }

        // 2. BORDER GUARD + STRIKE FORCE.
        let hq = match hq {
            Some(h) if at_war => h,
            _ => return,
        };
        let farms = self
            .owned_tiles(g, player)
            .iter()
            .filter(|&&t| self.building_of(g, t) == Some(BuildingType::Farm))
            .count() as i64;
        let aggression = self
            .enemy_threat(g, player)
            .max(self.reachable_enemy_max_defenders(g, player) + 1);
        let force = if self.enemy_has_device(g, player) {
            // PANIC: the enemy is halved NOW — field the biggest army the economy can
            // sustain (per-buy upkeep guards in `garrison` still apply), ignoring the
            // low *visible* defender count (their cap is halved, so it reads as weak).
            cap.min(farms + 3)
        } else {
            cap.min(self.params.garrison + self.params.strike_force.min(aggression + 1))
                .min(farms + 1)
        };

        let mut frontier: Vec<TileId> = self
            .owned_tiles(g, player)
            .into_iter()
            .filter(|&t| {
                self.building_of(g, t) != Some(BuildingType::Outpost)
                    && self.enemy_border_count(g, t, player) > 0
            })
            .collect();
        frontier.sort_by(|&a, &b| {
            self.enemy_border_count(g, b, player)
                .cmp(&self.enemy_border_count(g, a, player))
        });
        for tile in frontier {
            if g.current_soldier_amount(player) >= force {
                break;
            }
            self.garrison(g, player, tile, 1);
        }
        if g.current_soldier_amount(player) < force {
            let room = force - g.current_soldier_amount(player) + self.soldiers_on(g, hq, player);
            self.garrison(g, player, hq, 3i64.min(room));
        }
    }

    // --- offence ------------------------------------------------------------

    fn find_free_soldier(&self, g: &Game, player: PlayerId, exclude: TileId) -> Option<(UnitId, TileId)> {
        for tid in self.owned_tiles(g, player) {
            if tid == exclude {
                continue;
            }
            if let Some(&u) = g
                .tile_units(tid)
                .iter()
                .find(|&&u| g.units[u.0].kind == UnitType::Soldier)
            {
                return Some((u, tid));
            }
        }
        None
    }

    fn attack(&mut self, g: &mut Game, player: PlayerId) {
        if !self.params.attack {
            return;
        }
        let can_buy = self.money(g, player) >= self.params.reserve + 250;
        if self.params.assaults_per_turn <= 1 && !can_buy {
            return;
        }

        struct Target {
            tile: TileId,
            defenders: i64,
            is_device: bool,
            is_hq: bool,
            cut: f64,
        }
        let cut_priority = self.params.cut_priority;
        let mut targets: Vec<Target> = g
            .get_available_tiles()
            .into_iter()
            .filter(|&t| {
                let o = g.tiles[t.0].owner;
                o.is_some() && o != Some(player) && g.tiles[t.0].has_space_for_conquering_units()
            })
            .map(|t| Target {
                tile: t,
                defenders: g
                    .tile_units(t)
                    .iter()
                    .filter(|&&u| g.units[u.0].kind == UnitType::Soldier)
                    .count() as i64,
                // Destroying an enemy Device stops its loss clock — the top priority.
                is_device: self.building_of(g, t) == Some(BuildingType::StrangeDevice),
                is_hq: self.building_of(g, t) == Some(BuildingType::Headquarters),
                // Only the (non-shipped) cut bot pays the BFS cost.
                cut: if cut_priority {
                    crate::spatial::offensive_cut_value(g, player, t)
                } else {
                    0.0
                },
            })
            .filter(|t| {
                self.building_of(g, t.tile) != Some(BuildingType::Outpost) && t.defenders < 3
            })
            .collect();
        if cut_priority {
            // Enemy Device FIRST (its countdown wins them the game), then highest
            // cut-value (fraction of enemy severed), then cheapest.
            targets.sort_by(|a, b| {
                (b.is_device as i64)
                    .cmp(&(a.is_device as i64))
                    .then_with(|| {
                        b.cut
                            .partial_cmp(&a.cut)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .then(a.defenders.cmp(&b.defenders))
            });
        } else {
            // SHIPPED order: Device first (cracking it resets the clock + reopens the
            // slot), then HQ (collapses an opponent via the connectivity rule), then
            // fewest defenders.
            targets.sort_by(|a, b| {
                (b.is_device as i64)
                    .cmp(&(a.is_device as i64))
                    .then((b.is_hq as i64).cmp(&(a.is_hq as i64)))
                    .then(a.defenders.cmp(&b.defenders))
            });
        }

        let max_assaults = 1i64.max(self.params.assaults_per_turn);
        let mut assaults = 0;
        for t in targets {
            if assaults >= max_assaults || self.budget <= 0 {
                break;
            }
            let needed = t.defenders + 1;
            let mut placed = g
                .tile_conquering_units(t.tile)
                .iter()
                .filter(|&&u| {
                    g.units[u.0].owner == Some(player) && g.units[u.0].kind == UnitType::Soldier
                })
                .count() as i64;
            let to_add = needed - placed;
            if to_add <= 0 {
                continue;
            }
            let movable: i64 = self
                .owned_tiles(g, player)
                .iter()
                .map(|&ti| {
                    if ti == t.tile {
                        0
                    } else {
                        g.tile_units(ti)
                            .iter()
                            .filter(|&&u| g.units[u.0].kind == UnitType::Soldier)
                            .count() as i64
                    }
                })
                .sum();
            let buyable = if can_buy {
                g.free_soldier_amount(player)
                    .min(self.metal(g, player) / 50)
                    .min((self.money(g, player) - self.params.reserve) / 200)
            } else {
                0
            };
            if movable + buyable < to_add {
                continue;
            }
            while placed < needed {
                let mut did = false;
                if let Some((unit, from)) = self.find_free_soldier(g, player, t.tile) {
                    let ok = g.ai_move_unit(unit, from, t.tile);
                    did = self.do_action(ok);
                } else if can_buy
                    && g.free_soldier_amount(player) > 0
                    && self.metal(g, player) >= 50
                    && self.affords(g, player, &soldier_cost(), self.params.reserve)
                {
                    let ok = g.ai_buy_and_place_unit("Soldier", t.tile);
                    did = self.do_action(ok);
                }
                if !did {
                    break;
                }
                placed += 1;
            }
            if placed >= needed {
                assaults += 1;
            }
        }
    }
}
