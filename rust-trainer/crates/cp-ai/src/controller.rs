//! Port of `src/ai/nn/controller.ts` — `NeuralAiController`.
//!
//! Two-part turn:
//!   1. a deterministic safety scaffold (`ensure_wood_income`, `staff_income`)
//!      that guarantees solvency, run FIRST;
//!   2. the learned decision loop: build global features, enumerate candidates,
//!      select (argmax at temperature 0), execute, decrement the budget, then
//!      re-staff — until budget exhausted or Pass.
//!
//! The TS uses generators purely for rendering pacing; headless we run straight
//! through. Behaviour (state mutations, ordering) is identical.

use crate::candidates::{self, Intent};
use crate::features::global_features;
use crate::metrics as m;
use crate::policy::{self, Rng};
use crate::safety as s;
use crate::tiers::TierConfig;
use cp_sim::resources::{basic_worker_cost, expert_cost};
use cp_sim::{BuildingType, Game, PlayerId, TileId, TileType, UnitId, UnitType};

use crate::mlp::Genome;

/// One discretionary decision in the NN loop, captured BEFORE the chosen intent
/// executes. The parity exporter / harness consumes these. Mirrors the TS
/// `DecisionTrace`.
#[derive(Debug, Clone)]
pub struct DecisionTrace {
    pub round: i64,
    pub global_vec: Vec<f64>,
    pub candidates: Vec<DecisionCandidate>,
    pub scores: Vec<f64>,
    pub chosen_candidate_index: usize,
    pub chosen_intent: usize,
}

#[derive(Debug, Clone)]
pub struct DecisionCandidate {
    pub intent: usize,
    pub local: Vec<f64>,
    pub label: String,
}

/// `NeuralAiController`. Borrows the genome + config; the RNG is supplied per
/// turn so callers control reproducibility (training/parity pass a fixed seed).
///
/// `search` is OPT-IN: when `None` (the default), `plan_turn` is byte-identical
/// to the TS port and the parity gate stays green. When `Some(SearchConfig)`,
/// the learned-loop's `policy::select_index` call is replaced by
/// `search::select` (test-time MCTS). Everything else — the safety scaffold,
/// execute, retry-on-fail, re-staff, budget loop — is identical.
pub struct NeuralAiController<'a> {
    pub genome: &'a Genome,
    pub cfg: TierConfig,
    pub search: Option<crate::search::SearchConfig>,
    /// Optional learned value net (Stage B). Only consulted when `search` is
    /// `Some` AND its `leaf_eval == LeafEval::Value`. Never touches the policy
    /// genome / parity path.
    pub value_net: Option<&'a crate::value::ValueNet>,
}

/// One recorded self-play decision for AlphaZero training: the per-candidate
/// policy inputs + the MCTS visit-count target `pi`, the global feature vector
/// (for the value target), and which seat decided (to assign the outcome z).
pub struct RecordedDecision {
    pub player: PlayerId,
    pub policy_inputs: Vec<Vec<f64>>,
    pub pi: Vec<f64>,
    /// 36-dim global features (plain value-net target).
    pub global_vec: Vec<f64>,
    /// 41-dim spatial value features (spatial value-net target).
    pub value_vec: Vec<f64>,
}

impl<'a> NeuralAiController<'a> {
    pub fn new(genome: &'a Genome, cfg: TierConfig) -> Self {
        NeuralAiController {
            genome,
            cfg,
            search: None,
            value_net: None,
        }
    }

    /// Construct a search-enabled controller (Stage A test-time MCTS).
    pub fn with_search(
        genome: &'a Genome,
        cfg: TierConfig,
        search: crate::search::SearchConfig,
    ) -> Self {
        NeuralAiController {
            genome,
            cfg,
            search: Some(search),
            value_net: None,
        }
    }

    /// Construct a search-enabled controller with a learned value net for
    /// [`crate::search::LeafEval::Value`] leaf evaluation (Stage B).
    pub fn with_search_value(
        genome: &'a Genome,
        cfg: TierConfig,
        search: crate::search::SearchConfig,
        value_net: &'a crate::value::ValueNet,
    ) -> Self {
        NeuralAiController {
            genome,
            cfg,
            search: Some(search),
            value_net: Some(value_net),
        }
    }

    /// Public wrapper around the private `staff_income` scaffold, so the search
    /// module can re-staff after replaying an edge action exactly as the
    /// controller's loop does.
    pub fn staff_income_pub(&self, g: &mut Game, player: PlayerId) {
        self.staff_income(g, player);
    }

    /// Public wrapper running the full pre-loop safety scaffold (wood income then
    /// staffing), in the exact order `plan_turn` does. ADDITIVE — used by the
    /// distillation self-play to develop the economy faithfully before recording a
    /// policy decision; does not touch the parity path.
    pub fn ensure_income_pub(&self, g: &mut Game, player: PlayerId) {
        self.ensure_wood_income(g, player);
        self.staff_income(g, player);
    }

    // --- first round --------------------------------------------------------

    /// `placeHeadquarters` — deterministic HQ-placement heuristic (shared by both
    /// TS controllers). Claims the chosen tile via `first_round_actions`.
    pub fn place_headquarters(&self, g: &mut Game, player: PlayerId) {
        // The claim acts on the current player (via first_round_actions); the
        // caller must have `player` as the current seat (as the runner ensures).
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
            let score =
                (free * 3 + grass * 2 + forests * 2 + mountains * 3 + distance) as f64;
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

    /// Full turn. `trace` optionally receives one [`DecisionTrace`] per loop
    /// iteration (for the parity exporter). `rand` is consumed only at
    /// temperature>0/blunder>0.
    pub fn plan_turn<R: Rng>(
        &self,
        g: &mut Game,
        player: PlayerId,
        rand: &mut R,
        mut trace: Option<&mut dyn FnMut(DecisionTrace)>,
    ) {
        let mut budget = self.cfg.budget;

        // 1. Safety scaffold.
        self.ensure_wood_income(g, player);
        self.staff_income(g, player);

        // 2. Learned decision loop.
        let round = g.get_rounds_played();
        while budget > 0 {
            let gvec = global_features(g, player, round);
            let cands = candidates::enumerate(g, player, &self.cfg);
            // Stage A: when a SearchConfig is attached, replace the policy argmax
            // with test-time MCTS (same candidate indexing). With `search == None`
            // this is byte-identical to today, so the parity gate is untouched.
            let chosen_idx = match &self.search {
                None => policy::select_index(self.genome, &gvec, &cands, &self.cfg, rand),
                Some(sc) => crate::search::select_with_value(
                    self.genome,
                    g,
                    player,
                    &self.cfg,
                    sc,
                    self.value_net,
                ),
            };

            if let Some(sink) = trace.as_deref_mut() {
                let spatial = self.search.as_ref().map(|s| s.spatial_policy).unwrap_or(false);
                let scores: Vec<f64> = cands
                    .iter()
                    .map(|c| if spatial {
                        crate::mlp::score(self.genome, &crate::policy_spatial::policy_input_spatial(g, player, &gvec, c))
                    } else {
                        policy::score_candidate(self.genome, &gvec, c)
                    })
                    .collect();
                sink(DecisionTrace {
                    round,
                    global_vec: gvec.clone(),
                    candidates: cands
                        .iter()
                        .map(|c| DecisionCandidate {
                            intent: c.intent as usize,
                            local: c.local.clone(),
                            label: c.label.clone(),
                        })
                        .collect(),
                    scores,
                    chosen_candidate_index: chosen_idx,
                    chosen_intent: cands[chosen_idx].intent as usize,
                });
            }

            let choice = &cands[chosen_idx];
            if choice.intent == Intent::Pass {
                break;
            }
            let mut ok = candidates::execute_action(g, player, &self.cfg, &choice.action);
            if !ok {
                // Retry once with the failed candidate removed.
                let filtered: Vec<candidates::Candidate> = cands
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| i != chosen_idx)
                    .map(|(_, c)| c.clone())
                    .collect();
                if filtered.len() <= 1 {
                    break;
                }
                let ri = if self.search.as_ref().map(|s| s.spatial_policy).unwrap_or(false) {
                    crate::policy_spatial::select_index_spatial(self.genome, g, player, &gvec, &filtered)
                } else {
                    policy::select_index(self.genome, &gvec, &filtered, &self.cfg, rand)
                };
                if filtered[ri].intent == Intent::Pass {
                    break;
                }
                ok = candidates::execute_action(g, player, &self.cfg, &filtered[ri].action);
                if !ok {
                    break;
                }
            }
            budget -= 1;
            // Realise the obvious follow-up: staff anything left unstaffed.
            self.staff_income(g, player);
        }
    }

    /// Like [`plan_turn`], but uses [`crate::search::select_with_pi`] and records
    /// each multi-candidate decision (policy inputs + MCTS visit-count `pi` +
    /// global features) to `sink` for AlphaZero self-play training. Requires a
    /// `SearchConfig` to be attached (else it plays no turn). Mirrors the same
    /// scaffold + enumerate + execute loop so self-play matches real play; the
    /// failed-execute retry path is NOT recorded (no clean target there).
    pub fn plan_turn_record<R: Rng>(
        &self,
        g: &mut Game,
        player: PlayerId,
        rand: &mut R,
        sink: &mut dyn FnMut(RecordedDecision),
    ) {
        let sc = match &self.search {
            Some(s) => *s,
            None => return,
        };
        let mut budget = self.cfg.budget;
        self.ensure_wood_income(g, player);
        self.staff_income(g, player);
        let round = g.get_rounds_played();
        while budget > 0 {
            let gvec = global_features(g, player, round);
            let cands = candidates::enumerate(g, player, &self.cfg);
            if cands.is_empty() {
                break;
            }
            // select_with_pi enumerates the SAME deterministic candidate list, so
            // its `chosen` index maps into `cands` here.
            let res = crate::search::select_with_pi(self.genome, g, player, &self.cfg, &sc, self.value_net, rand);
            if res.policy_inputs.len() > 1 {
                sink(RecordedDecision {
                    player,
                    policy_inputs: res.policy_inputs,
                    pi: res.pi,
                    global_vec: gvec.clone(),
                    value_vec: crate::features::value_features_spatial(g, player, round),
                });
            }
            let chosen_idx = res.chosen.min(cands.len() - 1);
            let choice = &cands[chosen_idx];
            if choice.intent == Intent::Pass {
                break;
            }
            let mut ok = candidates::execute_action(g, player, &self.cfg, &choice.action);
            if !ok {
                let filtered: Vec<candidates::Candidate> = cands
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| i != chosen_idx)
                    .map(|(_, c)| c.clone())
                    .collect();
                if filtered.len() <= 1 {
                    break;
                }
                let ri = if sc.spatial_policy {
                    crate::policy_spatial::select_index_spatial(self.genome, g, player, &gvec, &filtered)
                } else {
                    policy::select_index(self.genome, &gvec, &filtered, &self.cfg, rand)
                };
                if filtered[ri].intent == Intent::Pass {
                    break;
                }
                ok = candidates::execute_action(g, player, &self.cfg, &filtered[ri].action);
                if !ok {
                    break;
                }
            }
            budget -= 1;
            self.staff_income(g, player);
        }
    }

    // --- action plumbing ----------------------------------------------------

    fn add_worker(&self, g: &mut Game, player: PlayerId, tid: TileId) -> bool {
        if g.free_unit_amount(player) <= 0 {
            return false;
        }
        if !s::affords(g, player, &basic_worker_cost(), s::STAFF_RESERVE) {
            return false;
        }
        g.ai_buy_and_place_unit("BasicWorker", tid)
    }

    fn add_expert(&self, g: &mut Game, player: PlayerId, tid: TileId) -> bool {
        self.add_expert_reserve(g, player, tid, self.cfg.reserve)
    }

    /// Buy + place an Expert on `tid` while keeping at least `reserve` money buffered.
    /// Staffing an income building is MECHANICAL, not strategic, so the staffing path
    /// uses the low `STAFF_RESERVE` rather than the strategic `cfg.reserve` (otherwise
    /// the 250-money Expert was almost never affordable early and the plants/mines ran
    /// far below optimal output — the metal-economy starvation root cause).
    fn add_expert_reserve(
        &self,
        g: &mut Game,
        player: PlayerId,
        tid: TileId,
        reserve: i64,
    ) -> bool {
        if g.free_unit_amount(player) <= 0 {
            return false;
        }
        if !g.tiles[tid.0].has_space_for_units() {
            return false;
        }
        if !s::affords(g, player, &expert_cost(), reserve) {
            return false;
        }
        g.ai_buy_and_place_unit("Expert", tid)
    }

    /// Number of BasicWorkers currently on a tile.
    fn worker_count(&self, g: &Game, tid: TileId) -> i64 {
        g.tile_units(tid)
            .iter()
            .filter(|&&u| g.units[u.0].kind == UnitType::BasicWorker)
            .count() as i64
    }

    // --- safety scaffold ----------------------------------------------------

    fn find_idle_on_plain(&self, g: &Game, player: PlayerId) -> Option<(UnitId, TileId)> {
        for tid in m::owned_tiles(g, player) {
            let ty = g.tiles[tid.0].tile_type;
            if g.tiles[tid.0].building.is_some()
                || ty == TileType::Forest
                || ty == TileType::AbundantForest
            {
                continue;
            }
            if let Some(w) = self.first_worker(g, tid) {
                return Some((w, tid));
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

    fn find_spare_worker(
        &self,
        g: &Game,
        player: PlayerId,
        exclude: TileId,
    ) -> Option<(UnitId, TileId)> {
        for tid in m::owned_tiles(g, player) {
            if tid == exclude {
                continue;
            }
            let ty = g.tiles[tid.0].building.as_ref().map(|b| b.kind);
            if matches!(
                ty,
                Some(BuildingType::Farm)
                    | Some(BuildingType::Mine)
                    | Some(BuildingType::Nuclear)
                    | Some(BuildingType::Hydro)
            ) {
                continue;
            }
            if let Some(w) = self.first_worker(g, tid) {
                return Some((w, tid));
            }
        }
        None
    }

    fn find_expendable_worker(&self, g: &Game, player: PlayerId) -> Option<(UnitId, TileId)> {
        if let Some(idle) = self.find_idle_on_plain(g, player) {
            return Some(idle);
        }
        // Surplus producer worker.
        for tid in m::owned_tiles(g, player) {
            let ty = g.tiles[tid.0].building.as_ref().map(|b| b.kind);
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
        let farms: Vec<TileId> = m::owned_tiles(g, player)
            .into_iter()
            .filter(|&t| {
                g.tiles[t.0].building.as_ref().map(|b| b.kind) == Some(BuildingType::Farm)
                    && m::has_type(g, t, UnitType::BasicWorker)
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

    /// `ensureWoodIncome` — staff enough forest harvesters to cover wood upkeep.
    fn ensure_wood_income(&self, g: &mut Game, player: PlayerId) {
        let upkeep = m::wood_upkeep(g, player);
        if upkeep <= 0.0 {
            return;
        }
        let harvesters = |g: &Game| -> i64 {
            m::owned_tiles(g, player)
                .into_iter()
                .filter(|&t| {
                    g.tiles[t.0].tile_type == TileType::Forest
                        && g.tiles[t.0].building.is_none()
                        && m::has_type(g, t, UnitType::BasicWorker)
                })
                .count() as i64
        };
        let mut need = 1i64.max((upkeep / 40.0).ceil() as i64);
        if (m::wood(g, player) as f64) < upkeep * 4.0 {
            need += 1;
        }
        let mut guard = 0;
        while harvesters(g) < need && guard < 8 {
            guard += 1;
            let f = m::owned_tiles(g, player).into_iter().find(|&t| {
                g.tiles[t.0].tile_type == TileType::Forest
                    && g.tiles[t.0].building.is_none()
                    && g.tiles[t.0].has_space_for_units()
                    && !m::has_type(g, t, UnitType::BasicWorker)
            });
            let f = match f {
                Some(t) => t,
                None => break,
            };
            let mut did = false;
            if g.free_unit_amount(player) > 0
                && s::affords(g, player, &basic_worker_cost(), s::STAFF_RESERVE)
            {
                did = self.add_worker(g, player, f);
            } else if let Some((unit, from)) = self.find_expendable_worker(g, player) {
                if from != f {
                    did = g.ai_move_unit(unit, from, f);
                }
            }
            if !did {
                break;
            }
        }
    }

    /// `staffIncome` — staff every income building toward OPTIMAL output.
    ///
    /// Staffing a producer is a MECHANICAL action (the policy never has to "decide" to
    /// add a 2nd mine worker), and the previous scaffold deliberately under-staffed:
    /// it put only ONE worker on a mine (20 metal/round) when the optimum is 2 workers
    /// + 1 Expert (`20 * 2 * 2 = 80`, a 4× metal loss), and it put an Expert but NO
    /// worker on a Nuclear plant — which then produces NOTHING (the formula needs both
    /// `workers > 0 && expert`). On a ~1-mine economy that starved the metal income so
    /// the AI could never fund an army. This rewrite fully staffs producers, gated only
    /// by the shared unit cap (`free_unit_amount`), per-tile space (≤3 units, enforced
    /// inside `ai_buy_and_place_unit`), and the LOW `STAFF_RESERVE` (mechanical staffing
    /// must not be starved by the strategic `cfg.reserve`).
    ///
    /// Two passes so a scarce unit cap is spent on coverage first, luxury second:
    ///   1. minimum-viable — every producer gets the staffing it needs to produce >0
    ///      (mine/farm: 1 worker; nuclear/hydro: 1 worker + 1 expert);
    ///   2. upgrade — mines toward the 2-worker + 1-expert optimum, then a 2nd hydro
    ///      worker (hydro = `80 * workers`).
    fn staff_income(&self, g: &mut Game, player: PlayerId) {
        let producers = |g: &Game, kinds: &[BuildingType]| -> Vec<TileId> {
            m::owned_tiles(g, player)
                .into_iter()
                .filter(|&t| {
                    g.tiles[t.0]
                        .building
                        .as_ref()
                        .map(|b| kinds.contains(&b.kind))
                        .unwrap_or(false)
                })
                .collect()
        };

        // --- Pass 1: minimum-viable staffing (each producer produces > 0) ----------
        for tid in m::owned_tiles(g, player) {
            let ty = g.tiles[tid.0].building.as_ref().map(|b| b.kind);
            match ty {
                Some(BuildingType::Farm) => {
                    if !m::has_type(g, tid, UnitType::BasicWorker) {
                        self.add_worker(g, player, tid);
                    }
                }
                Some(BuildingType::Mine) => {
                    self.ensure_worker(g, player, tid);
                }
                Some(BuildingType::Nuclear) | Some(BuildingType::Hydro) => {
                    // Both produce NOTHING without an Expert AND a worker.
                    if self.cfg.experts && !m::has_type(g, tid, UnitType::Expert) {
                        self.add_expert_reserve(g, player, tid, s::STAFF_RESERVE);
                    }
                    if m::has_type(g, tid, UnitType::Expert)
                        && !m::has_type(g, tid, UnitType::BasicWorker)
                    {
                        self.add_worker(g, player, tid);
                    }
                }
                _ => {
                    if g.tiles[tid.0].tile_type == TileType::AbundantForest
                        && !m::has_type(g, tid, UnitType::BasicWorker)
                    {
                        self.add_worker(g, player, tid);
                    }
                }
            }
        }

        // --- Pass 2: upgrade mines to optimal (2 workers + 1 expert = 80/round) -----
        // Experts double a mine's output, so add experts before the 2nd worker.
        if self.cfg.experts {
            for tid in producers(g, &[BuildingType::Mine]) {
                if m::has_type(g, tid, UnitType::BasicWorker)
                    && !m::has_type(g, tid, UnitType::Expert)
                {
                    self.add_expert_reserve(g, player, tid, s::STAFF_RESERVE);
                }
            }
        }
        for tid in producers(g, &[BuildingType::Mine]) {
            if self.worker_count(g, tid) < 2 && g.tiles[tid.0].has_space_for_units() {
                self.add_worker(g, player, tid);
            }
        }
        // A 2nd hydro worker (hydro = 80 * workers, expert-gated) if cap/space allow.
        for tid in producers(g, &[BuildingType::Hydro]) {
            if m::has_type(g, tid, UnitType::Expert)
                && self.worker_count(g, tid) < 2
                && g.tiles[tid.0].has_space_for_units()
            {
                self.add_worker(g, player, tid);
            }
        }
    }

    /// `ensureWorker` — guarantee one worker on a key building, relocating an
    /// idle/forest worker if capped.
    fn ensure_worker(&self, g: &mut Game, player: PlayerId, tid: TileId) {
        if m::has_type(g, tid, UnitType::BasicWorker) {
            return;
        }
        if g.free_unit_amount(player) > 0 {
            self.add_worker(g, player, tid);
            return;
        }
        let spare = self
            .find_idle_on_plain(g, player)
            .or_else(|| self.find_spare_worker(g, player, tid));
        if let Some((unit, from)) = spare {
            if from != tid {
                g.ai_move_unit(unit, from, tid);
            }
        }
    }
}

#[cfg(test)]
mod staffing_tests {
    use super::*;
    use crate::policy::DEFAULT_ARCH;
    use crate::tiers::TRAINING_CONFIG;
    use cp_sim::Game;

    /// `staff_income` must staff a Mine to OPTIMAL (2 BasicWorkers + 1 Expert =
    /// `20*2*2 = 80` metal/round) and a Nuclear to producing (worker + expert), given
    /// enough unit cap, tile space, and money. Regression guard for the metal-economy
    /// starvation root cause (the old scaffold put 1 worker on a mine = 20 metal, and
    /// an Expert-but-no-worker on a Nuclear = 0 output).
    #[test]
    fn staff_income_fully_staffs_mine_and_nuclear() {
        let mut g = Game::new(12, 12, &["P1", "P2"]);
        g.generate_map(12, 12, 1);
        let p1 = PlayerId(0);
        let id = |x: i32, y: i32| TileId((x * 12 + y) as usize);

        // Own a generous patch so the cap and `owned_tiles` are populated.
        for x in 3..9 {
            for y in 3..9 {
                g.set_tile_owner(id(x, y), Some(p1));
            }
        }
        // HQ (+3 unit cap) + two Villages (+3 each) => cap 9, enough for a full mine
        // (3 units) + a full nuclear (2 units) + slack.
        g.place_building(id(5, 8), BuildingType::Headquarters, Some(p1));
        g.place_building(id(3, 3), BuildingType::Village, Some(p1));
        g.place_building(id(4, 3), BuildingType::Village, Some(p1));
        // The income buildings under test.
        let mine = id(5, 5);
        let nuke = id(6, 6);
        g.place_building(mine, BuildingType::Mine, Some(p1));
        g.place_building(nuke, BuildingType::Nuclear, Some(p1));
        g.update_unit_amounts(p1);
        // Plenty of money so affordability never blocks staffing.
        g.set_player_resources(p1, 100_000, 100_000, 100_000, 100_000);

        let genome = Genome::zero(&DEFAULT_ARCH);
        let ctrl = NeuralAiController::new(&genome, TRAINING_CONFIG);
        // Run twice (the per-iteration scaffold runs repeatedly in a real turn).
        ctrl.staff_income(&mut g, p1);
        ctrl.staff_income(&mut g, p1);

        // Mine: 2 BasicWorkers + 1 Expert => 80 metal/round.
        assert_eq!(
            ctrl.worker_count(&g, mine),
            2,
            "mine should be staffed with 2 BasicWorkers"
        );
        assert!(
            m::has_type(&g, mine, UnitType::Expert),
            "mine should have an Expert (doubles output)"
        );
        assert_eq!(
            m::metal_income_per_round(&g, p1),
            80.0,
            "fully-staffed mine must yield 80 metal/round, not 20"
        );

        // Nuclear: must have BOTH a worker and an expert (else it produces 0).
        assert!(
            m::has_type(&g, nuke, UnitType::Expert),
            "nuclear must have an Expert"
        );
        assert!(
            m::has_type(&g, nuke, UnitType::BasicWorker),
            "nuclear must have a BasicWorker (Nuclear needs worker AND expert to produce)"
        );
    }
}
