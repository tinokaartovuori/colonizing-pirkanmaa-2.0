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
use cp_sim::resources::{
    basic_worker_cost, expert_cost, mine_build_cost, village_build_cost, ResourceMap,
};
use cp_sim::{BuildingType, Game, PlayerId, TileId, TileType, UnitId, UnitType};

use crate::mlp::Genome;

// --- TEMP staffing diagnostics (env-gated, parity-free) ----------------------
// Enabled by setting CP_DIAG_STAFF=1. Counts expert-add attempts and their
// failure reasons, plus worker adds, across all controller staffing calls.
// REMOVE before finalizing (or leave: pure eprintln/atomics, no behaviour change).
pub mod diag {
    use std::sync::atomic::{AtomicU64, Ordering};
    pub static ENABLED: AtomicU64 = AtomicU64::new(u64::MAX); // u64::MAX = "unchecked"
    pub static EXPERT_ATTEMPT: AtomicU64 = AtomicU64::new(0);
    pub static EXPERT_OK: AtomicU64 = AtomicU64::new(0);
    pub static EXPERT_FAIL_NOSLOT: AtomicU64 = AtomicU64::new(0);
    pub static EXPERT_FAIL_NOSPACE: AtomicU64 = AtomicU64::new(0);
    pub static EXPERT_FAIL_UNAFFORD: AtomicU64 = AtomicU64::new(0);
    pub static EXPERT_FAIL_BUY: AtomicU64 = AtomicU64::new(0);
    pub static WORKER_OK: AtomicU64 = AtomicU64::new(0);
    pub static VILLAGE_OK: AtomicU64 = AtomicU64::new(0);
    pub static MINE_EXPERT_GATE_SKIP: AtomicU64 = AtomicU64::new(0); // expert pass not reached on mine
    pub static UNAFFORD_MONEY_SUM: AtomicU64 = AtomicU64::new(0);
    pub static UNAFFORD_DRAIN5_SUM: AtomicU64 = AtomicU64::new(0);

    pub fn on() -> bool {
        let cached = ENABLED.load(Ordering::Relaxed);
        if cached != u64::MAX {
            return cached == 1;
        }
        let v = std::env::var("CP_DIAG_STAFF").map(|s| s == "1").unwrap_or(false);
        ENABLED.store(if v { 1 } else { 0 }, Ordering::Relaxed);
        v
    }
    pub fn inc(c: &AtomicU64) {
        c.fetch_add(1, Ordering::Relaxed);
    }
    pub fn reset() {
        for c in [
            &EXPERT_ATTEMPT, &EXPERT_OK, &EXPERT_FAIL_NOSLOT, &EXPERT_FAIL_NOSPACE,
            &EXPERT_FAIL_UNAFFORD, &EXPERT_FAIL_BUY, &WORKER_OK, &VILLAGE_OK,
            &MINE_EXPERT_GATE_SKIP, &UNAFFORD_MONEY_SUM, &UNAFFORD_DRAIN5_SUM,
        ] {
            c.store(0, Ordering::Relaxed);
        }
    }
    pub fn dump(label: &str) {
        eprintln!(
            "[DIAG-STAFF {label}] expert_attempt={} expert_ok={} fail(noslot={} nospace={} unafford={} buy={}) worker_ok={} village_ok={} mine_expert_gate_skip={}",
            EXPERT_ATTEMPT.load(Ordering::Relaxed),
            EXPERT_OK.load(Ordering::Relaxed),
            EXPERT_FAIL_NOSLOT.load(Ordering::Relaxed),
            EXPERT_FAIL_NOSPACE.load(Ordering::Relaxed),
            EXPERT_FAIL_UNAFFORD.load(Ordering::Relaxed),
            EXPERT_FAIL_BUY.load(Ordering::Relaxed),
            WORKER_OK.load(Ordering::Relaxed),
            VILLAGE_OK.load(Ordering::Relaxed),
            MINE_EXPERT_GATE_SKIP.load(Ordering::Relaxed),
        );
        let nf = EXPERT_FAIL_UNAFFORD.load(Ordering::Relaxed).max(1);
        eprintln!(
            "[DIAG-STAFF {label}] at-unafford avg money={} avg drain*5={} (expert needs money>=270+drain*5)",
            UNAFFORD_MONEY_SUM.load(Ordering::Relaxed) / nf,
            UNAFFORD_DRAIN5_SUM.load(Ordering::Relaxed) / nf,
        );
    }
}

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
    /// module can re-staff after replaying an edge action exactly as the controller's
    /// loop does. (Cap-expansion runs once per turn via `ensure_income_pub`, NOT here:
    /// running it per-action churned the economy — repeated farm-worker borrowing for
    /// wood — and measurably hurt play, so it stays a turn-start step.)
    pub fn staff_income_pub(&self, g: &mut Game, player: PlayerId) {
        self.staff_income(g, player);
    }

    /// Public wrapper running the full pre-loop safety scaffold (wood income, staffing,
    /// then the MECHANICAL economy guarantees), in the exact order the deployed CNN turn
    /// (`cnn_train.rs::cnn_plan_turn` / `scaffold_ensure`) runs. ADDITIVE — used by the
    /// distillation self-play AND the CNN bench/validate path to develop the economy
    /// faithfully before recording a policy decision; does NOT touch the MLP parity path
    /// (`plan_turn` / `plan_turn_record` stay byte-identical to the TS controller, which
    /// has no `ensure_metal_income` / `ensure_unit_cap` mirror — adding the mine build to
    /// the parity path would diverge on any map where a seat owns an early Mountain).
    ///
    /// Order: secure WOOD income, staff producers, expand the unit CAP (villages) if it
    /// blocks full staffing, then GUARANTEE the metal source as a SAFETY NET on whatever
    /// resources remain, then staff (mans the new mine). The mine is sequenced AFTER the
    /// cap/staff flow on purpose: building it first stole the early money/wood the
    /// village→cap chain needs and collapsed the economy (mines 1.7→0.7) — it must be a
    /// leftover-resource backstop, not a competitor for the early budget.
    pub fn ensure_income_pub(&self, g: &mut Game, player: PlayerId) {
        self.ensure_wood_income(g, player);
        self.staff_income(g, player);
        self.ensure_unit_cap(g, player);
        self.ensure_metal_income(g, player);
        self.staff_income(g, player);
    }

    /// CNN-training variant of [`ensure_income_pub`] that runs the SAME wood-income /
    /// worker-staffing / cap-village scaffold but does NOT place any Expert, and does
    /// NOT mechanically guarantee the metal mine (mine #1 becomes a late fallback —
    /// see [`ensure_metal_income_fallback_pub`]). This hands the EXPERT and MINE-COUNT
    /// decisions to the learned policy (via `StackProducer`/`BuildMine`) instead of the
    /// mechanical scaffold, while still guaranteeing wood income + 1st-worker staffing
    /// + the cap-village bootstrap (keeping the Pass-collapse safety net intact). Used
    /// ONLY by the CNN train/bench path; does NOT touch the MLP parity path.
    pub fn ensure_income_no_experts_pub(&self, g: &mut Game, player: PlayerId) {
        self.ensure_wood_income(g, player);
        self.staff_income_inner(g, player, false);
        self.ensure_unit_cap(g, player);
        self.staff_income_inner(g, player, false);
    }

    /// CNN re-staff after a candidate executes, WITHOUT placing Experts (the policy
    /// owns the Expert decision). Worker staffing still runs so newly-built producers
    /// get their 1st worker (income realised) — only the Expert step is withheld.
    pub fn staff_income_no_experts_pub(&self, g: &mut Game, player: PlayerId) {
        self.staff_income_inner(g, player, false);
    }

    /// EXPERT FALLBACK (CNN path): run AFTER the policy's turn loop. If the policy did
    /// NOT place the Expert(s) itself, this guarantees them — so the economy is never
    /// permanently understaffed, but the policy still got first chance to choose
    /// `StackProducer:Expert` (and be labelled for it). This is the deferred half of
    /// the old up-front expert placement.
    pub fn ensure_experts_fallback_pub(&self, g: &mut Game, player: PlayerId) {
        // staff_income with experts ON only fills the still-empty expert slots (every
        // worker slot is already filled by the no-experts scaffold above).
        self.staff_income_inner(g, player, true);
    }

    /// METAL-MINE FALLBACK (CNN path): build mine #1 ONLY if the policy has built ZERO
    /// mines by `min_round` — a true backstop so a policy that never learns mines still
    /// gets a metal source, while a policy that DID build mines keeps full ownership of
    /// mine COUNT. `accumulate_wood_for` still runs inside `ensure_metal_income`, so the
    /// wood-trap stays broken either way.
    pub fn ensure_metal_income_fallback_pub(&self, g: &mut Game, player: PlayerId, min_round: i64) {
        // `ensure_metal_income_gated` itself no-ops once a mine exists; before the gate
        // it only runs the wood-accumulation harvester (keeping a policy mine fundable),
        // and only AFTER `min_round` does it mechanically build mine #1 as a backstop.
        self.ensure_metal_income_gated(g, player, min_round);
        self.staff_income_inner(g, player, false);
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

        // 1. Safety scaffold. Staff once so producer income is realised, THEN expand
        //    the unit cap if it blocks full staffing, THEN staff again to fill it.
        self.ensure_wood_income(g, player);
        self.staff_income(g, player);
        self.ensure_unit_cap(g, player);
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
            // Realise the obvious follow-up: staff, expand the unit cap if it now
            // blocks staffing, then staff the new slots.
            self.staff_income(g, player);
            self.ensure_unit_cap(g, player);
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
        self.ensure_unit_cap(g, player);
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
            self.ensure_unit_cap(g, player);
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
        let ok = g.ai_buy_and_place_unit("BasicWorker", tid);
        if ok && diag::on() {
            diag::inc(&diag::WORKER_OK);
        }
        ok
    }

    fn add_expert(&self, g: &mut Game, player: PlayerId, tid: TileId) -> bool {
        self.add_expert_reserve(g, player, tid, self.cfg.reserve)
    }

    /// Buy + place an Expert on `tid` while keeping at least `reserve` money buffered.
    /// Staffing an income building is MECHANICAL, not strategic, so the staffing path
    /// uses the low `STAFF_RESERVE` rather than the strategic `cfg.reserve` (otherwise
    /// the 250-money Expert was almost never affordable early and the plants/mines ran
    /// far below optimal output — the metal-economy starvation root cause).
    ///
    /// AFFORDABILITY (metal-economy root-cause fix, 2026-06-07): the Expert is a ONE-TIME
    /// capital spend that DOUBLES a mine (or enables a plant) — the single
    /// highest-leverage economic action. The general `affords()` keeps a 5-rounds-of-drain
    /// solvency buffer (`money >= cost + reserve + drain*5`); instrumentation showed that
    /// buffer (avg `drain*5 ≈ 411`) — NOT the raw 250 cost — blocked ~46% of expert
    /// purchases (player typically held ~368 money, plenty for the 250 expert, but short
    /// of the ~681 the drain buffer demanded). A mine expert returns METAL not money, so
    /// it never "pays back" the money buffer, making the gate permanently unreachable as
    /// the economy grows. We instead gate it like an income build: raw resources + keep a
    /// modest money FLOOR (the strategic reserve + ~1 round of drain), so it never empties
    /// the treasury but is not strangled by a 5-round buffer it can't earn back in money.
    fn add_expert_reserve(
        &self,
        g: &mut Game,
        player: PlayerId,
        tid: TileId,
        reserve: i64,
    ) -> bool {
        if diag::on() {
            diag::inc(&diag::EXPERT_ATTEMPT);
        }
        if g.free_unit_amount(player) <= 0 {
            if diag::on() {
                diag::inc(&diag::EXPERT_FAIL_NOSLOT);
            }
            return false;
        }
        if !g.tiles[tid.0].has_space_for_units() {
            if diag::on() {
                diag::inc(&diag::EXPERT_FAIL_NOSPACE);
            }
            return false;
        }
        // Income-build affordability: keep `reserve` + ~1 round of money drain as a floor
        // (solvency-safe) rather than the 5-round buffer of `affords()` (the root-cause
        // gate that blocked experts as the economy grew — see doc above).
        let floor = reserve + m::money_drain_per_round(g, player).ceil() as i64;
        if !s::affords_income_build(g, player, &expert_cost(), floor) {
            if diag::on() {
                diag::inc(&diag::EXPERT_FAIL_UNAFFORD);
                diag::UNAFFORD_MONEY_SUM
                    .fetch_add(m::money(g, player).max(0) as u64, std::sync::atomic::Ordering::Relaxed);
                diag::UNAFFORD_DRAIN5_SUM.fetch_add(
                    (m::money_drain_per_round(g, player) * 5.0).max(0.0) as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
            }
            return false;
        }
        let ok = g.ai_buy_and_place_unit("Expert", tid);
        if diag::on() {
            if ok {
                diag::inc(&diag::EXPERT_OK);
            } else {
                diag::inc(&diag::EXPERT_FAIL_BUY);
            }
        }
        ok
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

    /// `ensureUnitCap` — MECHANICAL cap-expansion: build a Village when the shared
    /// unit cap is the only thing blocking `staff_income` from fully staffing the
    /// existing producers (2 workers + Expert per Mine, worker + Expert per plant).
    ///
    /// Root cause this fixes: `staff_income`'s coverage pass exhausts
    /// `free_unit_amount` putting 1 worker on each producer, so the Expert / 2nd-worker
    /// upgrade pass never fires (experts/game = 0, mines stuck at 20 metal). Nothing in
    /// the controller ever expanded the cap, so the metal economy could never fund an
    /// army. A Village (+3 unit slots) is the only cap source the AI controls; the
    /// learned policy is free to ignore villages — this guarantees the economy fills.
    ///
    /// Gates (so it never bankrupts and never builds a useless drain): only when the
    /// staffing DEFICIT (slots needed to fully staff producers, minus what the current
    /// free cap covers) is positive — i.e. a new village's +3 slots would be USED;
    /// only when net money stays solvent after the village's -10/round upkeep AND a few
    /// rounds of unit salaries the new slots imply; and only when affordable on the
    /// strategic reserve. One village per call, so the cap grows at most +3 per staffing
    /// pass and tracks need.
    fn ensure_unit_cap(&self, g: &mut Game, player: PlayerId) {
        if !self.cfg.experts {
            return; // No experts tier => no 3-unit producers to fund; cap rarely binds.
        }
        // The unit cap is cached; refresh so `free_unit_amount` reflects any village /
        // tile change earlier this turn (the learned loop may have built one).
        g.update_unit_amounts(player);
        // Producer staffing deficit: how many MORE unit slots full staffing wants.
        let mut deficit = 0i64;
        let mut any_underfilled_tile_has_space = false;
        for tid in m::owned_tiles(g, player) {
            let kind = match g.tiles[tid.0].building.as_ref().map(|b| b.kind) {
                Some(k) => k,
                None => continue,
            };
            let workers = self.worker_count(g, tid);
            let expert = m::has_type(g, tid, UnitType::Expert);
            // Units this producer should hold at optimal output.
            let optimal = match kind {
                BuildingType::Mine => 3,                       // 2 workers + 1 expert = 80 metal
                BuildingType::Nuclear | BuildingType::Hydro => 2, // 1 worker + 1 expert
                BuildingType::Farm => 1,                       // 1 worker
                _ => 0,
            };
            if optimal == 0 {
                continue;
            }
            let current = workers + if expert { 1 } else { 0 };
            let want = (optimal - current).max(0);
            if want > 0 {
                deficit += want;
                if g.tiles[tid.0].has_space_for_units() {
                    any_underfilled_tile_has_space = true;
                }
            }
        }
        // Only expand when the cap is what's blocking us: the existing free slots don't
        // already cover the deficit, AND there's a producer tile with physical space to
        // place the units the new slots would buy.
        let free = g.free_unit_amount(player);
        if deficit <= free || !any_underfilled_tile_has_space {
            return;
        }
        // Solvency: a Village costs -10 money/round upkeep. Require net money to stay
        // non-negative after that upkeep alone. We do NOT pre-charge the new workers'
        // salaries here: the workers go on PRODUCERS (a fully-staffed Nuclear pays
        // +160/worker, Hydro +80, Farm ~+44; a Mine pays metal not money) so they fund
        // their own salary — pre-charging them re-creates the give-up bug this fixes.
        // `affords` (in build_village) additionally buffers the strategic reserve + 5
        // rounds of money drain, so this stays bankruptcy-safe.
        if m::net_money_per_round(g, player) - 10.0 < 0.0 {
            return;
        }
        // A Village costs 200 WOOD. With zero villages, wood upkeep is 0, so
        // `ensure_wood_income` harvests NOTHING and wood sits at the ~100 starting level
        // forever — the cap-expansion can never be afforded (the deepest layer of the
        // starvation trap). When we genuinely want a village (real producer deficit) but
        // can't afford its wood, run a forest harvester to ACCUMULATE wood toward the
        // cost; the village is then built on a later turn once the buffer is there.
        if !s::affords(g, player, &village_build_cost(), self.cfg.reserve) {
            self.accumulate_wood_for(g, player, &village_build_cost());
            return;
        }
        self.build_village(g, player);
    }

    /// Ensure at least one forest harvester is running so wood accumulates toward a
    /// wood-costed build (a Village's 200-wood cost, a Mine's 200-wood cost, a Bridge's
    /// 300-wood cost). With no villages there is no wood upkeep, so the normal
    /// `ensure_wood_income` no-ops and wood never grows — this is the proactive harvest
    /// that unblocks the very first wood-blocked build (cap-expanding Village OR the first
    /// Mine — the metal-source bottleneck). Prefers a free unit slot; when capped (the
    /// common case in the trap), relocates an EXPENDABLE worker (idle / surplus producer
    /// worker) onto a forest so the wood income turns positive without permanently
    /// sacrificing producer output. One placement per call.
    ///
    /// `_cost` is accepted so callers document WHAT they are saving toward; the harvest
    /// itself is cost-agnostic (one running harvester grows wood toward any target).
    fn accumulate_wood_for(&self, g: &mut Game, player: PlayerId, _cost: &ResourceMap) {
        // Already have a working forest harvester? Then wood is already growing — just
        // wait for the buffer; don't pull workers off producers needlessly.
        let have_harvester = m::owned_tiles(g, player).into_iter().any(|t| {
            g.tiles[t.0].tile_type == TileType::Forest
                && g.tiles[t.0].building.is_none()
                && m::has_type(g, t, UnitType::BasicWorker)
        });
        if have_harvester {
            return;
        }
        let forest = m::owned_tiles(g, player).into_iter().find(|&t| {
            g.tiles[t.0].tile_type == TileType::Forest
                && g.tiles[t.0].building.is_none()
                && g.tiles[t.0].has_space_for_units()
                && !m::has_type(g, t, UnitType::BasicWorker)
        });
        let forest = match forest {
            Some(t) => t,
            None => return, // no harvestable forest — fall back to whatever wood exists
        };
        if g.free_unit_amount(player) > 0
            && s::affords(g, player, &basic_worker_cost(), s::STAFF_RESERVE)
        {
            self.add_worker(g, player, forest);
            return;
        }
        // Capped with all producers minimally staffed (the trap): there is no "spare"
        // worker. Temporarily borrow a Farm worker (least critical — farms feed money
        // we have plenty of in the trap; metal-mine workers are NOT touched). The
        // staff_income pass will re-fill the farm once the village raises the cap.
        let borrow = self
            .find_expendable_worker(g, player)
            .or_else(|| {
                m::owned_tiles(g, player).into_iter().find_map(|t| {
                    if g.tiles[t.0].building.as_ref().map(|b| b.kind) == Some(BuildingType::Farm) {
                        self.first_worker(g, t).map(|u| (u, t))
                    } else {
                        None
                    }
                })
            });
        if let Some((unit, from)) = borrow {
            if from != forest {
                g.ai_move_unit(unit, from, forest);
            }
        }
    }

    /// `ensureMetalIncome` — MECHANICAL metal-source guarantee: build a Mine on an owned
    /// buildable Mountain when the player is below a baseline metal economy.
    ///
    /// Root cause this fixes (the #1 economy bottleneck): metal is the army resource, but
    /// the ONLY metal source is the Mine, and nothing in the scaffold ever built one — the
    /// learned policy treats the Mine candidate as a trap (200 wood up-front on a
    /// wood-starved early economy, deferred payoff) so mines/game ≈ 0.5. With ~0.5 mines
    /// the metal income is ~10-25/round, which can never fund the soldier-cap → army chain
    /// (Outpost 100 metal + soldiers 30 metal each + upkeep). This mirrors `ensure_unit_cap`
    /// exactly — a Village was made mechanical to guarantee the unit cap; here a Mine is
    /// made mechanical to guarantee the metal source. The learned policy is still free to
    /// build MORE mines (plants, etc.); this only guarantees the baseline.
    ///
    /// Gates (so it never floods on a money-poor / mountain-rich map and never bankrupts):
    ///   - only when an owned, empty, Mine-buildable Mountain tile actually exists;
    ///   - only when the player has ZERO mines — this is a SAFETY NET for the metal-starved
    ///     tail of games, NOT a competitor to the learned policy (which already averages
    ///     ~1.7 mines/game). Guaranteeing more than the first mine here stole the early
    ///     budget from the village→cap→staff flow and COLLAPSED the economy in testing;
    ///   - solvency: `affords` (reserve + 5 rounds of drain) AND a comfortable money
    ///     cushion above the 200-money cost, so the mine never starves the (more
    ///     fundamental) cap/staff flow that the caller runs first.
    /// If WOOD is the blocker for the 200-wood cost (the early-economy trap), run a forest
    /// harvester to accumulate wood and defer the build to a later turn — exactly the
    /// village path. One mine per call; once it exists, hands off to the policy.
    fn ensure_metal_income(&self, g: &mut Game, player: PlayerId) {
        self.ensure_metal_income_gated(g, player, 0);
    }

    /// As [`ensure_metal_income`] but the actual mine BUILD is deferred until
    /// `build_round_gate` rounds have elapsed; the WOOD-accumulation harvester still
    /// runs from round 0 while the player owns 0 mines (so the wood-trap stays broken
    /// and a policy-built mine is always fundable). With `build_round_gate = 0` this is
    /// byte-identical to the original `ensure_metal_income`.
    fn ensure_metal_income_gated(&self, g: &mut Game, player: PlayerId, build_round_gate: i64) {
        // Owned, empty, Mine-buildable Mountain tiles — the only metal-source sites.
        let mountains: Vec<TileId> = m::owned_tiles(g, player)
            .into_iter()
            .filter(|&t| {
                g.tiles[t.0].tile_type == TileType::Mountain
                    && g.tiles[t.0].building.is_none()
                    && g.buildable_buildings(t).contains(&"Mine")
            })
            .collect();
        if mountains.is_empty() {
            return; // no metal source available — nothing mechanical to do
        }
        // Current mine count (built; staffing is staff_income's job afterward).
        let mines = m::owned_tiles(g, player)
            .into_iter()
            .filter(|&t| {
                g.tiles[t.0].building.as_ref().map(|b| b.kind) == Some(BuildingType::Mine)
            })
            .count() as i64;
        // SAFETY-NET, not a flood: guarantee only the FIRST metal source. The learned
        // policy already builds ~1.7 mines/game on average; this exists purely so the
        // games where it builds ZERO (the metal-starved tail) still get one. Once a mine
        // exists, leave further mines to the policy — building more here stole the early
        // money/wood the village→cap→staff flow needs and COLLAPSED the economy in
        // testing (mines 1.7→0.7, villages 3.0→0.8). One mine, then hands off.
        if mines >= 1 {
            return;
        }
        let cost = mine_build_cost();
        // Solvency: keep the strategic reserve + 5 rounds of drain buffered (the same
        // `affords` gate the village path uses). A Mine has no per-round MONEY upkeep and
        // is +20 money/worker once staffed, so the only risk is the one-time spend.
        // Stricter than the village path: also require a comfortable money cushion so the
        // mine's 200-money cost doesn't starve the (more fundamental) cap/staff flow that
        // runs after this — the early-money crunch is what regressed the economy above.
        if !s::affords(g, player, &cost, self.cfg.reserve) {
            return;
        }
        let cost_money = -mine_build_cost().get(cp_sim::resources::BasicResource::Money).unwrap_or(0);
        if m::money(g, player) < cost_money + self.cfg.reserve + 100 {
            return; // not enough headroom — let the economy mature, retry next turn
        }
        // Wood is the early blocker (200 wood up-front, no wood income with 0 villages).
        // Accumulate toward it via a forest harvester, then build on a later turn — the
        // exact same trap-breaker the first cap-expanding Village uses.
        if !s::has_wood_buffer(g, player, &cost) {
            self.accumulate_wood_for(g, player, &cost);
            return;
        }
        // Defer the mechanical mine BUILD until the gate elapses, so the learned policy
        // gets first crack at owning mine #1 (it has wood ready by now — the harvester
        // above ran from round 0). The fallback only fires when the policy still has 0
        // mines past the gate.
        if g.get_rounds_played() < build_round_gate {
            return;
        }
        self.build_mine(g, player, mountains[0]);
    }

    /// Buy + place a Mine on an owned empty Mountain tile (chosen by `ensure_metal_income`),
    /// mirroring `build_village` — solvency/wood gated by the caller, uses `cfg.reserve` so
    /// it never dips into the strategic buffer. Acts on the current seat (== `player`).
    fn build_mine(&self, g: &mut Game, player: PlayerId, spot: TileId) -> bool {
        let cost = mine_build_cost();
        if !s::affords(g, player, &cost, self.cfg.reserve) || !s::has_wood_buffer(g, player, &cost)
        {
            return false;
        }
        if g.tiles[spot.0].tile_type != TileType::Mountain
            || g.tiles[spot.0].building.is_some()
            || !g.buildable_buildings(spot).contains(&"Mine")
        {
            return false;
        }
        debug_assert_eq!(g.current_player(), player);
        g.ai_build_building("Mine", spot)
    }

    /// Buy + place a Village on an empty owned grassland tile, mirroring the
    /// candidate-list `build_village` tile choice but WITHOUT the strategic income/wood
    /// gates (those are for the learned policy; this is the mechanical cap-fill path,
    /// already solvency-gated by `ensure_unit_cap`). Uses `cfg.reserve` so it never
    /// dips into the strategic buffer. Acts on the current seat (== `player`).
    fn build_village(&self, g: &mut Game, player: PlayerId) -> bool {
        let cost = village_build_cost();
        if !s::affords(g, player, &cost, self.cfg.reserve) || !s::has_wood_buffer(g, player, &cost)
        {
            return false;
        }
        // First empty owned grassland that can host a building.
        let spot = m::owned_tiles(g, player).into_iter().find(|&t| {
            g.tiles[t.0].tile_type == TileType::Grassland
                && g.tiles[t.0].building.is_none()
                && g.buildable_buildings(t).contains(&"Village")
        });
        let spot = match spot {
            Some(t) => t,
            None => return false,
        };
        debug_assert_eq!(g.current_player(), player);
        let built = g.ai_build_building("Village", spot);
        if built && diag::on() {
            diag::inc(&diag::VILLAGE_OK);
        }
        if built {
            // The unit cap is cached (`free_unit_amount` reads `max_unit_amount`); refresh
            // it so the immediately-following `staff_income` can spend the +3 new slots.
            g.update_unit_amounts(player);
        }
        built
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
        self.staff_income_inner(g, player, true);
    }

    /// `staff_income` with an explicit `place_experts` toggle. With `place_experts =
    /// false` it performs every WORKER staffing (1st mine worker, 2nd mine worker,
    /// farm/plant/forest workers) but NEVER buys an Expert — so the CNN training
    /// scaffold can guarantee worker income up front while LEAVING the Expert
    /// (StackProducer:Expert) decision to the learned policy. The parity path
    /// (`plan_turn` via `staff_income`) always passes `true`, so it is byte-identical.
    fn staff_income_inner(&self, g: &mut Game, player: PlayerId, place_experts: bool) {
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

        // --- Pass 0: MINES + PLANTS first, to OPTIMAL (metal/energy fund the army) ---
        // Metal is the army bottleneck, so the scarce unit cap goes to mines/plants
        // BEFORE farms (money is rarely the constraint — see the metal-economy root
        // cause).
        //
        // Each mine -> 1 worker -> expert (doubles output) -> 2nd worker = 80 metal/round;
        // each plant -> worker + expert (else produces 0). Per-mine sequential so a mine
        // that already has a worker completes to its expert + 2nd worker before the cap is
        // spent elsewhere (full-staffing one mine = 80 metal beats two half-mines = 40+40).
        for tid in producers(g, &[BuildingType::Mine]) {
            self.ensure_worker(g, player, tid); // 1st worker
            if place_experts
                && self.cfg.experts
                && m::has_type(g, tid, UnitType::BasicWorker)
                && !m::has_type(g, tid, UnitType::Expert)
            {
                self.add_expert_reserve(g, player, tid, s::STAFF_RESERVE); // expert: ×2
            } else if place_experts
                && diag::on()
                && self.cfg.experts
                && !m::has_type(g, tid, UnitType::Expert)
            {
                diag::inc(&diag::MINE_EXPERT_GATE_SKIP);
            }
            if self.worker_count(g, tid) < 2 && g.tiles[tid.0].has_space_for_units() {
                self.add_worker(g, player, tid); // 2nd worker
            }
        }
        for tid in producers(g, &[BuildingType::Nuclear, BuildingType::Hydro]) {
            if place_experts && self.cfg.experts && !m::has_type(g, tid, UnitType::Expert) {
                self.add_expert_reserve(g, player, tid, s::STAFF_RESERVE);
            }
            if m::has_type(g, tid, UnitType::Expert)
                && !m::has_type(g, tid, UnitType::BasicWorker)
            {
                self.add_worker(g, player, tid);
            }
        }

        // --- Pass 1: minimum-viable staffing for the rest (each producer produces >0) -
        for tid in m::owned_tiles(g, player) {
            let ty = g.tiles[tid.0].building.as_ref().map(|b| b.kind);
            match ty {
                Some(BuildingType::Farm) => {
                    if !m::has_type(g, tid, UnitType::BasicWorker) {
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

    /// `ensure_unit_cap` must build Village(s) to expand the unit cap when it is the
    /// only thing blocking full producer staffing, so the (already-correct)
    /// `staff_income` can then place 2 workers + Expert on a mine. Regression guard for
    /// the unit-cap-starvation root cause: with only the HQ (+3 cap) the coverage pass
    /// exhausts the cap at 1 worker/producer, the Expert/2nd-worker upgrade never fires,
    /// and metal stays at ~20 (1/4 of the 80 optimum).
    #[test]
    fn ensure_unit_cap_builds_villages_to_fully_staff() {
        let mut g = Game::new(12, 12, &["P1", "P2"]);
        g.generate_map(12, 12, 1);
        let p1 = PlayerId(0);
        let id = |x: i32, y: i32| TileId((x * 12 + y) as usize);

        for x in 3..9 {
            for y in 3..9 {
                g.set_tile_owner(id(x, y), Some(p1));
            }
        }
        // ONLY the HQ (+3 cap). No villages — the AI must build them itself.
        g.place_building(id(5, 8), BuildingType::Headquarters, Some(p1));
        let mine = id(5, 5);
        let nuke = id(6, 6);
        g.place_building(mine, BuildingType::Mine, Some(p1));
        g.place_building(nuke, BuildingType::Nuclear, Some(p1));
        g.update_unit_amounts(p1);
        // Plenty of resources so only the unit cap (not affordability) limits us.
        g.set_player_resources(p1, 1_000_000, 1_000_000, 1_000_000, 1_000_000);

        let genome = Genome::zero(&DEFAULT_ARCH);
        let ctrl = NeuralAiController::new(&genome, TRAINING_CONFIG);

        // No village to start; cap is HQ-only (+3).
        let villages0 = m::owned_tiles(&g, p1)
            .into_iter()
            .filter(|&t| g.tiles[t.0].building.as_ref().map(|b| b.kind) == Some(BuildingType::Village))
            .count();
        assert_eq!(villages0, 0, "precondition: no villages");

        // Run the full scaffold loop several times (a real turn runs it repeatedly,
        // in the staff -> cap -> staff order).
        for _ in 0..6 {
            ctrl.staff_income(&mut g, p1);
            ctrl.ensure_unit_cap(&mut g, p1);
            ctrl.staff_income(&mut g, p1);
        }

        let villages = m::owned_tiles(&g, p1)
            .into_iter()
            .filter(|&t| g.tiles[t.0].building.as_ref().map(|b| b.kind) == Some(BuildingType::Village))
            .count();
        assert!(villages >= 1, "ensure_unit_cap should build at least one Village, got {villages}");

        // With the expanded cap the mine must reach optimal (2 workers + expert = 80).
        assert_eq!(ctrl.worker_count(&g, mine), 2, "mine should reach 2 workers");
        assert!(m::has_type(&g, mine, UnitType::Expert), "mine should get an Expert");
        assert_eq!(
            m::metal_income_per_round(&g, p1),
            80.0,
            "fully-staffed mine must yield 80 metal/round after cap expansion"
        );
        // Nuclear gets its worker + expert too.
        assert!(m::has_type(&g, nuke, UnitType::Expert), "nuclear should get an Expert");
        assert!(m::has_type(&g, nuke, UnitType::BasicWorker), "nuclear should get a worker");
    }
}
