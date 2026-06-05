//! Stage A test-time MCTS — PUCT search over per-candidate nodes, using the
//! existing trained policy net as a PRIOR and the deterministic `cp_sim::Game`
//! forward model for lookahead. NO retraining, NO new genome format.
//!
//! This module is OPT-IN: it is only reached when a [`SearchConfig`] is attached
//! to the controller. With search off, the controller path is byte-identical to
//! today, so the parity gate is untouched.
//!
//! Design (mirrors `MCTS-DESIGN.md`, Stage A):
//!
//! - **Node granularity = one candidate.** A node is a mid-turn state for the
//!   ROOT player. Its edges are the candidates returned by [`candidates::enumerate`]
//!   at that node, in enumerate() order. We do NOT store a `Game` per node; node
//!   state is re-derived by cloning the root `Game` and replaying the edge
//!   `Action`s down the path. (The root `Game` is captured mid-turn, after the
//!   safety scaffold has run, exactly where the controller would call
//!   `policy::select_index`.)
//! - **PUCT:** `argmax Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))`,
//!   `Q = W/N` (0 if unvisited).
//! - **Priors P(s,a):** softmax (temperature `tau_prior`) over the existing
//!   `policy::score_candidate` values for the node's candidates.
//! - **Leaf eval (configurable via [`LeafEval`]):** either a short rollout
//!   (horizon *rounds*) advancing ALL seats via `plan_turn` with search OFF +
//!   `end_turn`, OR a STATIC direct scoring of the leaf state (no turns). Both
//!   give exact ±1/0 on a terminal (rollout via `end_turn` Win/Tie; static via
//!   sole-survivor / dead-root in `metric_value`); otherwise a metric score
//!   (tile-frac lead vs mean opponent, net-income lead, wealth lead, soldier
//!   lead) mapped to [-1, 1], from the root player's perspective.
//! - **Opponents:** only the root player's candidate choices branch. When a
//!   simulated turn ends, every other seat is advanced as a forced deterministic
//!   transition (its `plan_turn` with search OFF + `end_turn`). Value is always
//!   from the root player's perspective.
//! - **Budget:** `n_sims` simulations per real decision, then return the
//!   MOST-VISITED root edge's candidate index (mirroring `policy::select_index`).

use crate::candidates::{self, Candidate, Intent};
use crate::controller::NeuralAiController;
use crate::features::global_features;
use crate::metrics as m;
use crate::policy::{self, XorShift32};
use crate::tiers::TierConfig;
use cp_sim::{EndTurnOutcome, Game, PlayerId};

/// Leaf-evaluation mode for the MCTS.
///
/// - [`LeafEval::Rollout`] (default, preserves Stage-A behaviour): play a short
///   multi-turn rollout from the leaf with all seats on the no-search policy,
///   then score the horizon state. Accurate but ~50-300× slower than static.
/// - [`LeafEval::Static`]: evaluate the leaf STATE DIRECTLY via the existing
///   relative metrics (no turns simulated). Exact ±1/0 if the leaf is already
///   terminal. Much faster — the leaf cost collapses to one `metric_value` call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeafEval {
    /// Short multi-turn rollout to `horizon` completed game-ROUNDS, then metric.
    Rollout { horizon: i64 },
    /// Direct static evaluation of the leaf state (no turns simulated).
    Static,
    /// **Stage B:** evaluate the leaf with a LEARNED VALUE NET (one forward pass
    /// over the leaf's 36-dim global features, already squashed to `[-1, 1]` via
    /// tanh). Terminal states still resolve to exact ±1/0 (handled in
    /// `metric_value`'s sole-survivor / dead-root branches). A `ValueNet` must be
    /// supplied via [`select_with_value`]; if none is present this falls back to
    /// [`LeafEval::Static`].
    Value,
}

/// Knobs for the test-time MCTS. Present only on the search path; `None` on the
/// controller means byte-identical-to-today behaviour.
#[derive(Debug, Clone, Copy)]
pub struct SearchConfig {
    /// Simulations per real decision. Most-visited root edge wins.
    pub n_sims: usize,
    /// PUCT exploration constant.
    pub c_puct: f64,
    /// Softmax temperature applied to `score_candidate` values to form priors.
    pub tau_prior: f64,
    /// Leaf-evaluation mode (rollout vs static). See [`LeafEval`].
    pub leaf_eval: LeafEval,
    /// Hard cap on game-rounds inside a rollout (safety against runaway games);
    /// the rollout also stops at `start_round + horizon`. Ignored for `Static`.
    pub round_cap: i64,
    /// Seed offset for the per-search RNG (rollouts/opponent turns run search
    /// OFF at temperature 0, so the RNG is never actually consumed; kept for
    /// faithful construction).
    pub seed: u32,
    /// Exp-I: use the SPATIAL policy input (`policy_spatial::policy_input_spatial`,
    /// genome arch `DEFAULT_ARCH_SPATIAL`) for MCTS priors + recorded self-play
    /// inputs, instead of the standard 63-dim input. Default `false` keeps the
    /// shipped/parity path byte-identical.
    pub spatial_policy: bool,
    /// AlphaZero ROOT exploration — **self-play only**. Symmetric Dirichlet(`alpha`)
    /// noise is mixed into the root priors with weight `dirichlet_eps`:
    /// `P'(a) = (1-eps)·P(a) + eps·eta_a`, so the search visits low-prior moves (the
    /// cure for prior-collapse-to-Pass). BOTH 0 = off. Consumed ONLY by
    /// [`select_with_pi`] (the data-collection path); [`select`]/[`select_with_value`]
    /// never read them, so the benchmark + shipped controller stay byte-identical
    /// and the parity gate is untouched.
    pub dirichlet_alpha: f64,
    pub dirichlet_eps: f64,
    /// Self-play played-move temperature: 0 = greedy argmax (today). `>0` samples the
    /// PLAYED move from `visit_counts^(1/temp)` while `round < temp_until_round`, then
    /// reverts to greedy. The recorded `pi` training target is ALWAYS the raw visit
    /// distribution — temperature only diversifies which move is *played* (→ diverse
    /// self-play positions), never the target.
    pub move_temperature: f64,
    pub temp_until_round: i64,
}

impl Default for SearchConfig {
    fn default() -> Self {
        SearchConfig {
            n_sims: 200,
            c_puct: 1.5,
            tau_prior: 1.0,
            leaf_eval: LeafEval::Rollout { horizon: 10 },
            round_cap: 400,
            seed: 0x5EA2C4,
            spatial_policy: false,
            dirichlet_alpha: 0.0,
            dirichlet_eps: 0.0,
            move_temperature: 0.0,
            temp_until_round: 0,
        }
    }
}

/// One MCTS node = a mid-turn state for the root player, reached by replaying
/// `path` (a sequence of edge indices) from the root `Game`.
struct Node {
    /// Edge indices from the root down to (and excluding) this node. Replaying
    /// these `Action`s from the root `Game` reconstructs this node's state.
    path: Vec<usize>,
    /// Cached mid-turn state at this node — captured at creation, BEFORE the
    /// `global_features` cap-refresh, so it equals the old `replay(path)` byte for
    /// byte. Lets simulations expand as `parent.game + one action` instead of
    /// replaying from the root each time (the speedup; search output is identical).
    game: Game,
    /// The candidates enumerated at this node (edges), in enumerate() order.
    candidates: Vec<Candidate>,
    /// Per-edge priors P(s,a) (softmax over score_candidate). Same length as
    /// `candidates`.
    priors: Vec<f64>,
    /// Per-edge child node index into the arena (None = not yet expanded).
    children: Vec<Option<usize>>,
    /// Per-edge visit count N(s,a).
    edge_visits: Vec<f64>,
    /// Per-edge total value W(s,a) (root-player perspective).
    edge_value: Vec<f64>,
    /// Total visits to this node N(s) = sum of edge_visits.
    visits: f64,
    /// True once this node has been visited at least once (leaf evaluated).
    expanded: bool,
    /// True if this node is terminal for the root player's turn (only a Pass
    /// candidate, or the game ended).
    terminal: bool,
}

struct Tree<'a> {
    nodes: Vec<Node>,
    genome: &'a crate::mlp::Genome,
    player: PlayerId,
    cfg: TierConfig,
    sc: SearchConfig,
    /// Optional learned value net for [`LeafEval::Value`] leaf evaluation. `None`
    /// for the rollout/static paths (parity-neutral; the value net never touches
    /// the policy genome).
    value_net: Option<&'a crate::value::ValueNet>,
}

impl<'a> Tree<'a> {
    /// Build a node for the given path, enumerating its candidates and priors.
    /// `g` must already be the state reached by replaying `path` from the root.
    /// `&mut Game` because `global_features` refreshes the unit caps (an internal
    /// mutation that does not affect the parity-relevant state fingerprint).
    fn make_node(&self, g: &mut Game, path: Vec<usize>) -> Node {
        // Snapshot BEFORE global_features (which refreshes caps) so the cache is
        // the exact replay(path) state.
        let game = g.clone();
        let round = g.get_rounds_played();
        let gvec = global_features(g, self.player, round);
        let candidates = candidates::enumerate(g, self.player, &self.cfg);
        let scores: Vec<f64> = if self.sc.spatial_policy {
            candidates
                .iter()
                .map(|c| crate::mlp::score(self.genome, &crate::policy_spatial::policy_input_spatial(g, self.player, &gvec, c)))
                .collect()
        } else {
            candidates
                .iter()
                .map(|c| policy::score_candidate(self.genome, &gvec, c))
                .collect()
        };
        let priors = softmax(&scores, self.sc.tau_prior);
        let n = candidates.len();
        // Terminal at this node if the only choice is Pass (turn effectively over)
        // or the game already ended (handled by the caller via live-player check).
        let terminal = candidates.iter().all(|c| c.intent == Intent::Pass);
        Node {
            path,
            game,
            candidates,
            priors,
            children: vec![None; n],
            edge_visits: vec![0.0; n],
            edge_value: vec![0.0; n],
            visits: 0.0,
            expanded: false,
            terminal,
        }
    }

}

/// PUCT edge selection: `argmax Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))`.
/// `Q = W/N` (0 if unvisited). Ties resolve to the LOWEST index (matching the
/// argmax convention in `policy::select_index`).
fn puct_select(node: &Node, c_puct: f64) -> usize {
    let sqrt_n = node.visits.max(0.0).sqrt();
    let mut best = 0usize;
    let mut best_score = f64::NEG_INFINITY;
    for a in 0..node.candidates.len() {
        let n_sa = node.edge_visits[a];
        let q = if n_sa > 0.0 {
            node.edge_value[a] / n_sa
        } else {
            0.0
        };
        let u = c_puct * node.priors[a] * sqrt_n / (1.0 + n_sa);
        let score = q + u;
        if score > best_score {
            best_score = score;
            best = a;
        }
    }
    best
}

/// Numerically-stable softmax with temperature `tau` over `scores`. Falls back to
/// a uniform distribution if `scores` is empty (cannot happen — Pass is always a
/// candidate) or `tau <= 0`.
fn softmax(scores: &[f64], tau: f64) -> Vec<f64> {
    let n = scores.len();
    if n == 0 {
        return Vec::new();
    }
    if tau <= 0.0 {
        // Degenerate: put all mass on the argmax (ties → lowest index).
        let mut best = 0usize;
        for i in 1..n {
            if scores[i] > scores[best] {
                best = i;
            }
        }
        let mut p = vec![0.0; n];
        p[best] = 1.0;
        return p;
    }
    let mut max = f64::NEG_INFINITY;
    for &s in scores {
        if s > max {
            max = s;
        }
    }
    let mut sum = 0.0;
    let mut p: Vec<f64> = scores
        .iter()
        .map(|&s| {
            let e = ((s - max) / tau).exp();
            sum += e;
            e
        })
        .collect();
    if sum > 0.0 {
        for v in &mut p {
            *v /= sum;
        }
    } else {
        let u = 1.0 / n as f64;
        for v in &mut p {
            *v = u;
        }
    }
    p
}

/// Standard-normal sample via Box–Muller from a uniform `Rng` (`next_f64` ∈ [0,1)).
fn sample_normal<R: crate::policy::Rng>(rng: &mut R) -> f64 {
    let u1 = rng.next_f64().max(1e-12);
    let u2 = rng.next_f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Gamma(shape = `alpha`, scale = 1) via Marsaglia–Tsang, with the α<1 boost
/// (`Gamma(α) = Gamma(α+1)·U^(1/α)`). Used to build a Dirichlet sample.
fn sample_gamma<R: crate::policy::Rng>(rng: &mut R, alpha: f64) -> f64 {
    if alpha < 1.0 {
        let u = rng.next_f64().max(1e-12);
        return sample_gamma(rng, alpha + 1.0) * u.powf(1.0 / alpha);
    }
    let d = alpha - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = sample_normal(rng);
        let v0 = 1.0 + c * x;
        if v0 <= 0.0 {
            continue;
        }
        let v = v0 * v0 * v0;
        let u = rng.next_f64();
        if u < 1.0 - 0.0331 * x * x * x * x {
            return d * v;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

/// Symmetric Dirichlet(`alpha`) sample of length `n` (normalized i.i.d. Gamma draws).
fn sample_dirichlet<R: crate::policy::Rng>(rng: &mut R, alpha: f64, n: usize) -> Vec<f64> {
    let mut g: Vec<f64> = (0..n).map(|_| sample_gamma(rng, alpha).max(1e-12)).collect();
    let s: f64 = g.iter().sum();
    if s > 0.0 {
        for v in &mut g {
            *v /= s;
        }
    } else {
        for v in &mut g {
            *v = 1.0 / n as f64;
        }
    }
    g
}

/// Most-visited edge (ties → lowest index), the greedy played-move rule.
fn argmax_visits(visits: &[f64]) -> usize {
    let mut best = 0usize;
    let mut best_n = f64::NEG_INFINITY;
    for (a, &v) in visits.iter().enumerate() {
        if v > best_n {
            best_n = v;
            best = a;
        }
    }
    best
}

/// Sample a played move ∝ `visits^(1/temp)` (AlphaZero temperature selection).
/// Falls back to [`argmax_visits`] if the weights are degenerate.
fn sample_move<R: crate::policy::Rng>(visits: &[f64], temp: f64, rng: &mut R) -> usize {
    let inv = 1.0 / temp;
    let w: Vec<f64> = visits.iter().map(|&v| if v > 0.0 { v.powf(inv) } else { 0.0 }).collect();
    let s: f64 = w.iter().sum();
    if !(s > 0.0) {
        return argmax_visits(visits);
    }
    let mut r = rng.next_f64() * s;
    for (i, &wi) in w.iter().enumerate() {
        r -= wi;
        if r <= 0.0 {
            return i;
        }
    }
    w.len() - 1
}

/// Re-staff after a candidate executes, mirroring the controller's per-iteration
/// `staff_income` call. Implemented via a throwaway controller so we reuse the
/// EXACT scaffold logic (which is private to the controller).
fn staff_after_action(g: &mut Game, player: PlayerId, cfg: &TierConfig, genome: &crate::mlp::Genome) {
    let ctrl = NeuralAiController::new(genome, *cfg);
    ctrl.staff_income_pub(g, player);
}

/// Advance every NON-root seat by one forced deterministic turn (its `plan_turn`
/// with search OFF + `end_turn`), then `end_turn` for the root if it is the root
/// player's turn boundary. Returns the terminal outcome if the game ended.
///
/// Concretely: the root player's mid-turn decisions have all been replayed into
/// `g` already; this finishes the root player's turn (`end_turn`) and then plays
/// out one full turn for each remaining live seat until it is the root player's
/// turn again (or the game ends).
fn advance_round_after_root_turn(
    g: &mut Game,
    root_player: PlayerId,
    cfg: &TierConfig,
    genome: &crate::mlp::Genome,
    rng: &mut XorShift32,
) -> Option<EndTurnOutcome> {
    let ctrl = NeuralAiController::new(genome, *cfg);
    // 1. End the root player's turn.
    match g.end_turn() {
        EndTurnOutcome::Win(p) => return Some(EndTurnOutcome::Win(p)),
        EndTurnOutcome::Tie => return Some(EndTurnOutcome::Tie),
        _ => {}
    }
    // 2. Play every other live seat until it is the root player's turn again.
    while g.live_players().len() > 1 {
        let cur = g.current_player();
        if cur == root_player {
            break;
        }
        ctrl.plan_turn(g, cur, rng, None); // search OFF (no recursion)
        match g.end_turn() {
            EndTurnOutcome::Win(p) => return Some(EndTurnOutcome::Win(p)),
            EndTurnOutcome::Tie => return Some(EndTurnOutcome::Tie),
            _ => {}
        }
    }
    None
}

/// Map a non-terminal leaf `Game` state to a value in [-1, 1] from `root_player`'s
/// perspective, by blending four relative leads vs the MEAN of living opponents:
/// tile fraction, net income, total wealth, and soldier count. Reuses the exact
/// telemetry metrics `run.rs` uses. Each lead is normalized to ~1.0 for a
/// "meaningful" advantage, clamped, then averaged.
fn metric_value(g: &Game, root_player: PlayerId) -> f64 {
    // If the root player is dead, that's a loss.
    let alive = g.live_players().iter().any(|&p| p == root_player);
    if !alive {
        return -1.0;
    }
    let opp: Vec<PlayerId> = g
        .live_players()
        .iter()
        .copied()
        .filter(|&p| p != root_player)
        .collect();
    if opp.is_empty() {
        // Root player is the only one left → effectively won.
        return 1.0;
    }
    let k = opp.len() as f64;
    let total = g.get_tile_count().max(1) as f64;

    let my_tiles = g.get_tile_count_for_player(root_player) as f64;
    let opp_tiles = opp
        .iter()
        .map(|&p| g.get_tile_count_for_player(p) as f64)
        .sum::<f64>()
        / k;
    // Tile-fraction lead: difference of fractions, where a ~1/3-of-map lead ≈ 1.0.
    let tile_lead = ((my_tiles - opp_tiles) / total / 0.33).clamp(-1.0, 1.0);

    let my_income = m::net_money_per_round(g, root_player);
    let opp_income = opp.iter().map(|&p| m::net_money_per_round(g, p)).sum::<f64>() / k;
    let income_lead = ((my_income - opp_income) / 200.0).clamp(-1.0, 1.0);

    let my_wealth = m::total_wealth(g, root_player);
    let opp_wealth = opp.iter().map(|&p| m::total_wealth(g, p)).sum::<f64>() / k;
    let wealth_lead = ((my_wealth - opp_wealth) / 2000.0).clamp(-1.0, 1.0);

    let my_sol = g.current_soldier_amount(root_player) as f64;
    let opp_sol = opp
        .iter()
        .map(|&p| g.current_soldier_amount(p) as f64)
        .sum::<f64>()
        / k;
    let mil_lead = ((my_sol - opp_sol) / 5.0).clamp(-1.0, 1.0);

    // Tile control dominates the win condition; weight it highest.
    let v = 0.5 * tile_lead + 0.2 * income_lead + 0.2 * wealth_lead + 0.1 * mil_lead;
    v.clamp(-1.0, 1.0)
}

/// [`LeafEval::Value`] leaf evaluation: a single forward pass of the learned
/// value net over the leaf's 36-dim global features for the root player.
///
/// Terminal states (root is the sole survivor, or has been eliminated) resolve
/// to the EXACT ±1 via `metric_value`'s terminal branches — we never ask the net
/// to predict a known outcome. For non-terminal leaves the net's output is
/// already in `[-1, 1]` (tanh output). If no value net is supplied we fall back
/// to the hand-crafted [`metric_value`] (== `LeafEval::Static`).
fn value_leaf(
    leaf: &Game,
    root_player: PlayerId,
    value_net: Option<&crate::value::ValueNet>,
) -> f64 {
    // Resolve exact terminals first (cheap; no net call needed).
    let alive = leaf.live_players().iter().any(|&p| p == root_player);
    if !alive {
        return -1.0;
    }
    if leaf.live_players().len() <= 1 {
        return 1.0; // root is the sole survivor
    }
    match value_net {
        Some(net) => {
            let mut g = leaf.clone();
            let round = g.get_rounds_played();
            // Auto-select the feature width from the net: plain 36-dim global, or
            // the 41-dim spatial value features (global + board summaries).
            let gvec = if net.arch[0] == crate::features::GLOBAL_DIM {
                global_features(&mut g, root_player, round)
            } else {
                crate::features::value_features_spatial(&mut g, root_player, round)
            };
            net.forward(&gvec) // already in [-1, 1] (tanh output)
        }
        // No net → behave like Static so the call still produces a sane value.
        None => metric_value(leaf, root_player),
    }
}

/// Leaf evaluation. Dispatches on `sc.leaf_eval`:
///
/// - [`LeafEval::Static`]: evaluate the leaf STATE DIRECTLY (no turns simulated).
///   If the leaf is already terminal (root player is the only seat left, or has
///   been eliminated), `metric_value` returns the exact ±1; otherwise it returns
///   the blended relative-metric value in [-1, 1] from the root's perspective.
/// - [`LeafEval::Rollout { horizon }`]: from the leaf `Game` (the root player's
///   mid-turn state with the leaf's edges replayed), play a short rollout
///   advancing ALL seats via `plan_turn` (search OFF) + `end_turn` until
///   `horizon` completed rounds. Exact ±1/0 on a Win/Tie terminal; otherwise the
///   metric value of the horizon state.
fn evaluate_leaf(
    leaf: &Game,
    root_player: PlayerId,
    cfg: &TierConfig,
    genome: &crate::mlp::Genome,
    sc: &SearchConfig,
    value_net: Option<&crate::value::ValueNet>,
    rng: &mut XorShift32,
) -> f64 {
    let horizon = match sc.leaf_eval {
        // Static: score the mid-turn leaf state as-is. No clone, no turns. The
        // terminal case (sole survivor / dead root) is handled inside
        // `metric_value` and resolves to an exact ±1.
        LeafEval::Static => return metric_value(leaf, root_player),
        // Value: learned value net forward pass over the leaf's global features.
        // Exact terminal verdict first (sole survivor / dead root); otherwise the
        // net's tanh output in [-1, 1]. Falls back to Static if no net supplied.
        LeafEval::Value => return value_leaf(leaf, root_player, value_net),
        LeafEval::Rollout { horizon } => horizon,
    };

    let mut g = leaf.clone();
    let ctrl = NeuralAiController::new(genome, *cfg);
    let start_round = g.get_rounds_played();
    let horizon = (start_round + horizon).min(sc.round_cap);

    // The leaf is mid-turn for the root player; finish the root player's turn and
    // step every other seat to bring us back to the root player's turn.
    if let Some(out) = advance_round_after_root_turn(&mut g, root_player, cfg, genome, rng) {
        return terminal_value(&g, root_player, out);
    }

    // Now play full rounds (root + opponents) until the horizon or game end.
    while g.live_players().len() > 1 && g.get_rounds_played() < horizon {
        let cur = g.current_player();
        debug_assert_eq!(cur, root_player);
        ctrl.plan_turn(&mut g, cur, rng, None); // root player, search OFF
        if let Some(out) = advance_round_after_root_turn(&mut g, root_player, cfg, genome, rng) {
            return terminal_value(&g, root_player, out);
        }
    }
    metric_value(&g, root_player)
}

/// Exact terminal value from the root player's perspective.
fn terminal_value(g: &Game, root_player: PlayerId, out: EndTurnOutcome) -> f64 {
    match out {
        EndTurnOutcome::Win(p) => {
            if p == root_player {
                1.0
            } else {
                -1.0
            }
        }
        EndTurnOutcome::Tie => 0.0,
        // Shouldn't be passed here, but score the live state defensively.
        _ => metric_value(g, root_player),
    }
}

/// One MCTS simulation: descend from the root via PUCT until reaching an
/// unexpanded node (or a terminal), expand it, evaluate it, and back up the value
/// along the visited path. Returns the leaf value (root-player perspective).
fn simulate(tree: &mut Tree, _root_game: &Game, rng: &mut XorShift32) -> f64 {
    // Selection: walk down, recording (node, edge) pairs, until we hit a node we
    // need to expand/evaluate.
    let mut visited: Vec<(usize, usize)> = Vec::new();
    let mut node = 0usize;

    loop {
        // A terminal node (only Pass / no opponents left) is evaluated directly.
        if tree.nodes[node].terminal {
            break;
        }
        // First visit to this node → it's a leaf: stop and evaluate.
        if !tree.nodes[node].expanded {
            break;
        }
        let edge = puct_select(&tree.nodes[node], tree.sc.c_puct);
        visited.push((node, edge));
        match tree.nodes[node].children[edge] {
            Some(child) => {
                node = child;
            }
            None => {
                // Expand this edge: build the child node from the replayed state.
                let mut path = tree.nodes[node].path.clone();
                path.push(edge);
                let mut child_game = {
                    // Apply the edge to the parent's CACHED state, then re-staff
                    // (== old replay(child), but O(1) action instead of replay-from-root).
                    let mut g = tree.nodes[node].game.clone();
                    let action = tree.nodes[node].candidates[edge].action.clone();
                    let _ = candidates::execute_action(&mut g, tree.player, &tree.cfg, &action);
                    staff_after_action(&mut g, tree.player, &tree.cfg, tree.genome);
                    g
                };
                let child = tree.make_node(&mut child_game, path);
                tree.nodes.push(child);
                let child_idx = tree.nodes.len() - 1;
                tree.nodes[node].children[edge] = Some(child_idx);
                node = child_idx;
                break;
            }
        }
    }

    // Evaluation of the reached leaf/terminal node: roll the game out (from this
    // mid-turn state) to an exact verdict or the metric horizon. The node's cached
    // state is byte-identical to the old replay(node).
    let leaf_game = tree.nodes[node].game.clone();
    let value = evaluate_leaf(
        &leaf_game,
        tree.player,
        &tree.cfg,
        tree.genome,
        &tree.sc,
        tree.value_net,
        rng,
    );
    tree.nodes[node].expanded = true;
    tree.nodes[node].visits += 1.0;

    // Backup along the visited edges.
    for &(n, e) in &visited {
        tree.nodes[n].edge_visits[e] += 1.0;
        tree.nodes[n].edge_value[e] += value;
        tree.nodes[n].visits += 1.0;
    }
    value
}

/// Run test-time MCTS for the current mid-turn decision and return the chosen
/// candidate index (into `enumerate(g, player, cfg)` at the CURRENT state — the
/// same indexing `policy::select_index` returns). `g` is the live mid-turn game
/// (after the safety scaffold / prior executes); it is NOT mutated (we clone it
/// internally for branching).
pub fn select(
    genome: &crate::mlp::Genome,
    g: &Game,
    player: PlayerId,
    cfg: &TierConfig,
    sc: &SearchConfig,
) -> usize {
    select_with_value(genome, g, player, cfg, sc, None)
}

/// Like [`select`], but with an optional learned value net used by
/// [`LeafEval::Value`] for leaf evaluation. For all other leaf modes the value
/// net is ignored. This is the Stage-B entry point.
pub fn select_with_value(
    genome: &crate::mlp::Genome,
    g: &Game,
    player: PlayerId,
    cfg: &TierConfig,
    sc: &SearchConfig,
    value_net: Option<&crate::value::ValueNet>,
) -> usize {
    // Capture the root mid-turn state.
    let root_game = g.clone();
    let mut tree = Tree {
        nodes: Vec::new(),
        genome,
        player,
        cfg: *cfg,
        sc: *sc,
        value_net,
    };
    let mut root_for_enum = root_game.clone();
    let root = tree.make_node(&mut root_for_enum, Vec::new());
    let n_cands = root.candidates.len();
    tree.nodes.push(root);

    if n_cands <= 1 {
        return 0;
    }

    let mut rng = XorShift32::new(sc.seed ^ (g.get_rounds_played() as u32).wrapping_mul(2654435761));

    for _ in 0..sc.n_sims {
        simulate(&mut tree, &root_game, &mut rng);
    }

    // Return the MOST-VISITED root edge (ties → lowest index, matching argmax).
    let root_node = &tree.nodes[0];
    let mut best = 0usize;
    let mut best_n = -1.0f64;
    for a in 0..root_node.candidates.len() {
        if root_node.edge_visits[a] > best_n {
            best_n = root_node.edge_visits[a];
            best = a;
        }
    }
    best
}

/// Result of a search used for AlphaZero self-play data collection: the chosen
/// candidate index, the MCTS visit-count policy target `pi` over the root
/// candidates (normalised, Σ=1), and the per-candidate policy network inputs at
/// the root (so a training example needs no re-enumeration).
pub struct SearchResult {
    pub chosen: usize,
    pub pi: Vec<f64>,
    pub policy_inputs: Vec<Vec<f64>>,
}

/// Like [`select_with_value`], but also returns the root visit distribution `pi`
/// and the per-candidate policy inputs — the AlphaZero policy target + features.
/// The chosen index matches [`select_with_value`] exactly (most-visited root
/// edge); this is the data-collection entry point for self-play training.
pub fn select_with_pi<R: crate::policy::Rng>(
    genome: &crate::mlp::Genome,
    g: &Game,
    player: PlayerId,
    cfg: &TierConfig,
    sc: &SearchConfig,
    value_net: Option<&crate::value::ValueNet>,
    rng_ext: &mut R,
) -> SearchResult {
    let root_game = g.clone();
    let mut tree = Tree {
        nodes: Vec::new(),
        genome,
        player,
        cfg: *cfg,
        sc: *sc,
        value_net,
    };
    let mut root_for_enum = root_game.clone();
    let root = tree.make_node(&mut root_for_enum, Vec::new());
    tree.nodes.push(root);

    // Per-candidate policy inputs at the root (global features + intent + local).
    let mut gclone = root_game.clone();
    let round = gclone.get_rounds_played();
    let gvec = global_features(&mut gclone, player, round);
    let policy_inputs: Vec<Vec<f64>> = tree.nodes[0]
        .candidates
        .iter()
        .map(|c| {
            if sc.spatial_policy {
                crate::policy_spatial::policy_input_spatial(&gclone, player, &gvec, c)
            } else {
                crate::policy::policy_input(&gvec, c)
            }
        })
        .collect();

    let n_cands = tree.nodes[0].candidates.len();
    if n_cands <= 1 {
        // Trivial decision: no search signal. One-hot pi on the only candidate.
        let mut pi = vec![0.0; n_cands];
        if n_cands == 1 {
            pi[0] = 1.0;
        }
        return SearchResult { chosen: 0, pi, policy_inputs };
    }

    // AlphaZero ROOT exploration: perturb the root priors with Dirichlet noise so
    // the search actually visits low-prior moves (Expand/Attack), instead of
    // collapsing all visits onto the highest-prior move (Pass). Self-play only —
    // this is the `select_with_pi` (data-collection) path; the deterministic
    // inference path (`select_with_value`) never reaches here, so parity is safe.
    if sc.dirichlet_alpha > 0.0 && sc.dirichlet_eps > 0.0 && n_cands > 1 {
        let noise = sample_dirichlet(rng_ext, sc.dirichlet_alpha, n_cands);
        let eps = sc.dirichlet_eps;
        let priors = &mut tree.nodes[0].priors;
        for a in 0..n_cands {
            priors[a] = (1.0 - eps) * priors[a] + eps * noise[a];
        }
    }

    let mut rng = XorShift32::new(sc.seed ^ (g.get_rounds_played() as u32).wrapping_mul(2654435761));
    for _ in 0..sc.n_sims {
        simulate(&mut tree, &root_game, &mut rng);
    }

    let edge_visits = tree.nodes[0].edge_visits.clone();
    let total: f64 = edge_visits.iter().sum();
    // The TRAINING target pi is always the raw visit distribution (never temperature-
    // adjusted) — temperature only affects which move is PLAYED below.
    let pi: Vec<f64> = if total > 0.0 {
        edge_visits.iter().map(|&v| v / total).collect()
    } else {
        let mut p = vec![0.0; n_cands];
        p[0] = 1.0;
        p
    };
    // Played-move selection: sample ∝ visits^(1/temp) early in the game (exploration
    // → diverse self-play positions), greedy argmax later. Default temp 0 = argmax,
    // identical to the old behaviour.
    let round = root_game.get_rounds_played();
    let chosen = if sc.move_temperature > 1e-9 && round < sc.temp_until_round {
        sample_move(&edge_visits, sc.move_temperature, rng_ext)
    } else {
        argmax_visits(&edge_visits)
    };
    SearchResult { chosen, pi, policy_inputs }
}
