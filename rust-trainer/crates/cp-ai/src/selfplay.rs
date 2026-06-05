//! AlphaZero self-play data generation: play one full game where every seat is
//! the same MCTS controller, recording each decision's (policy inputs, MCTS
//! visit-count target π, global features) and, at game end, the outcome z from
//! each deciding seat's perspective. Produces ready-to-train policy + value
//! examples. Mirrors `run::run_game`'s loop (HQ placement → plan_turn → end_turn)
//! but via `controller::plan_turn_record`.
//!
//! Additive: not on the parity path.

use cp_sim::{BuildingType, EndTurnOutcome, Game, PlayerId};

use crate::controller::{NeuralAiController, RecordedDecision};
use crate::hard_ai::HardAi;
use crate::mlp::Genome;
use crate::policy::XorShift32;
use crate::policy_train::PolicyExample;
use crate::search::SearchConfig;
use crate::tiers::TierConfig;
use crate::value::{ValueExample, ValueNet};

/// Training examples harvested from one self-play game.
pub struct SelfPlayData {
    pub policy: Vec<PolicyExample>,
    pub value: Vec<ValueExample>,
}

/// End a self-play game once the BOARD has not changed for this many rounds (and no
/// Device is counting down) — a frozen game where both sides just Pass. Recording
/// those hundreds of Pass turns floods the policy target with "Pass is correct" and
/// drags the net back to passivity (the recurring collapse). 40 sits well above any
/// non-Device static stretch (the Device countdown is ≤~30 and is exempted anyway)
/// but far below the 150–240-round freezes observed.
pub const STALL_ROUNDS: i64 = 40;

/// Cheap hash of the board state that matters for "is the game progressing": every
/// tile's (owner, building-kind) plus each seat's soldier count. Two identical
/// signatures N rounds apart ⇒ nothing was expanded, built, or fought.
pub fn board_signature(g: &Game, n_players: usize) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |v: u64| {
        h ^= v;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    };
    for (i, t) in g.get_tiles().iter().enumerate() {
        let o = match t.owner { Some(p) => p.0 as u64 + 1, None => 0 };
        let b = match &t.building { Some(bb) => bb.kind as u64 + 1, None => 0 };
        mix(((i as u64) << 16) | (o << 8) | b);
    }
    for s in 0..n_players {
        mix(g.current_soldier_amount(PlayerId(s)) as u64);
    }
    h
}

/// True if a Strange Device stands anywhere — its countdown is an active win attempt,
/// so such a game must NEVER be cut as a stalemate even though the board is static.
pub fn device_on_board(g: &Game) -> bool {
    g.get_tiles()
        .iter()
        .any(|t| matches!(&t.building, Some(b) if b.kind == BuildingType::StrangeDevice))
}

/// Play one self-play game (all seats = the same net + search) and return the
/// policy/value training examples. `z` is +1 for the winner's decisions, -1 for
/// the losers', 0 on tie/timeout (the AlphaZero terminal signal; dense shaping is
/// layered on later by the trainer).
#[allow(clippy::too_many_arguments)]
pub fn play_one_game(
    seed: u32,
    width: i32,
    height: i32,
    n_players: usize,
    genome: &Genome,
    cfg: &TierConfig,
    sc: &SearchConfig,
    value_net: Option<&ValueNet>,
    round_cap: i64,
    spatial_value: bool,
    timeout_penalty: f64,
    win_speed: f64,
) -> SelfPlayData {
    let names: Vec<String> = (0..n_players).map(|i| format!("P{}", i + 1)).collect();
    let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    let mut g = Game::new(width, height, &name_refs);
    g.generate_map(width, height, seed);

    let ctrl = match value_net {
        Some(vn) => NeuralAiController::with_search_value(genome, *cfg, *sc, vn),
        None => NeuralAiController::with_search(genome, *cfg, *sc),
    };
    let mut rng = XorShift32::new(seed);

    // Round 0: HQ placement for every seat.
    for _ in 0..n_players {
        let cur = g.current_player();
        ctrl.place_headquarters(&mut g, cur);
        g.change_turn();
    }

    let mut pending: Vec<RecordedDecision> = Vec::new();
    let mut winner: Option<PlayerId> = None;
    let mut last_sig = board_signature(&g, n_players);
    let mut last_progress_round = g.get_rounds_played();
    while g.live_players().len() > 1 && g.get_rounds_played() < round_cap {
        let cur = g.current_player();
        ctrl.plan_turn_record(&mut g, cur, &mut rng, &mut |d| pending.push(d));
        match g.end_turn() {
            EndTurnOutcome::Win(p) => {
                winner = Some(p);
                break;
            }
            EndTurnOutcome::Tie => break,
            _ => {}
        }
        // Stalemate cut: a frozen game (no board change for STALL_ROUNDS, no Device
        // counting down) ends now as a tie instead of recording hundreds of Pass turns.
        let r = g.get_rounds_played();
        let sig = board_signature(&g, n_players);
        if sig != last_sig {
            last_sig = sig;
            last_progress_round = r;
        } else if r - last_progress_round >= STALL_ROUNDS && !device_on_board(&g) {
            break;
        }
    }

    // Decisive winner: explicit Win, else sole survivor; otherwise none (tie/timeout → z=0).
    let winner_pid = winner.or_else(|| {
        let live = g.live_players();
        if live.len() == 1 {
            Some(live[0])
        } else {
            None
        }
    });
    // Decisiveness reward:
    //  - win: (1-win_speed) + win_speed*(1 - rounds/cap) → faster kill = closer to +1.
    //  - loss: -1.
    //  - timeout/tie: -timeout_penalty (a draw is worse than not-losing → push to convert).
    // All stay within [-1,1] (the value head's tanh range).
    let rounds = g.get_rounds_played() as f64;
    let cap = round_cap.max(1) as f64;
    let win_z = (1.0 - win_speed) + win_speed * (1.0 - rounds / cap).max(0.0);
    let z_of = |p: PlayerId| -> f64 {
        match winner_pid {
            Some(w) if w == p => win_z,
            Some(_) => -1.0,
            None => -timeout_penalty,
        }
    };

    let mut data = SelfPlayData { policy: Vec::new(), value: Vec::new() };
    for d in pending {
        let z = z_of(d.player);
        let x = if spatial_value { d.value_vec } else { d.global_vec };
        data.value.push(ValueExample { x, z });
        data.policy.push(PolicyExample { inputs: d.policy_inputs, pi: d.pi });
    }
    data
}

/// Play one game where seat 0 is our MCTS net (recorded) and seat 1 is the
/// held-out HARD heuristic. Only seat-0 decisions are harvested — this is the
/// "main exploiter" lever (AlphaStar): self-play only teaches beating *yourself*;
/// mixing in games vs hard teaches beating *its* style. `z` is from seat 0's
/// perspective with the same decisiveness reward as `play_one_game`.
///
/// Additive: not on the parity path. Mirrors `bench_vs_hard`'s game loop but via
/// `plan_turn_record` for seat 0 so MCTS decisions become training examples.
#[allow(clippy::too_many_arguments)]
pub fn play_one_game_vs_hard(
    seed: u32,
    width: i32,
    height: i32,
    genome: &Genome,
    cfg: &TierConfig,
    sc: &SearchConfig,
    value_net: Option<&ValueNet>,
    round_cap: i64,
    spatial_value: bool,
    timeout_penalty: f64,
    win_speed: f64,
) -> SelfPlayData {
    let mut g = Game::new(width, height, &["P1", "P2"]);
    g.generate_map(width, height, seed);

    let champ = match value_net {
        Some(vn) => NeuralAiController::with_search_value(genome, *cfg, *sc, vn),
        None => NeuralAiController::with_search(genome, *cfg, *sc),
    };
    let mut hard = HardAi::hard();
    let mut rng = XorShift32::new(seed);

    // Round 0: HQ placement (seat 0 = our net, seat 1 = hard).
    for _ in 0..2 {
        let cur = g.current_player();
        if cur.0 == 0 { champ.place_headquarters(&mut g, cur); } else { hard.place_headquarters(&mut g, cur); }
        g.change_turn();
    }

    let mut pending: Vec<RecordedDecision> = Vec::new();
    let mut winner: Option<PlayerId> = None;
    let mut last_sig = board_signature(&g, 2);
    let mut last_progress_round = g.get_rounds_played();
    while g.live_players().len() > 1 && g.get_rounds_played() < round_cap {
        let cur = g.current_player();
        if cur.0 == 0 {
            champ.plan_turn_record(&mut g, cur, &mut rng, &mut |d| pending.push(d));
        } else {
            hard.plan_turn(&mut g, cur);
        }
        match g.end_turn() {
            EndTurnOutcome::Win(p) => { winner = Some(p); break; }
            EndTurnOutcome::Tie => break,
            _ => {}
        }
        let r = g.get_rounds_played();
        let sig = board_signature(&g, 2);
        if sig != last_sig {
            last_sig = sig;
            last_progress_round = r;
        } else if r - last_progress_round >= STALL_ROUNDS && !device_on_board(&g) {
            break;
        }
    }

    let winner_pid = winner.or_else(|| {
        let live = g.live_players();
        if live.len() == 1 { Some(live[0]) } else { None }
    });
    // Same decisiveness reward as play_one_game, but seat-0-centric (all recorded
    // decisions belong to seat 0, so z is win/loss/timeout for our net).
    let rounds = g.get_rounds_played() as f64;
    let cap = round_cap.max(1) as f64;
    let win_z = (1.0 - win_speed) + win_speed * (1.0 - rounds / cap).max(0.0);
    let z = match winner_pid {
        Some(w) if w.0 == 0 => win_z,
        Some(_) => -1.0,
        None => -timeout_penalty,
    };

    let mut data = SelfPlayData { policy: Vec::new(), value: Vec::new() };
    for d in pending {
        let x = if spatial_value { d.value_vec } else { d.global_vec };
        data.value.push(ValueExample { x, z });
        data.policy.push(PolicyExample { inputs: d.policy_inputs, pi: d.pi });
    }
    data
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::DEFAULT_ARCH;
    use crate::policy_train::random_genome;
    use crate::search::{LeafEval, SearchConfig};
    use crate::tiers::TRAINING_CONFIG;

    #[test]
    fn self_play_produces_consistent_examples() {
        let genome = random_genome(&DEFAULT_ARCH.to_vec(), 7);
        let sc = SearchConfig {
            n_sims: 12,
            leaf_eval: LeafEval::Static,
            ..Default::default()
        };
        let data = play_one_game(123, 10, 10, 2, &genome, &TRAINING_CONFIG, &sc, None, 30, false, 0.0, 0.0);
        // A game should yield some decisions.
        assert!(!data.policy.is_empty(), "no policy examples harvested");
        assert_eq!(data.policy.len(), data.value.len());
        for ex in &data.policy {
            // pi is a distribution over >=2 candidates, summing to ~1.
            assert!(ex.inputs.len() >= 2);
            assert_eq!(ex.inputs.len(), ex.pi.len());
            let s: f64 = ex.pi.iter().sum();
            assert!((s - 1.0).abs() < 1e-6, "pi sum {s}");
        }
        for ex in &data.value {
            assert!(ex.z == 1.0 || ex.z == -1.0 || ex.z == 0.0);
        }
    }

    #[test]
    fn vs_hard_self_play_records_only_seat0() {
        let genome = random_genome(&DEFAULT_ARCH.to_vec(), 7);
        let sc = SearchConfig { n_sims: 12, leaf_eval: LeafEval::Static, ..Default::default() };
        let data = play_one_game_vs_hard(123, 10, 10, &genome, &TRAINING_CONFIG, &sc, None, 30, false, 0.3, 0.3);
        assert!(!data.policy.is_empty(), "no policy examples harvested vs hard");
        assert_eq!(data.policy.len(), data.value.len());
        // All recorded value targets share one seat-0 outcome z (decisiveness reward),
        // so every example carries the identical z for a single game.
        let z0 = data.value[0].z;
        for ex in &data.value {
            assert_eq!(ex.z, z0, "vs-hard z must be uniform (seat-0 outcome)");
            assert!((-1.0..=1.0).contains(&ex.z));
        }
        for ex in &data.policy {
            assert_eq!(ex.inputs.len(), ex.pi.len());
            let s: f64 = ex.pi.iter().sum();
            assert!((s - 1.0).abs() < 1e-6, "pi sum {s}");
        }
    }
}
