//! Smoke tests for the HARD heuristic port (`cp_ai::HardAi`).
//!
//! HardAi is a benchmark opponent, not on the parity path, so these only assert
//! the things that matter for it being a valid opponent: it places an HQ, plays
//! full turns without crashing, is deterministic for fixed inputs, and actually
//! builds an economy (grows past its starting tiles).

use cp_ai::HardAi;
use cp_sim::{BuildingType, EndTurnOutcome, Game, PlayerId};

/// Run a full HardAi-vs-HardAi game to completion; return (rounds, p1_tiles,
/// p2_tiles, finished_cleanly).
fn run_hard_vs_hard(seed: u32, w: i32, h: i32, cap: i64) -> (i64, i64, i64, bool) {
    let mut g = Game::new(w, h, &["P1", "P2"]);
    g.generate_map(w, h, seed);
    let mut a = HardAi::hard();
    let mut b = HardAi::hard();

    for _ in 0..2 {
        let cur = g.current_player();
        if cur.0 == 0 {
            a.place_headquarters(&mut g, cur);
        } else {
            b.place_headquarters(&mut g, cur);
        }
        g.change_turn();
    }

    let mut finished = false;
    while g.live_players().len() > 1 && g.get_rounds_played() < cap {
        let cur = g.current_player();
        if cur.0 == 0 {
            a.plan_turn(&mut g, cur);
        } else {
            b.plan_turn(&mut g, cur);
        }
        match g.end_turn() {
            EndTurnOutcome::Win(_) | EndTurnOutcome::Tie => {
                finished = true;
                break;
            }
            _ => {}
        }
    }
    (
        g.get_rounds_played(),
        g.get_tile_count_for_player(PlayerId(0)),
        g.get_tile_count_for_player(PlayerId(1)),
        finished,
    )
}

#[test]
fn hard_ai_plays_a_full_game_without_crashing() {
    // A handful of seeds/sizes; just must not panic and must run.
    for &(seed, w, h) in &[(1u32, 12, 12), (42, 14, 12), (99, 16, 14), (7, 18, 14)] {
        let (rounds, _t1, _t2, _fin) = run_hard_vs_hard(seed, w, h, 120);
        assert!(rounds >= 0, "game produced a sane round count");
    }
}

#[test]
fn hard_ai_is_deterministic() {
    let r1 = run_hard_vs_hard(123, 14, 12, 120);
    let r2 = run_hard_vs_hard(123, 14, 12, 120);
    assert_eq!(r1, r2, "HardAi must be deterministic for fixed inputs");
}

#[test]
fn hard_ai_builds_an_economy() {
    // After a number of rounds each HardAi seat should have grown well past its
    // 9 starting tiles AND built income buildings (farms/mines). This guards
    // against a port that silently no-ops (the benchmark would be meaningless).
    let seed = 256;
    let (w, h, cap) = (16, 14, 60);
    let mut g = Game::new(w, h, &["P1", "P2"]);
    g.generate_map(w, h, seed);
    let mut a = HardAi::hard();
    let mut b = HardAi::hard();
    for _ in 0..2 {
        let cur = g.current_player();
        if cur.0 == 0 {
            a.place_headquarters(&mut g, cur);
        } else {
            b.place_headquarters(&mut g, cur);
        }
        g.change_turn();
    }
    while g.live_players().len() > 1 && g.get_rounds_played() < cap {
        let cur = g.current_player();
        if cur.0 == 0 {
            a.plan_turn(&mut g, cur);
        } else {
            b.plan_turn(&mut g, cur);
        }
        if !matches!(g.end_turn(), EndTurnOutcome::Continue | EndTurnOutcome::PlayersLost(_)) {
            break;
        }
    }
    // At least one seat should have grown and built farms — the core of the
    // heuristic's economy. (We don't require both, since one may be losing.)
    let mut max_tiles = 0;
    let mut total_farms = 0;
    for i in 0..2 {
        let pid = PlayerId(i);
        let tiles = g.get_tile_count_for_player(pid);
        max_tiles = max_tiles.max(tiles);
        for tid in g.owned_tiles(pid) {
            if g.tiles[tid.0].building.as_ref().map(|bb| bb.kind) == Some(BuildingType::Farm) {
                total_farms += 1;
            }
        }
    }
    assert!(max_tiles > 9, "HardAi should expand past its starting tiles (got {max_tiles})");
    assert!(total_farms > 0, "HardAi should build farms (economy)");
}
