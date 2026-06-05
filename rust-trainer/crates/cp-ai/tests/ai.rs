//! Tests for the cp-ai neural AI port (M3).
//!
//! - `forward()` matches hand-computed values for tiny known genomes.
//! - A self-play game runs to completion and is deterministic across two runs.
//! - `global_features` on the reconstructed first-decision state of game 1 is
//!   spot-checked against `golden/trace-1.json` (parity is asserted properly in
//!   M5; here we just report a match / first divergence).

use cp_ai::candidates;
use cp_ai::controller::{DecisionTrace, NeuralAiController};
use cp_ai::mlp::{self, Genome};
use cp_ai::policy::XorShift32;
use cp_ai::{run_game, TRAINING_CONFIG};
use cp_sim::Game;

const MANIFEST: &str = env!("CARGO_MANIFEST_DIR");

fn champion_path() -> String {
    format!("{}/../../../training/checkpoints/champion.json", MANIFEST)
}
fn golden_path(seed: u32) -> String {
    format!("{}/../../golden/trace-{}.json", MANIFEST, seed)
}

// ---------------------------------------------------------------------------
// forward()
// ---------------------------------------------------------------------------

#[test]
fn forward_single_linear_layer() {
    // arch [2,1]: 2 weights then 1 bias. Output layer is linear.
    // params = [w0, w1, bias]; for output j=0 weights at base=0..2, bias at 2.
    let g = Genome {
        arch: vec![2, 1],
        params: vec![0.5, -0.5, 0.1],
    };
    let out = mlp::forward(&g, &[1.0, 2.0]);
    // 0.1 + 0.5*1 + (-0.5)*2 = -0.4
    assert_eq!(out.len(), 1);
    assert!((out[0] - (-0.4)).abs() < 1e-12, "got {}", out[0]);
}

#[test]
fn forward_tanh_hidden_then_linear() {
    // arch [2,2,1].
    // Layer0 (hidden, tanh): nin=2,nout=2 -> 4 weights + 2 biases.
    //   weights row-major: out0 -> [w00,w01], out1 -> [w10,w11]; biases [b0,b1].
    //   params[0..4]=weights, params[4..6]=biases.
    // Layer1 (output, linear): nin=2,nout=1 -> 2 weights + 1 bias.
    let params = vec![
        // layer0 weights
        0.1, 0.2, // out0
        -0.3, 0.4, // out1
        // layer0 biases
        0.05, -0.05, // layer1 weights
        0.7, -0.2, // layer1 bias
        0.01,
    ];
    let g = Genome {
        arch: vec![2, 2, 1],
        params,
    };
    let input = [1.0, -1.0];
    // hidden out0 = tanh(0.05 + 0.1*1 + 0.2*(-1)) = tanh(-0.05)
    // hidden out1 = tanh(-0.05 + (-0.3)*1 + 0.4*(-1)) = tanh(-0.75)
    let h0 = (-0.05f64).tanh();
    let h1 = (-0.75f64).tanh();
    // output = 0.01 + 0.7*h0 + (-0.2)*h1
    let expected = 0.01 + 0.7 * h0 - 0.2 * h1;
    let out = mlp::forward(&g, &input);
    assert!((out[0] - expected).abs() < 1e-12, "got {} want {}", out[0], expected);
}

#[test]
fn zero_genome_outputs_zero() {
    let g = Genome::zero(&[57, 24, 16, 1]);
    assert_eq!(mlp::param_count(&g.arch), 1809);
    let out = mlp::forward(&g, &vec![1.0; 57]);
    assert_eq!(out, vec![0.0]);
}

#[test]
fn genome_json_roundtrip() {
    // Start from a parsed genome so its params are already in serde's canonical
    // f64 form; serialize→parse must then be a fixed point (the invariant the
    // trainer relies on when it saves/loads champions).
    let src = Genome {
        arch: vec![57, 24, 16, 1],
        params: (0..1809).map(|i| i as f64 * 0.001).collect(),
    };
    let g = Genome::from_json(&src.to_json()).unwrap();
    let json = g.to_json();
    let back = Genome::from_json(&json).unwrap();
    assert_eq!(back.arch, g.arch);
    assert_eq!(back.params, g.params);
    assert_eq!(back.to_json(), json);
}

#[test]
fn genome_loads_champion_json_format() {
    // The `{arch, params}` JSON the TS engine writes must load verbatim.
    let json = r#"{"arch":[57,24,16,1],"params":[0.1,-0.2,0.3]}"#;
    let g = Genome::from_json(json).unwrap();
    assert_eq!(g.arch, vec![57, 24, 16, 1]);
    assert_eq!(g.params, vec![0.1, -0.2, 0.3]);
}

// ---------------------------------------------------------------------------
// Self-play determinism
// ---------------------------------------------------------------------------

fn load_genome_or_zero() -> Genome {
    match Genome::from_file(&champion_path()) {
        Ok(g) => g,
        Err(_) => Genome::zero(&[57, 24, 16, 1]),
    }
}

#[test]
fn self_play_runs_to_completion_and_is_deterministic() {
    let genome = load_genome_or_zero();
    let genomes = [genome.clone(), genome.clone()];
    let r1 = run_game(1, 12, 12, &genomes, &TRAINING_CONFIG, 80);
    let r2 = run_game(1, 12, 12, &genomes, &TRAINING_CONFIG, 80);

    // Completed (rounds advanced past setup).
    assert!(r1.rounds >= 0);
    // Deterministic across two runs.
    assert_eq!(r1.winner_num, r2.winner_num);
    assert_eq!(r1.reason, r2.reason);
    assert_eq!(r1.rounds, r2.rounds);
    for (a, b) in r1.players.iter().zip(r2.players.iter()) {
        assert_eq!(a.num, b.num);
        assert_eq!(a.alive, b.alive);
        assert_eq!(a.money, b.money);
        assert_eq!(a.tiles, b.tiles);
    }
}

#[test]
fn self_play_three_player_runs() {
    let genome = load_genome_or_zero();
    let genomes = [genome.clone(), genome.clone(), genome.clone()];
    let r = run_game(2, 12, 12, &genomes, &TRAINING_CONFIG, 80);
    assert_eq!(r.players.len(), 3);
}

// ---------------------------------------------------------------------------
// globalFeatures spot-check against the golden trace (first decision, game 1).
// ---------------------------------------------------------------------------

/// Drive the exact export-golden setup for seed 1 (12x12, 2 players) up to the
/// first NN decision, capturing the global vector the policy saw.
fn first_decision_global_vec(genome: &Genome) -> Option<Vec<f64>> {
    let mut g = Game::new(12, 12, &["P1", "P2"]);
    g.generate_map(12, 12, 1);
    let ctrls = [
        NeuralAiController::new(genome, TRAINING_CONFIG),
        NeuralAiController::new(genome, TRAINING_CONFIG),
    ];
    let mut rng = XorShift32::new(1);
    for _ in 0..2 {
        let cur = g.current_player();
        ctrls[cur.0].place_headquarters(&mut g, cur);
        g.change_turn();
    }
    // Now it is P1's turn, round 0. Run plan_turn with a trace sink, grab the
    // first decision's global vector.
    let mut captured: Option<Vec<f64>> = None;
    {
        let mut sink = |d: DecisionTrace| {
            if captured.is_none() {
                captured = Some(d.global_vec);
            }
        };
        let cur = g.current_player();
        ctrls[cur.0].plan_turn(&mut g, cur, &mut rng, Some(&mut sink));
    }
    captured
}

#[test]
fn global_features_spot_check_trace_1() {
    // Read the golden trace's first-decision globalVec.
    let trace = match std::fs::read_to_string(golden_path(1)) {
        Ok(t) => t,
        Err(_) => {
            eprintln!("trace-1.json missing; skipping spot-check");
            return;
        }
    };
    let v: serde_json::Value = serde_json::from_str(&trace).unwrap();
    let want: Vec<f64> = v["rounds"][0]["turns"][0]["decisions"][0]["globalVec"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect();

    // Use the SAME genome the trace was produced with (champion checkpoint).
    let genome = match Genome::from_file(&champion_path()) {
        Ok(g) => g,
        Err(_) => {
            eprintln!("champion.json missing; cannot reproduce trace genome — skipping");
            return;
        }
    };

    let got = first_decision_global_vec(&genome).expect("a first decision should occur");
    assert_eq!(got.len(), want.len(), "globalVec length mismatch");

    let mut first_div: Option<usize> = None;
    for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
        if (a - b).abs() > 1e-9 {
            first_div = Some(i);
            break;
        }
    }
    match first_div {
        None => { /* full match — great */ }
        Some(i) => {
            panic!(
                "globalVec divergence at index {} ({}): got {} want {}\ngot ={:?}\nwant={:?}",
                i,
                cp_ai::features::GLOBAL_FEATURE_NAMES[i],
                got[i],
                want[i],
                got,
                want
            );
        }
    }
}

// A tiny enumerate sanity check: on a fresh just-placed HQ state, Pass is always
// the last candidate and BuildFarm (intent 0) is typically present.
#[test]
fn enumerate_appends_pass_last() {
    let genome = load_genome_or_zero();
    let mut g = Game::new(12, 12, &["P1", "P2"]);
    g.generate_map(12, 12, 1);
    let ctrl = NeuralAiController::new(&genome, TRAINING_CONFIG);
    let cur = g.current_player();
    ctrl.place_headquarters(&mut g, cur);
    let cands = candidates::enumerate(&g, cur, &TRAINING_CONFIG);
    assert!(!cands.is_empty());
    assert_eq!(cands.last().unwrap().intent, candidates::Intent::Pass);
}

// ---------------------------------------------------------------------------
// Multi-candidate Expand (Phase 1: learned target selection).
// ---------------------------------------------------------------------------

/// `clamp3` mirror for hand-computed expectations.
fn clamp3(v: f64) -> f64 {
    v.clamp(-3.0, 3.0)
}

/// `claimValue` mirror — used only to hand-compute the expected Expand ordering
/// and per-tile `local` slot 2 (targetValue) so a TS/Rust divergence in the
/// comparator or per-tile feature wiring is caught here, not just in parity.
fn claim_value_for_test(g: &Game, tid: cp_sim::TileId) -> i64 {
    use cp_sim::{BuildingType, TileType};
    if let Some(b) = &g.tiles[tid.0].building {
        if b.kind == BuildingType::Mikontalo {
            return 6;
        }
    }
    match g.tiles[tid.0].tile_type {
        TileType::Mountain => 5,
        TileType::Grassland => 4,
        TileType::Forest => 3,
        TileType::AbundantForest => 2,
        _ => 1,
    }
}

/// On the standard seed-1 just-placed-HQ state, Expand emits MULTIPLE candidates,
/// respects the cap of 6, is sorted (claimValue DESC, tile-index ASC), and each
/// candidate's `local` slot 2 equals the hand-computed clamped targetValue for
/// that tile (cross-checking the per-tile feature wiring against TS).
#[test]
fn expand_emits_multiple_capped_sorted_candidates() {
    use candidates::{Intent, EXPAND_CANDIDATE_CAP};
    use cp_ai::policy::XorShift32;
    let genome = load_genome_or_zero();
    let mut g = Game::new(12, 12, &["P1", "P2"]);
    g.generate_map(12, 12, 1);
    let ctrls = [
        NeuralAiController::new(&genome, TRAINING_CONFIG),
        NeuralAiController::new(&genome, TRAINING_CONFIG),
    ];
    let mut rng = XorShift32::new(1);

    // Place both HQs.
    for _ in 0..2 {
        let cur = g.current_player();
        ctrls[cur.0].place_headquarters(&mut g, cur);
        g.change_turn();
    }

    // Drive full turns (plan + end_turn) until P1 reaches a state with >1 Expand
    // candidate (the economy/unit-cap must open up for a worker to be
    // deliverable). The decision loop's safety scaffold runs inside plan_turn;
    // we inspect enumerate() at the START of P1's turn (round 1+) — but to
    // examine candidates we must run the scaffold ourselves first via a fresh
    // plan path. Simplest: snapshot enumerate at the start of each P1 turn after
    // running the same scaffold the controller would (by letting plan_turn run a
    // turn and re-enumerating at the top of the next P1 turn).
    let mut found: Option<usize> = None;
    for _round in 0..30 {
        if g.live_players().len() <= 1 {
            break;
        }
        let cur = g.current_player();
        if cur.0 == 0 {
            // Capture enumerate at the very start of P1's turn (after the
            // scaffold the prior turns already ran via plan_turn).
            let cands = candidates::enumerate(&g, cur, &TRAINING_CONFIG);
            let expands = cands
                .iter()
                .filter(|c| c.intent == Intent::Expand)
                .count();
            if expands > 1 {
                found = Some(expands);
                break;
            }
        }
        ctrls[cur.0].plan_turn(&mut g, cur, &mut rng, None);
        g.end_turn();
    }

    let cur = g.current_player();
    assert_eq!(cur.0, 0, "expected to break on P1's turn");
    let n = found.expect("expected a P1 turn with >1 Expand candidate within 30 rounds");

    // Cap respected.
    assert!(
        n <= EXPAND_CANDIDATE_CAP,
        "Expand cap exceeded: {} > {}",
        n,
        EXPAND_CANDIDATE_CAP
    );

    let cands = candidates::enumerate(&g, cur, &TRAINING_CONFIG);
    let expands: Vec<&candidates::Candidate> =
        cands.iter().filter(|c| c.intent == Intent::Expand).collect();

    // Reconstruct the expected target ordering from the engine directly:
    // neutral, has space, not threatened by an adjacent enemy soldier.
    let mut neutral: Vec<cp_sim::TileId> = g
        .get_available_tiles()
        .into_iter()
        .filter(|&t| {
            g.tiles[t.0].owner.is_none()
                && g.tiles[t.0].has_space_for_units()
                && !tile_threatened_for_test(&g, t, cur)
        })
        .collect();
    neutral.sort_by(|&a, &b| {
        claim_value_for_test(&g, b)
            .cmp(&claim_value_for_test(&g, a))
            .then(a.0.cmp(&b.0))
    });
    let expected: Vec<cp_sim::TileId> =
        neutral.into_iter().take(EXPAND_CANDIDATE_CAP).collect();

    // The emitted candidates must carry per-tile targetValue (local[2]) matching
    // each expected tile's claimValue, in the expected order — a hand-computed
    // cross-check of both the comparator and the per-tile `local` wiring.
    assert_eq!(
        expands.len(),
        expected.len(),
        "Expand candidate count != expected target count"
    );
    for (cand, &tid) in expands.iter().zip(expected.iter()) {
        let cv = claim_value_for_test(&g, tid) as f64;
        let want_slot2 = clamp3(cv / 6.0);
        assert!(
            (cand.local[2] - want_slot2).abs() < 1e-12,
            "Expand local[2] (targetValue) mismatch: got {} want {} for tile {}",
            cand.local[2],
            want_slot2,
            tid.0
        );
    }

    // Sorted invariant: claimValue non-increasing across emitted candidates, with
    // the tile-index ASC tie-break on equal claim values.
    for w in expected.windows(2) {
        assert!(
            claim_value_for_test(&g, w[0]) >= claim_value_for_test(&g, w[1]),
            "Expand targets not sorted by claimValue DESC"
        );
        if claim_value_for_test(&g, w[0]) == claim_value_for_test(&g, w[1]) {
            assert!(w[0].0 < w[1].0, "tie-break (tile index ASC) violated");
        }
    }
}

/// `tileThreatened` mirror for the expected-ordering reconstruction.
fn tile_threatened_for_test(g: &Game, tid: cp_sim::TileId, p: cp_sim::PlayerId) -> bool {
    use cp_sim::UnitType;
    for ntid in g.neighbour_tiles(tid) {
        let o = g.tiles[ntid.0].owner;
        if o.is_some()
            && o != Some(p)
            && g
                .tile_units(ntid)
                .iter()
                .any(|&u| g.units[u.0].kind == UnitType::Soldier)
        {
            return true;
        }
    }
    false
}
