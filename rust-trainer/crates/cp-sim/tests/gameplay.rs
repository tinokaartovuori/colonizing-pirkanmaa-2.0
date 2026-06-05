//! Scripted headless scenarios exercising the turn-flow logic ported from
//! `GameEventHandler.endTurn` and the model production/conquest rules. These are
//! the Rust analogue of the TS `tests/gameplay.test.ts` minimal-scenario checks.

use cp_sim::{BuildingType, EndTurnOutcome, Game, TileType, UnitType};

/// Build a tiny empty map by hand (no worldgen) so scenarios are fully
/// controlled. All tiles are grassland; `Game::new` already created the players.
fn empty_grassland_game(w: i32, h: i32, players: &[&str]) -> Game {
    let mut g = Game::new(w, h, players);
    // Reuse worldgen's tile constructor by generating then overwriting types is
    // overkill; instead synthesise grassland tiles directly via a seed that we
    // then normalise. Simpler: generate with a seed and force every tile to
    // grassland with no building/units. We just need plain land for the tests.
    g.generate_map(w, h, 1);
    for t in &mut g.tiles {
        t.tile_type = TileType::Grassland;
        t.building = None;
        t.units.clear();
        t.conquering_units.clear();
        t.owner = None;
        t.wood_left = 0;
        t.rounds_stumps = 0;
    }
    g
}

#[test]
fn farm_with_worker_produces_175_after_four_rounds() {
    let mut g = empty_grassland_game(3, 3, &["P1", "P2"]);
    let p1 = g.player_id_by_num(1).unwrap();

    // P1 owns the centre tile with a farm + a worker on it.
    let tid = g.get_tile_at(cp_sim::coordinate::Coordinate::new(1, 1)).unwrap();
    g.set_tile_owner(tid, Some(p1));
    g.place_building(tid, BuildingType::Farm, Some(p1));
    g.spawn_unit_on_tile(UnitType::BasicWorker, p1, tid, false);

    // Known starting money is 400; zero it so we measure pure farm output and
    // can ignore the worker's -5 salary by setting it back each round.
    let start_money = g.players[p1.0].money();
    assert_eq!(start_money, 400);

    // Generating resources alone (no salary) for 4 rounds yields exactly one
    // 175 payout on the 4th (phase 1->2->3->4->5=produce).
    for round in 1..=4 {
        g.generate_resources(tid);
        let money = g.players[p1.0].money();
        if round < 4 {
            assert_eq!(money, 400, "no payout before round 4 (round {round})");
        } else {
            assert_eq!(money, 400 + 175, "farm pays 175 on round 4");
        }
    }
}

#[test]
fn farm_without_worker_never_pays() {
    let mut g = empty_grassland_game(3, 3, &["P1", "P2"]);
    let p1 = g.player_id_by_num(1).unwrap();
    let tid = g.get_tile_at(cp_sim::coordinate::Coordinate::new(1, 1)).unwrap();
    g.set_tile_owner(tid, Some(p1));
    g.place_building(tid, BuildingType::Farm, Some(p1));
    for _ in 0..8 {
        g.generate_resources(tid);
    }
    assert_eq!(g.players[p1.0].money(), 400);
}

#[test]
fn soldier_conquers_weakly_defended_tile() {
    // P1 (current) stages a soldier as a conquering unit on P2's tile that has
    // no defenders; strict `>` (1 > 0) means it is taken.
    let mut g = empty_grassland_game(3, 3, &["P1", "P2"]);
    let p1 = g.player_id_by_num(1).unwrap();
    let p2 = g.player_id_by_num(2).unwrap();
    assert_eq!(g.current_player(), p1);

    let tid = g.get_tile_at(cp_sim::coordinate::Coordinate::new(1, 1)).unwrap();
    g.set_tile_owner(tid, Some(p2));
    // Attacker soldier staged as conquering on the enemy tile.
    g.spawn_unit_on_tile(UnitType::Soldier, p1, tid, true);

    g.conquer_tile(tid, p1);

    assert_eq!(g.tiles[tid.0].owner, Some(p1), "tile changes hands");
    // The attacker becomes an owned unit on the tile.
    assert_eq!(g.tiles[tid.0].units.len(), 1);
    assert!(g.tiles[tid.0].conquering_units.is_empty());
}

#[test]
fn equal_soldiers_do_not_conquer_strict_gt() {
    // 1 attacker vs 1 defender: NOT > so the assault fails and attackers die.
    let mut g = empty_grassland_game(3, 3, &["P1", "P2"]);
    let p1 = g.player_id_by_num(1).unwrap();
    let p2 = g.player_id_by_num(2).unwrap();
    let tid = g.get_tile_at(cp_sim::coordinate::Coordinate::new(1, 1)).unwrap();
    g.set_tile_owner(tid, Some(p2));
    g.spawn_unit_on_tile(UnitType::Soldier, p2, tid, false); // defender
    g.spawn_unit_on_tile(UnitType::Soldier, p1, tid, true); // attacker

    g.conquer_tile(tid, p1);

    assert_eq!(g.tiles[tid.0].owner, Some(p2), "owner unchanged on a tie");
    assert!(g.tiles[tid.0].conquering_units.is_empty(), "failed attackers removed");
    assert_eq!(g.tiles[tid.0].units.len(), 1, "defender survives");
}

#[test]
fn outpost_cannot_be_taken_even_with_more_soldiers() {
    let mut g = empty_grassland_game(3, 3, &["P1", "P2"]);
    let p1 = g.player_id_by_num(1).unwrap();
    let p2 = g.player_id_by_num(2).unwrap();
    let tid = g.get_tile_at(cp_sim::coordinate::Coordinate::new(1, 1)).unwrap();
    g.set_tile_owner(tid, Some(p2));
    g.place_building(tid, BuildingType::Outpost, Some(p2));
    // 2 attackers, 0 defenders — would normally win, but outpost blocks it.
    g.spawn_unit_on_tile(UnitType::Soldier, p1, tid, true);
    g.spawn_unit_on_tile(UnitType::Soldier, p1, tid, true);

    g.conquer_tile(tid, p1);

    assert_eq!(g.tiles[tid.0].owner, Some(p2), "outpost holds");
    assert!(g.tiles[tid.0].conquering_units.is_empty(), "attackers wiped");
}

#[test]
fn bankruptcy_neutralizes_player_in_end_turn() {
    // P2 owns a tile but has negative money -> end_turn neutralises P2.
    let mut g = empty_grassland_game(3, 3, &["P1", "P2"]);
    let p1 = g.player_id_by_num(1).unwrap();
    let p2 = g.player_id_by_num(2).unwrap();

    // Give each a connected HQ tile so connectivity-cut doesn't strip them first.
    let t1 = g.get_tile_at(cp_sim::coordinate::Coordinate::new(0, 0)).unwrap();
    g.set_tile_owner(t1, Some(p1));
    g.place_building(t1, BuildingType::Headquarters, Some(p1));

    let t2 = g.get_tile_at(cp_sim::coordinate::Coordinate::new(2, 2)).unwrap();
    g.set_tile_owner(t2, Some(p2));
    g.place_building(t2, BuildingType::Headquarters, Some(p2));

    // Push P2 into the red.
    g.set_player_resources(p2, -50, 10, 10, 10);

    // P1 is current; end_turn evaluates losses for everyone.
    let outcome = g.end_turn();

    // P2 is removed from the live players and neutralised (its HQ tile freed).
    assert!(!g.players[p2.0].alive, "bankrupt player marked lost");
    assert_eq!(g.live_players().len(), 1);
    assert_eq!(g.live_players()[0], p1);
    // After neutralisation P2 owns nothing and its HQ is now conquered/neutral.
    assert!(g.players[p2.0].objects.is_empty());
    match outcome {
        EndTurnOutcome::Win(w) => assert_eq!(w, p1),
        other => panic!("expected P1 win after sole survivor, got {other:?}"),
    }
}

#[test]
fn seventy_percent_ownership_triggers_win() {
    // 3x3 = 9 tiles; owning 7 (>=70%) wins. Give P1 the whole left+middle.
    let mut g = empty_grassland_game(3, 3, &["P1", "P2"]);
    let p1 = g.player_id_by_num(1).unwrap();
    let p2 = g.player_id_by_num(2).unwrap();

    // P1 owns 7 tiles (indices 0..7), with an HQ on the first so connectivity
    // keeps them; ensure they are orthogonally connected (column-major 3x3:
    // index = x*3 + y, so 0..6 are columns 0 and 1 plus (2,0) — all touch).
    let coords = [
        (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0),
    ];
    for (i, (x, y)) in coords.iter().enumerate() {
        let t = g.get_tile_at(cp_sim::coordinate::Coordinate::new(*x, *y)).unwrap();
        g.set_tile_owner(t, Some(p1));
        if i == 0 {
            g.place_building(t, BuildingType::Headquarters, Some(p1));
        }
    }
    // P2 keeps an HQ tile so it is alive going in.
    let t2 = g.get_tile_at(cp_sim::coordinate::Coordinate::new(2, 2)).unwrap();
    g.set_tile_owner(t2, Some(p2));
    g.place_building(t2, BuildingType::Headquarters, Some(p2));

    assert_eq!(g.get_tile_count_for_player(p1), 7);
    let outcome = g.end_turn();
    match outcome {
        EndTurnOutcome::Win(w) => assert_eq!(w, p1, "P1 dominates with >=70%"),
        other => panic!("expected domination win, got {other:?}"),
    }
    assert!(!g.players[p2.0].alive);
}

#[test]
fn hq_connectivity_cut_strands_disconnected_tile() {
    // P2 (opponent) owns an HQ tile and a separate, disconnected tile. After P1's
    // end_turn the stranded tile is neutralised (set to None).
    let mut g = empty_grassland_game(5, 1, &["P1", "P2"]);
    let p1 = g.player_id_by_num(1).unwrap();
    let p2 = g.player_id_by_num(2).unwrap();

    // Row map (height 1): x=0..4. P2 HQ at x=0, P2 stray at x=4 (not adjacent).
    let hq = g.get_tile_at(cp_sim::coordinate::Coordinate::new(0, 0)).unwrap();
    g.set_tile_owner(hq, Some(p2));
    g.place_building(hq, BuildingType::Headquarters, Some(p2));
    let stray = g.get_tile_at(cp_sim::coordinate::Coordinate::new(4, 0)).unwrap();
    g.set_tile_owner(stray, Some(p2));

    // P1 needs to exist with a tile to remain alive; give it x=2.
    let mid = g.get_tile_at(cp_sim::coordinate::Coordinate::new(2, 0)).unwrap();
    g.set_tile_owner(mid, Some(p1));
    g.place_building(mid, BuildingType::Headquarters, Some(p1));

    g.end_turn();

    assert_eq!(g.tiles[hq.0].owner, Some(p2), "HQ tile retained");
    assert_eq!(g.tiles[stray.0].owner, None, "stranded tile neutralised");
}

#[test]
fn full_headless_game_runs_to_completion_deterministically() {
    // Two players each get an HQ; run end_turns until someone wins or a cap.
    // This exercises the whole pipeline without panicking and terminates.
    let run = || -> (i64, EndTurnOutcome) {
        let mut g = Game::new(12, 12, &["P1", "P2"]);
        g.generate_map(12, 12, 1);
        // Claim HQs at the golden hq placement indices (57, 123).
        let t1 = g.get_tiles()[57].id;
        let t2 = g.get_tiles()[123].id;
        g.first_round_actions(t1);
        g.change_turn();
        g.first_round_actions(t2);
        g.change_turn();

        let mut last = EndTurnOutcome::Continue;
        for _ in 0..200 {
            last = g.end_turn();
            if matches!(last, EndTurnOutcome::Win(_) | EndTurnOutcome::Tie) {
                break;
            }
        }
        (g.get_rounds_played(), last)
    };
    let a = run();
    let b = run();
    assert_eq!(a.0, b.0, "rounds played is deterministic");
    assert_eq!(a.1, b.1, "outcome is deterministic");
}
