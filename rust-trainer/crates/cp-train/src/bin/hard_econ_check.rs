//! Benchmark-integrity check: does the HARD bot ever DRIVE ITSELF into bankruptcy
//! (negative resource → instant loss) WITHOUT external pressure? If it does, then
//! some of our AI's "wins" are really hard's economic suicide, distorting the
//! vs-hard win-rate. We want: hard goes bankrupt ONLY from an external cause
//! (its income cut by the opponent taking its tiles), never on its own.
//!
//! Two modes:
//!   --mode solo   : hard plays ALONE (1 player, no opponent) for N rounds. With no
//!                   enemy it can only bankrupt via its own (mis)management. Expect: never.
//!   --mode vshard : hard vs hard. At every end_turn we scan BOTH players for a
//!                   negative resource and record their TILE COUNT at that moment —
//!                   bankruptcy while still HEALTHY (many tiles) = self-inflicted
//!                   (distorting); bankruptcy while CUT DOWN (few tiles) = external (ok).
//!
//! Usage:
//!   cargo run --release -p cp-train --bin hard_econ_check -- --mode solo   --games 100 --rounds 250
//!   cargo run --release -p cp-train --bin hard_econ_check -- --mode vshard --games 300 --cap 200

use cp_ai::HardAi;
use cp_sim::resources::BasicResource;
use cp_sim::model::ObjId;
use cp_sim::{EndTurnOutcome, Game, PlayerId};
use rayon::prelude::*;

const RES: [BasicResource; 4] = [BasicResource::Money, BasicResource::Wood, BasicResource::Stone, BasicResource::Metal];

fn min_resource(g: &Game, p: PlayerId) -> i64 {
    RES.iter().map(|&r| g.players[p.0].resources.get(r).unwrap_or(0)).min().unwrap_or(0)
}
fn tile_count(g: &Game, p: PlayerId) -> i64 {
    g.players[p.0].objects.iter().filter(|o| matches!(o, ObjId::Tile(_))).count() as i64
}

/// One solo game: hard (seat0) vs a PASSIVE opponent (seat1: places its HQ then
/// never acts — never moves, expands, or attacks). Seat0 therefore faces ZERO
/// external pressure, so any negative resource is purely self-inflicted. Runs
/// until seat0 conquers the passive seat (or `rounds`/cap). Returns seat0's
/// minimum resource ever held and the round it first went negative (-1 if never).
/// The engine cannot run a true 1-player game (the lone player instantly "wins"),
/// so a do-nothing seat is the faithful no-pressure proxy.
fn solo(seed: u32, w: i32, h: i32, rounds: i64) -> (i64, i64) {
    let mut g = Game::new(w, h, &["P1", "P2"]);
    g.generate_map(w, h, seed);
    let placer = HardAi::hard();
    for _ in 0..2 { let cur = g.current_player(); placer.place_headquarters(&mut g, cur); g.change_turn(); }
    let mut hard = HardAi::hard();
    let mut overall_min = i64::MAX;
    let mut first_neg = -1i64;
    let mut steps = 0i64;
    while g.live_players().len() > 1 && steps < rounds {
        let cur = g.current_player();
        if cur.0 == 0 { hard.plan_turn(&mut g, cur); } // seat1 (passive) does nothing
        let _ = g.end_turn();
        let m = min_resource(&g, PlayerId(0));
        overall_min = overall_min.min(m);
        if m < 0 && first_neg < 0 { first_neg = g.get_rounds_played(); break; }
        steps += 1;
    }
    (overall_min, first_neg)
}

/// One hard-vs-hard game: returns any bankruptcy events as (tiles_at_bankruptcy, round).
fn vshard(seed: u32, w: i32, h: i32, cap: i64) -> Vec<(i64, i64)> {
    let mut g = Game::new(w, h, &["P1", "P2"]);
    g.generate_map(w, h, seed);
    let h0 = HardAi::hard();
    let mut p0 = HardAi::hard();
    let mut p1 = HardAi::hard();
    for _ in 0..2 { let cur = g.current_player(); h0.place_headquarters(&mut g, cur); g.change_turn(); }
    let mut events = Vec::new();
    let mut flagged = [false, false];
    while g.live_players().len() > 1 && g.get_rounds_played() < cap {
        let cur = g.current_player();
        if cur.0 == 0 { p0.plan_turn(&mut g, cur); } else { p1.plan_turn(&mut g, cur); }
        let out = g.end_turn();
        // Scan both seats for a first-time negative resource.
        for s in 0..2 {
            if !flagged[s] && min_resource(&g, PlayerId(s)) < 0 {
                flagged[s] = true;
                events.push((tile_count(&g, PlayerId(s)), g.get_rounds_played()));
            }
        }
        if matches!(out, EndTurnOutcome::Win(_) | EndTurnOutcome::Tie) { break; }
    }
    events
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let args = if let Some(p) = a.iter().position(|x| x == "--") { a[p + 1..].to_vec() } else { a[1..].to_vec() };
    let mut mode = String::from("solo");
    let (mut games, mut rounds, mut cap, mut w, mut h, mut seed, mut threads) = (100usize, 250i64, 200i64, 14i32, 12i32, 1u32, 16usize);
    let mut i = 0;
    while i < args.len() {
        let k = args[i].clone();
        macro_rules! v { () => {{ i += 1; args.get(i).cloned().unwrap_or_default() }} }
        match k.as_str() {
            "--mode" => mode = v!(),
            "--games" => games = v!().parse().unwrap_or(games),
            "--rounds" => rounds = v!().parse().unwrap_or(rounds),
            "--cap" => cap = v!().parse().unwrap_or(cap),
            "--width" => w = v!().parse().unwrap_or(w),
            "--height" => h = v!().parse().unwrap_or(h),
            "--seed" => seed = v!().parse().unwrap_or(seed),
            "--threads" => threads = v!().parse().unwrap_or(threads),
            _ => {}
        }
        i += 1;
    }
    rayon::ThreadPoolBuilder::new().num_threads(threads.max(1)).build_global().ok();
    let mkseed = |gi: usize| seed.wrapping_add((gi as u32).wrapping_mul(2_654_435_761));

    if mode == "solo" {
        println!("hard_econ_check SOLO: {} games, hard alone, {} rounds, {}x{}", games, rounds, w, h);
        let results: Vec<(i64, i64)> = (0..games).into_par_iter().map(|gi| solo(mkseed(gi), w, h, rounds)).collect();
        let bankruptcies = results.iter().filter(|(_, fn_)| *fn_ >= 0).count();
        let worst_min = results.iter().map(|(m, _)| *m).min().unwrap_or(0);
        println!("  bankruptcies (went negative with NO opponent): {}/{}", bankruptcies, games);
        println!("  worst minimum resource ever held across all games/rounds: {}", worst_min);
        if bankruptcies == 0 {
            println!("  => VERDICT: hard NEVER self-bankrupts solo. Any vs-game bankruptcy is externally caused. Benchmark is clean.");
        } else {
            println!("  => WARNING: hard self-bankrupts with no opponent in {} games — benchmark IS distorted.", bankruptcies);
        }
    } else {
        println!("hard_econ_check VS-HARD: {} games, cap {}, {}x{}", games, cap, w, h);
        let all: Vec<(i64, i64)> = (0..games).into_par_iter().flat_map(|gi| vshard(mkseed(gi), w, h, cap)).collect();
        let total = all.len();
        let healthy = all.iter().filter(|(tiles, _)| *tiles >= 8).count();   // self-inflicted while strong
        let cutdown = total - healthy;                                       // external (already reduced)
        println!("  total bankruptcy events (a seat went negative): {} over {} games", total, games);
        println!("  while HEALTHY (>=8 tiles, self-inflicted/distorting): {}", healthy);
        println!("  while CUT DOWN (<8 tiles, externally caused/ok):     {}", cutdown);
        if total > 0 {
            let avg_tiles = all.iter().map(|(t, _)| *t).sum::<i64>() as f64 / total as f64;
            let avg_round = all.iter().map(|(_, r)| *r).sum::<i64>() as f64 / total as f64;
            println!("  mean tiles at bankruptcy: {:.1} | mean round: {:.1}", avg_tiles, avg_round);
        }
    }
}
