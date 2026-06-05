//! CEILING PROBE: does a "hard + offensive-cut targeting" heuristic beat the
//! shipped HARD bot, and by how much? This answers whether the *strategy* the
//! spatial AI is meant to learn (concentrate force on the articulation tile that
//! severs the most enemy territory) has a high enough ceiling to justify building
//! macro-actions + retraining. If even hard+cut can't clear ~70% vs hard, then
//! 70% is likely unreachable in this draw-prone game and the target should be
//! recalibrated — independent of any AI cleverness.
//!
//! The cut bot is HARD with `cut_priority` (orders attack targets by
//! `offensive_cut_value` instead of HQ-first). To remove first-move/seat bias,
//! the cut bot plays seat 0 on even games and seat 1 on odd games; wins are
//! attributed to "cut" vs "hard" regardless of seat.
//!
//! Usage:
//!   cargo run --release -p cp-train --bin cut_vs_hard -- \
//!     --games 400 --cap 120 --width 14 --height 12 --seed 7 --threads 4

use cp_ai::HardAi;
use cp_sim::{EndTurnOutcome, Game, PlayerId};
use rayon::prelude::*;

struct Cfg { games: usize, cap: i64, width: i32, height: i32, seed: u32, threads: usize }
impl Default for Cfg {
    fn default() -> Self { Cfg { games: 400, cap: 120, width: 14, height: 12, seed: 7, threads: 4 } }
}

/// Outcome from the CUT bot's perspective: 0 = cut win, 1 = hard win, 2 = tie,
/// 3 = unresolved at cap.
fn play(seed: u32, cut_seat: usize, cfg: &Cfg) -> u8 {
    let mut g = Game::new(cfg.width, cfg.height, &["P1", "P2"]);
    g.generate_map(cfg.width, cfg.height, seed);
    // Seat that holds the cut bot this game (the other seat is plain hard).
    let mut p0 = if cut_seat == 0 { HardAi::hard_cut() } else { HardAi::hard() };
    let mut p1 = if cut_seat == 1 { HardAi::hard_cut() } else { HardAi::hard() };
    let placer = HardAi::hard();

    for _ in 0..2 {
        let cur = g.current_player();
        placer.place_headquarters(&mut g, cur);
        g.change_turn();
    }

    let mut winner_seat: Option<usize> = None;
    let mut tie = false;
    while g.live_players().len() > 1 && g.get_rounds_played() < cfg.cap {
        let cur = g.current_player();
        if cur.0 == 0 { p0.plan_turn(&mut g, cur); } else { p1.plan_turn(&mut g, cur); }
        match g.end_turn() {
            EndTurnOutcome::Win(p) => { winner_seat = Some(p.0); break; }
            EndTurnOutcome::Tie => { tie = true; break; }
            _ => {}
        }
    }
    if winner_seat.is_none() && !tie {
        let live = g.live_players();
        if live.len() == 1 { winner_seat = Some(live[0].0); }
    }
    match winner_seat {
        Some(s) => if s == cut_seat { 0 } else { 1 },
        None => if tie { 2 } else { 3 },
    }
}

fn main() {
    let mut cfg = Cfg::default();
    let a: Vec<String> = std::env::args().collect();
    let args = if let Some(p) = a.iter().position(|x| x == "--") { a[p + 1..].to_vec() } else { a[1..].to_vec() };
    let mut i = 0;
    while i < args.len() {
        let k = args[i].clone();
        macro_rules! v { () => {{ i += 1; args.get(i).cloned().unwrap_or_default() }} }
        match k.as_str() {
            "--games" => cfg.games = v!().parse().unwrap_or(cfg.games),
            "--cap" => cfg.cap = v!().parse().unwrap_or(cfg.cap),
            "--width" => cfg.width = v!().parse().unwrap_or(cfg.width),
            "--height" => cfg.height = v!().parse().unwrap_or(cfg.height),
            "--seed" => cfg.seed = v!().parse().unwrap_or(cfg.seed),
            "--threads" => cfg.threads = v!().parse().unwrap_or(cfg.threads),
            _ => {}
        }
        i += 1;
    }
    rayon::ThreadPoolBuilder::new().num_threads(cfg.threads.max(1)).build_global().ok();

    println!("cut_vs_hard: {} games on {}x{}, cap {}, seed {} ({} threads) — cut bot alternates seats",
        cfg.games, cfg.width, cfg.height, cfg.cap, cfg.seed, cfg.threads);

    let codes: Vec<u8> = (0..cfg.games)
        .into_par_iter()
        .map(|gi| {
            let seed = cfg.seed.wrapping_add((gi as u32).wrapping_mul(2_654_435_761));
            play(seed, gi % 2, &cfg)
        })
        .collect();

    let (mut cut, mut hard, mut tie, mut unres) = (0usize, 0usize, 0usize, 0usize);
    for c in &codes {
        match c { 0 => cut += 1, 1 => hard += 1, 2 => tie += 1, _ => unres += 1 }
    }
    let n = cfg.games.max(1) as f64;
    let decisive = (cut + hard).max(1) as f64;
    println!("\noutcomes ({} games):", cfg.games);
    println!("  CUT  win  : {:4}  ({:5.1}%)", cut, 100.0 * cut as f64 / n);
    println!("  HARD win  : {:4}  ({:5.1}%)", hard, 100.0 * hard as f64 / n);
    println!("  tie       : {:4}  ({:5.1}%)", tie, 100.0 * tie as f64 / n);
    println!("  unresolved: {:4}  ({:5.1}%)", unres, 100.0 * unres as f64 / n);
    println!("\n  cut win-rate (all games)      : {:5.1}%", 100.0 * cut as f64 / n);
    println!("  cut share of DECISIVE games   : {:5.1}%  ({} decisive)", 100.0 * cut as f64 / decisive, cut + hard);
    println!("\n  >>> 70% target needs cut win-rate >= 70% of ALL games. Draw-proneness caps this.");
}
