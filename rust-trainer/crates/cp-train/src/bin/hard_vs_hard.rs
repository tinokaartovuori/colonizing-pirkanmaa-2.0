//! Diagnostic: how long does a *natural* game take? Plays the held-out HARD bot
//! against itself with an effectively-unlimited round cap and reports the
//! distribution of rounds-to-decision, plus a truncation table answering the
//! practical question: "at training cap = C, what fraction of natural games are
//! still unresolved?"
//!
//! The hard bot is no genius, but it gives a baseline game length independent of
//! how strong our trainee is — which is exactly what we need to pick the training
//! `--cap`. Pure heuristic (no MCTS) → very fast, so we can run thousands of games.
//!
//! Usage:
//!   cargo run --release -p cp-train --bin hard_vs_hard -- \
//!     --games 500 --cap 5000 --width 12 --height 12 --seed 1 --threads 4

use cp_ai::HardAi;
use cp_sim::{EndTurnOutcome, Game, PlayerId, WinCause};
use rayon::prelude::*;

struct Cfg { games: usize, cap: i64, width: i32, height: i32, seed: u32, threads: usize }
impl Default for Cfg {
    fn default() -> Self { Cfg { games: 500, cap: 5000, width: 12, height: 12, seed: 1, threads: 4 } }
}

/// The way a game resolved (STRANGE-DEVICE-DESIGN.md §10). `Tiebreak` is the
/// harness-side tile-majority resolution at the cap (it REPLACES the old "timeout"
/// and is a *win*, not a draw); `Tie` is a true tie (equal tiles at the cap) and
/// should stay ~0 once the Device is doing its job.
#[derive(Clone, Copy, PartialEq)]
enum Cause { Device, Domination, Conquest, Bankruptcy, Tiebreak, Tie }

/// One game's result: round at which it ended + outcome code
/// (0 = seat0 win, 1 = seat1 win, 2 = true tie) + win cause + final board
/// occupancy (fractions of ALL tiles, the same denominator the bench uses) +
/// whether each seat ever built a Device (build/builder-win telemetry).
struct GameOut { rounds: i64, code: u8, cause: Cause, frac0: f64, frac1: f64, frac_neutral: f64, built0: bool, built1: bool }

fn map_cause(c: Option<WinCause>) -> Cause {
    match c {
        Some(WinCause::Device) => Cause::Device,
        Some(WinCause::Domination) => Cause::Domination,
        Some(WinCause::Conquest) => Cause::Conquest,
        Some(WinCause::Bankruptcy) => Cause::Bankruptcy,
        None => Cause::Conquest, // sole-survivor with no recorded cause == last-standing
    }
}

fn play(seed: u32, cfg: &Cfg) -> GameOut {
    let mut g = Game::new(cfg.width, cfg.height, &["P1", "P2"]);
    g.generate_map(cfg.width, cfg.height, seed);
    let a = HardAi::hard();
    let mut p0 = HardAi::hard();
    let mut p1 = HardAi::hard();

    // Round 0: HQ placement for both seats (place_headquarters is &self).
    for _ in 0..2 {
        let cur = g.current_player();
        a.place_headquarters(&mut g, cur);
        g.change_turn();
    }

    let mut code = 3u8; // 3 = not yet resolved by a natural terminal outcome
    let mut cause = Cause::Tie;
    while g.live_players().len() > 1 && g.get_rounds_played() < cfg.cap {
        let cur = g.current_player();
        if cur.0 == 0 { p0.plan_turn(&mut g, cur); } else { p1.plan_turn(&mut g, cur); }
        match g.end_turn() {
            EndTurnOutcome::Win(p) => { code = if p.0 == 0 { 0 } else { 1 }; cause = map_cause(g.last_win_cause()); break; }
            EndTurnOutcome::Tie => { code = 2; cause = Cause::Tie; break; }
            _ => {}
        }
    }
    // Sole-survivor fallback (matches the bench loops).
    if code == 3 {
        let live = g.live_players();
        if live.len() == 1 { code = if live[0].0 == 0 { 0 } else { 1 }; cause = map_cause(g.last_win_cause()); }
    }
    let total = g.get_tile_count().max(1) as f64;
    let frac0 = g.get_tile_count_for_player(PlayerId(0)) as f64 / total;
    let frac1 = g.get_tile_count_for_player(PlayerId(1)) as f64 / total;
    let frac_neutral = g.get_neutral_tiles() as f64 / total;
    // Harness-side tile-majority tiebreak: a game still unresolved at the cap is
    // resolved to the tile leader (a *win*), guaranteeing zero draws. Equal tiles
    // → a true Tie (should be ~0).
    if code == 3 {
        let (t0, t1) = (g.get_tile_count_for_player(PlayerId(0)), g.get_tile_count_for_player(PlayerId(1)));
        if t0 > t1 { code = 0; cause = Cause::Tiebreak; }
        else if t1 > t0 { code = 1; cause = Cause::Tiebreak; }
        else { code = 2; cause = Cause::Tie; }
    }
    let built0 = g.seat_events(PlayerId(0)).strange_devices_built > 0;
    let built1 = g.seat_events(PlayerId(1)).strange_devices_built > 0;
    GameOut { rounds: g.get_rounds_played(), code, cause, frac0, frac1, frac_neutral, built0, built1 }
}

fn pct(sorted: &[i64], p: f64) -> i64 {
    if sorted.is_empty() { return 0; }
    let idx = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
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

    println!("hard_vs_hard: {} games on {}x{}, safety cap {}, seed {} ({} threads)",
        cfg.games, cfg.width, cfg.height, cfg.cap, cfg.seed, cfg.threads);

    let outs: Vec<GameOut> = (0..cfg.games)
        .into_par_iter()
        .map(|gi| play(cfg.seed.wrapping_add((gi as u32).wrapping_mul(2_654_435_761)), &cfg))
        .collect();

    let (mut w0, mut w1, mut tie) = (0usize, 0usize, 0usize);
    let mut decisive_rounds: Vec<i64> = Vec::new();
    // Outcome-cause breakdown (STRANGE-DEVICE-DESIGN.md §10).
    let (mut c_device, mut c_dom, mut c_conq, mut c_bank, mut c_tb, mut c_tie) =
        (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
    let mut device_rounds: Vec<i64> = Vec::new();
    for o in &outs {
        match o.code {
            0 => w0 += 1,
            1 => w1 += 1,
            _ => tie += 1,
        }
        match o.cause {
            Cause::Device => { c_device += 1; device_rounds.push(o.rounds); }
            Cause::Domination => c_dom += 1,
            Cause::Conquest => c_conq += 1,
            Cause::Bankruptcy => c_bank += 1,
            Cause::Tiebreak => c_tb += 1,
            Cause::Tie => c_tie += 1,
        }
        // Rounds-to-decision counts only NATURAL resolutions (not the cap tiebreak,
        // whose rounds == cap and would skew the distribution).
        if !matches!(o.cause, Cause::Tiebreak | Cause::Tie) {
            decisive_rounds.push(o.rounds);
        }
    }
    let n = cfg.games.max(1);
    let decisive = w0 + w1;
    let pc = |x: usize| 100.0 * x as f64 / n as f64;
    println!("\nwinner seat ({} games):", n);
    println!("  seat0 win : {:4}  ({:5.1}%)", w0, pc(w0));
    println!("  seat1 win : {:4}  ({:5.1}%)", w1, pc(w1));
    println!("  true tie  : {:4}  ({:5.1}%)", tie, pc(tie));

    // The headline "did the redesign work" signal: the non-decisive fraction
    // (true tie only — the tile-majority tiebreak is now a WIN, not a draw).
    println!("\noutcome by CAUSE (§10):");
    println!("  Strange Device   : {:4}  ({:5.1}%)", c_device, pc(c_device));
    println!("  Domination (>=70%): {:4}  ({:5.1}%)", c_dom, pc(c_dom));
    println!("  Conquest (0 tiles): {:4}  ({:5.1}%)", c_conq, pc(c_conq));
    println!("  Bankruptcy       : {:4}  ({:5.1}%)", c_bank, pc(c_bank));
    println!("  Tile-majority TB : {:4}  ({:5.1}%)  [resolved at cap {} — a win, not a draw]", c_tb, pc(c_tb), cfg.cap);
    println!("  Non-decisive (true tie): {:4}  ({:5.1}%)  <-- target ~0", c_tie, pc(c_tie));
    if !device_rounds.is_empty() {
        let s: i64 = device_rounds.iter().sum();
        println!("  (Device wins resolved at mean round {:.1})", s as f64 / device_rounds.len() as f64);
    }

    // Builder economics — the lever the balance target cares about: when a seat
    // COMMITS to the Device, does it pay off? `builder win` = a game where exactly
    // one seat built a Device and that seat won (by ANY route). Device-survival =
    // of all games with a build, how many ended by the countdown reaching 0.
    let mut builds = 0usize;          // games with >=1 Device built
    let mut solo_builds = 0usize;     // games where exactly one seat built
    let mut builder_wins = 0usize;    // of solo_builds, the builder won
    let mut both_built = 0usize;
    for o in &outs {
        let any = o.built0 || o.built1;
        if any { builds += 1; }
        if o.built0 && o.built1 { both_built += 1; }
        else if o.built0 || o.built1 {
            solo_builds += 1;
            let builder = if o.built0 { 0u8 } else { 1u8 };
            if o.code == builder { builder_wins += 1; }
        }
    }
    println!("\nDevice builder economics (balance target: builder win-rate ~50%):");
    println!("  build rate (games with a Device built) : {:5.1}%  ({}/{})", pc(builds), builds, n);
    if solo_builds > 0 {
        println!("  builder WIN-RATE (single-builder games) : {:5.1}%  ({}/{} games)  <-- target ~50%",
            100.0 * builder_wins as f64 / solo_builds as f64, builder_wins, solo_builds);
    }
    if both_built > 0 {
        println!("  (both seats built in {} games — slot reopened after a destroy)", both_built);
    }
    if builds > 0 {
        println!("  Device-survival (built -> countdown win) : {:5.1}%  ({}/{} builds ended by Device)",
            100.0 * c_device as f64 / builds as f64, c_device, builds);
    }
    // Of bankruptcy games, was it the Device-builder who went broke, or the attacker
    // racing to crack it? (Decides whether lowering the build cost would help.)
    let mut bank_builder = 0usize;
    let mut bank_other = 0usize;
    for o in &outs {
        if o.cause != Cause::Bankruptcy { continue; }
        // winner is o.code (0/1); the bankrupt loser is the other seat.
        let loser_built = if o.code == 0 { o.built1 } else { o.built0 };
        if loser_built { bank_builder += 1; } else { bank_other += 1; }
    }
    let bank_total = bank_builder + bank_other;
    if bank_total > 0 {
        println!("  bankrupt loser was the Device-BUILDER : {:5.1}%  ({}/{} bankruptcies)",
            100.0 * bank_builder as f64 / bank_total as f64, bank_builder, bank_total);
    }

    // Board occupancy: what fraction of ALL tiles does the winner / loser hold at
    // game end, and how much stays neutral? This is the natural ceiling for the
    // training bench's tileFrac (same denominator), so we can judge whether our
    // trainee's stuck ~0.20-0.25 is "behind" or just normal for this map.
    let mut win_frac = 0.0; let mut lose_frac = 0.0; let mut dec_n = 0usize;
    let mut neutral_sum = 0.0;
    let mut stale_max = 0.0; let mut stale_min = 0.0; let mut stale_n = 0usize;
    for o in &outs {
        neutral_sum += o.frac_neutral;
        match o.code {
            0 => { win_frac += o.frac0; lose_frac += o.frac1; dec_n += 1; }
            1 => { win_frac += o.frac1; lose_frac += o.frac0; dec_n += 1; }
            _ => { // tie / stalemate: record the leader vs trailer split
                let (hi, lo) = if o.frac0 >= o.frac1 { (o.frac0, o.frac1) } else { (o.frac1, o.frac0) };
                stale_max += hi; stale_min += lo; stale_n += 1;
            }
        }
    }
    println!("\nboard occupancy (fraction of ALL {} tiles):", outs.first().map(|_| "map").unwrap_or("map"));
    println!("  mean neutral at end : {:5.1}%", 100.0 * neutral_sum / n as f64);
    if dec_n > 0 {
        println!("  DECISIVE: winner holds {:5.1}%  vs  loser {:5.1}%  ({} games)",
            100.0 * win_frac / dec_n as f64, 100.0 * lose_frac / dec_n as f64, dec_n);
    }
    if stale_n > 0 {
        println!("  STALEMATE/tie: leader {:5.1}%  vs  trailer {:5.1}%  ({} games)",
            100.0 * stale_max / stale_n as f64, 100.0 * stale_min / stale_n as f64, stale_n);
    }

    decisive_rounds.sort_unstable();
    if !decisive_rounds.is_empty() {
        let sum: i64 = decisive_rounds.iter().sum();
        let mean = sum as f64 / decisive_rounds.len() as f64;
        println!("\nrounds-to-decision over {} DECISIVE games:", decisive);
        println!("  min {}  p10 {}  p25 {}  median {}  p75 {}  p90 {}  p95 {}  p99 {}  max {}  | mean {:.1}",
            decisive_rounds[0], pct(&decisive_rounds, 10.0), pct(&decisive_rounds, 25.0),
            pct(&decisive_rounds, 50.0), pct(&decisive_rounds, 75.0), pct(&decisive_rounds, 90.0),
            pct(&decisive_rounds, 95.0), pct(&decisive_rounds, 99.0),
            decisive_rounds[decisive_rounds.len() - 1], mean);
    }

    // Truncation table: at training cap = C, what fraction of NATURAL games would
    // be cut off (i.e. still unresolved when the cap fires)? A game is "cut" if it
    // ended (decisively) at a round > C, or never resolved at all.
    println!("\ntruncation table — fraction of natural games still unresolved at cap C:");
    println!("   cap C |  cut% | of which decisive-but-late / never-resolve");
    for &c in &[40i64, 60, 80, 100, 120, 140, 160, 180, 200, 240, 300, 400] {
        let mut late = 0usize;   // would have resolved, but after C
        let mut never = 0usize;  // never resolved at all (tie/capped)
        for o in &outs {
            let resolved = o.code == 0 || o.code == 1;
            if resolved {
                if o.rounds > c { late += 1; }
            } else {
                // tie/capped: unresolved at any finite cap below where it ended
                if o.rounds > c { never += 1; }
            }
        }
        let cut = late + never;
        println!("  {:6} | {:5.1}% |  late {:4}  / never {:4}", c, 100.0 * cut as f64 / n as f64, late, never);
    }
}
