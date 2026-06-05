//! AlphaZero training loop (Phase `trainloop`): iterate
//!   self-play (MCTS) → replay buffer → gradient train (policy CE + value MSE)
//!   → periodic benchmark vs the held-out HARD heuristic → checkpoint.
//!
//! Reuses the verified pieces: `cp_ai::selfplay::play_one_game` (data),
//! `cp_ai::policy_train::PolicyTrainer` (CE/Adam), `cp_ai::value::ValueTrainer`
//! (MSE/Adam), `cp_ai::search` (MCTS priors from the policy net). Writes the
//! dashboard artifacts (`log.jsonl`, `benchmark-history.jsonl`, `champion.json`,
//! `value.json`) into `--out` so the live dashboard shows the real vs-hard
//! win-rate climbing toward the 70% goal.
//!
//! Usage:
//!   cargo run --release -p cp-train --bin alphazero -- \
//!     --out rust-trainer/checkpoints-az --iters 200 --games 24 --sims 64 \
//!     --epochs 4 --batch 64 --bench-games 40 --bench-every 2

use std::collections::VecDeque;
use std::fs::{create_dir_all, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use cp_ai::controller::{DecisionTrace, NeuralAiController};
use cp_ai::mlp::{score, Genome};
use cp_ai::policy::{XorShift32, DEFAULT_ARCH};
use cp_ai::policy_spatial::DEFAULT_ARCH_SPATIAL;
use cp_ai::policy_train::{random_genome, softmax, PolicyExample, PolicyTrainer};
use cp_ai::selfplay::{play_one_game, play_one_game_vs_hard, SelfPlayData};
use cp_ai::value::{ValueExample, ValueNet, ValueTrainer, VALUE_ARCH_SPATIAL};
use cp_ai::{HardAi, LeafEval, SearchConfig, TRAINING_CONFIG};
use cp_sim::model::UnitType;
use cp_sim::{BuildingType, EndTurnOutcome, Game, PlayerId, TileType, WinCause};
use rayon::prelude::*;

/// Pure SplitMix64 mix — deterministic per-(iter, game) seeds for parallel
/// self-play / bench (no shared mutable RNG across threads).
fn smix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

// --- tiny deterministic RNG for shuffles / self-play seeds -----------------
struct SplitMix64(u64);
impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        if n == 0 { 0 } else { (self.next() % n as u64) as usize }
    }
}

#[derive(Clone)]
struct Cfg {
    out: PathBuf,
    iters: usize,
    games: usize,        // self-play games per iter
    sims: usize,         // MCTS sims per decision
    epochs: usize,       // training passes over the buffer per iter
    batch: usize,
    buffer: usize,       // replay buffer capacity (examples)
    bench_games: usize,
    bench_every: usize,
    players: usize,
    width: i32,
    height: i32,
    cap: i64,
    lr_policy: f64,
    lr_value: f64,
    seed: u64,
    leaf_value: bool,    // use the value net as the MCTS leaf (else static)
    init_policy: Option<PathBuf>, // warm-start the policy from this genome (else random)
    init_value: Option<PathBuf>,  // warm-start the value net (else random)
    shaping: f64,        // λ blend of value target toward Φ(position) (0 = pure outcome)
    spatial_value: bool, // train a 41-dim SPATIAL value net (global + board summaries)
    threads: usize,      // worker threads (0 = auto: leave ~4 cores free for the desktop)
    timeout_penalty: f64, // value target for a non-decisive game (timeout/tie); 0 = neutral
    win_speed: f64,       // fraction of the win value tied to speed (faster kill = closer to +1)
    vs_hard_frac: f64,    // fraction of self-play games played vs the HARD bot (main-exploiter lever)
    aggression: bool,     // --shaping uses the AGGRESSION potential (push-to-kill) instead of positional Φ
    combined_shaping: bool, // --shaping uses the COMBINED potential (territory + aggression); needs spatial value
    spatial_policy: bool, // Exp I: SPATIAL policy input (cut-value etc.); cold-start, arch DEFAULT_ARCH_SPATIAL
    kl_anchor: f64,       // KL trust-region weight pulling the policy toward the FROZEN warm-start (0 = off)
    dirichlet_alpha: f64, // AlphaZero root Dirichlet noise α (self-play only; 0 = off)
    dirichlet_eps: f64,   // weight of the Dirichlet noise mixed into root priors (0 = off)
    move_temp: f64,       // played-move temperature in self-play (0 = greedy argmax)
    temp_until_round: i64, // apply move_temp while round < this (then greedy)
}
impl Default for Cfg {
    fn default() -> Self {
        Cfg {
            out: PathBuf::from("rust-trainer/checkpoints-az"),
            iters: 200, games: 24, sims: 64, epochs: 4, batch: 64, buffer: 30_000,
            bench_games: 40, bench_every: 2, players: 2, width: 12, height: 12, cap: 120,
            lr_policy: 1e-3, lr_value: 1e-3, seed: 1, leaf_value: false,
            init_policy: None, init_value: None, shaping: 0.0, spatial_value: false, threads: 0,
            timeout_penalty: 0.0, win_speed: 0.0, vs_hard_frac: 0.0, aggression: false, combined_shaping: false, spatial_policy: false,
            kl_anchor: 0.0,
            dirichlet_alpha: 0.0, dirichlet_eps: 0.0, move_temp: 0.0, temp_until_round: 0,
        }
    }
}

fn parse_args() -> Cfg {
    let mut c = Cfg::default();
    let a: Vec<String> = std::env::args().collect();
    let args = if let Some(p) = a.iter().position(|x| x == "--") { a[p + 1..].to_vec() } else { a[1..].to_vec() };
    let mut i = 0;
    while i < args.len() {
        let key = args[i].clone();
        macro_rules! nextval { () => {{ i += 1; args.get(i).cloned().unwrap_or_default() }} }
        match key.as_str() {
            "--out" => c.out = PathBuf::from(nextval!()),
            "--iters" => c.iters = nextval!().parse().unwrap_or(c.iters),
            "--games" => c.games = nextval!().parse().unwrap_or(c.games),
            "--sims" => c.sims = nextval!().parse().unwrap_or(c.sims),
            "--epochs" => c.epochs = nextval!().parse().unwrap_or(c.epochs),
            "--batch" => c.batch = nextval!().parse().unwrap_or(c.batch),
            "--buffer" => c.buffer = nextval!().parse().unwrap_or(c.buffer),
            "--bench-games" => c.bench_games = nextval!().parse().unwrap_or(c.bench_games),
            "--bench-every" => c.bench_every = nextval!().parse().unwrap_or(c.bench_every),
            "--players" => c.players = nextval!().parse().unwrap_or(c.players),
            "--width" => c.width = nextval!().parse().unwrap_or(c.width),
            "--height" => c.height = nextval!().parse().unwrap_or(c.height),
            "--cap" => c.cap = nextval!().parse().unwrap_or(c.cap),
            "--lr" => { let v = nextval!().parse().unwrap_or(c.lr_policy); c.lr_policy = v; c.lr_value = v; }
            "--seed" => c.seed = nextval!().parse().unwrap_or(c.seed),
            "--leaf-value" => c.leaf_value = true,
            "--init-policy" => c.init_policy = Some(PathBuf::from(nextval!())),
            "--init-value" => c.init_value = Some(PathBuf::from(nextval!())),
            "--shaping" => c.shaping = nextval!().parse().unwrap_or(c.shaping),
            "--spatial-value" => c.spatial_value = true,
            "--threads" => c.threads = nextval!().parse().unwrap_or(0),
            "--timeout-penalty" => c.timeout_penalty = nextval!().parse().unwrap_or(c.timeout_penalty),
            "--win-speed" => c.win_speed = nextval!().parse().unwrap_or(c.win_speed),
            "--kl-anchor" => c.kl_anchor = nextval!().parse().unwrap_or(c.kl_anchor),
            "--vs-hard-frac" => c.vs_hard_frac = nextval!().parse::<f64>().unwrap_or(c.vs_hard_frac).clamp(0.0, 1.0),
            "--aggression" => c.aggression = true,
            "--combined-shaping" => c.combined_shaping = true,
            "--spatial-policy" => c.spatial_policy = true,
            "--dirichlet-alpha" => c.dirichlet_alpha = nextval!().parse().unwrap_or(c.dirichlet_alpha),
            "--dirichlet-eps" => c.dirichlet_eps = nextval!().parse().unwrap_or(c.dirichlet_eps),
            "--move-temp" => c.move_temp = nextval!().parse().unwrap_or(c.move_temp),
            "--temp-until-round" => c.temp_until_round = nextval!().parse().unwrap_or(c.temp_until_round),
            _ => {}
        }
        i += 1;
    }
    c
}

fn now_iso() -> String {
    // Minimal UTC ISO-8601 (civil date from Unix days) so dashboard timestamps parse.
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) as i64;
    let days = secs.div_euclid(86400);
    let tod = secs.rem_euclid(86400);
    let (h, mi, s) = (tod / 3600, (tod % 3600) / 60, tod % 60);
    // days since 1970-01-01 → civil (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", y, m, d, h, mi, s)
}

/// Position potential Φ(s) ∈ [-1,1] from the recorded GLOBAL feature vector,
/// encoding the user's "good position" signals (REWARD-DESIGN.md): tile lead
/// (P3/N6, dominant), expansion/domination (P2), economy (P1), military (P5).
/// Indices follow `features::GLOBAL_FEATURE_NAMES`.
fn phi(x: &[f64]) -> f64 {
    let lead = x.get(29).copied().unwrap_or(0.0); // leadMargin ∈ ~[-1,1]
    let dom = (x.get(32).copied().unwrap_or(0.0) - 0.5).clamp(-1.0, 1.0); // dominationProgress centred
    let econ = x.get(4).copied().unwrap_or(0.0).clamp(-1.0, 1.0); // netMoney/100 clamped
    let mil = (x.get(16).copied().unwrap_or(0.0) - x.get(30).copied().unwrap_or(0.0)).clamp(-1.0, 1.0);
    (0.55 * lead + 0.2 * dom + 0.15 * econ + 0.1 * mil).clamp(-1.0, 1.0)
}

/// AGGRESSION potential Φ_aggr(s) ∈ [-1,1] from the 41-dim SPATIAL value vector.
/// Encodes "push toward the kill" (the conversion / stalemate-breaking signal),
/// NOT tile-hoarding — territory/expansion shaping made exp B draw-happy. Reads
/// the spatial dims appended by `features::value_features_spatial`:
///   idx 37 = frontier_fraction (contact with the enemy)
///   idx 38 = enemy_hq_push      (siege pressure: own tiles near the enemy HQ)
///   idx 40 = mean_cut_risk      (overextension: being cut from your own HQ)
/// Aggression is rewarded; overextension is penalised. Only meaningful with a
/// 41-dim (spatial) value vector; falls back to neutral 0 otherwise.
fn phi_aggression(x: &[f64]) -> f64 {
    if x.len() < 41 { return 0.0; }
    let frontier = x[37].clamp(0.0, 1.0);
    let push = x[38].clamp(0.0, 1.0);
    let cut = x[40].clamp(0.0, 1.0);
    let aggr = 0.6 * push + 0.4 * frontier; // [0,1], dominated by HQ siege pressure
    // Map aggression [0,1] → roughly [-0.4, +1.0] with a gentle floor (early-game
    // non-contact is unavoidable, so don't punish it as hard as a loss), minus an
    // overextension penalty.
    (1.4 * aggr - 0.4 - 0.4 * cut).clamp(-1.0, 1.0)
}

/// COMBINED potential: the expansion carrot (territory/lead/econ — `phi`) PLUS the
/// conversion carrot (push-to-kill / frontier − overextension — `phi_aggression`).
/// Territory alone is draw-happy (sit on a lead); aggression alone has no carrot to
/// EXPAND (goes passive). Blending both means the AI is pulled to grow AND to press
/// the enemy — it can never max the potential by just hoarding tiles OR by just
/// defending. Needs the 41-dim spatial value vector for the aggression half (the
/// territory half reads the global-feature prefix, present in both widths).
fn phi_combined(x: &[f64]) -> f64 {
    (0.6 * phi(x) + 0.4 * phi_aggression(x)).clamp(-1.0, 1.0)
}

/// Build a search config. `explore` enables AlphaZero root Dirichlet noise +
/// played-move temperature — set ONLY for self-play data generation, NEVER for the
/// benchmark or the replay recorder (those stay deterministic/greedy = strongest
/// play, comparable metrics, clean replays).
fn sc_for(cfg: &Cfg, leaf_value: bool, seed: u32, explore: bool) -> SearchConfig {
    SearchConfig {
        n_sims: cfg.sims,
        leaf_eval: if leaf_value { LeafEval::Value } else { LeafEval::Static },
        seed,
        spatial_policy: cfg.spatial_policy,
        dirichlet_alpha: if explore { cfg.dirichlet_alpha } else { 0.0 },
        dirichlet_eps: if explore { cfg.dirichlet_eps } else { 0.0 },
        move_temperature: if explore { cfg.move_temp } else { 0.0 },
        temp_until_round: if explore { cfg.temp_until_round } else { 0 },
        ..Default::default()
    }
}

/// Intent-histogram width. 11 intents today (BuildFarm…Pass); slot 11 is reserved
/// for `BuildStrangeDevice` once Phase E adds it as a neural intent — emitting a
/// 12-wide vector now means the dashboard contract does not change when it lands.
const NUM_INTENTS: usize = 12;
const INTENT_NAMES: [&str; NUM_INTENTS] = [
    "BuildFarm", "BuildMine", "BuildVillage", "BuildOutpost", "BuildHydro",
    "BuildNuclear", "Expand", "HireSoldier", "Attack", "StackProducer", "Pass",
    "BuildStrangeDevice",
];

/// How a set of wins splits across the §10 outcome causes (Device/Domination/
/// Conquest/Bankruptcy) plus the harness-side tile-majority Tiebreak. One of
/// these is kept for the champion's wins and one for the hard bot's wins, so the
/// dashboard can show "who won, and HOW" for each side.
#[derive(Default, Clone, Copy)]
struct CauseTally { device: u32, domination: u32, conquest: u32, bankruptcy: u32, tiebreak: u32 }
impl CauseTally {
    fn add_natural(&mut self, c: Option<WinCause>) {
        match c {
            Some(WinCause::Device) => self.device += 1,
            Some(WinCause::Domination) => self.domination += 1,
            Some(WinCause::Conquest) => self.conquest += 1,
            Some(WinCause::Bankruptcy) => self.bankruptcy += 1,
            None => self.conquest += 1, // last-standing fallback (live_players()==1)
        }
    }
    fn json(&self) -> String {
        format!("{{\"device\":{},\"domination\":{},\"conquest\":{},\"bankruptcy\":{},\"tiebreak\":{}}}",
            self.device, self.domination, self.conquest, self.bankruptcy, self.tiebreak)
    }
}

/// Per-game outcome record (one bench game). The champion plays `champ_seat`
/// (alternated across games so win-rate is seat-averaged + reportable by seat).
struct GameRec {
    champ_seat: u8,
    champ_won: bool,
    hard_won: bool,
    true_tie: bool,        // cap reached with equal tiles → genuine non-decisive game
    by_tiebreak: bool,     // resolved by tile-majority at the cap (a win, not a draw)
    cause: Option<WinCause>, // natural win cause (None when by_tiebreak / true_tie)
    champ_frac: f64,
    rounds: i64,
    device_built: bool,    // a Strange Device existed at some point this game
    intents: [u64; NUM_INTENTS],
    decisions: u64,
}

/// Full benchmark outcome. Carries the legacy win/loss/timeout/tile_frac (so the
/// existing dashboard curves keep working) PLUS the §10 outcome-cause breakdown
/// split by winner, device build/survival, by-seat win-rate, intent histogram,
/// and mean rounds-to-resolution per cause.
struct BenchResult {
    n: usize,
    win: f64, loss: f64, timeout: f64, tile_frac: f64,
    wins_seat0: usize, n_seat0: usize, wins_seat1: usize, n_seat1: usize,
    champ_cause: CauseTally, // how the champion's wins split
    hard_cause: CauseTally,  // how the hard bot's wins split
    true_tie: u32,
    device_games: usize,     // games in which a Device was built (either side)
    device_wins: u32,        // games that ended by a Device countdown win
    intents: [u64; NUM_INTENTS],
    decisions: u64,
    // mean rounds-to-resolution per cause: [device, domination, conquest, bankruptcy, tiebreak]
    rounds_sum: [f64; 5], rounds_cnt: [u32; 5],
}

/// Champion (MCTS) vs the held-out HARD heuristic — the 70% oracle. Games run in
/// parallel; the champion's seat alternates by game index so the reported win-rate
/// is seat-averaged (corrects the measured first-mover advantage) and splittable
/// by seat. Records the win CAUSE per side, device build/survival, the champion's
/// intent histogram, and rounds-to-resolution per cause.
fn bench_vs_hard(genome: &Genome, value: Option<&ValueNet>, cfg: &Cfg, games: usize, base_seed: u32) -> BenchResult {
    let sc = sc_for(cfg, value.is_some() && cfg.leaf_value, base_seed ^ 0xB17_C0DE, false);
    let recs: Vec<GameRec> = (0..games)
        .into_par_iter()
        .map(|gi| {
            let seed = base_seed.wrapping_add((gi as u32).wrapping_mul(2_654_435_761));
            let champ_seat: u8 = (gi % 2) as u8; // alternate seats
            let mut g = Game::new(cfg.width, cfg.height, &["P1", "P2"]);
            g.generate_map(cfg.width, cfg.height, seed);
            let champ = match value {
                Some(vn) if cfg.leaf_value => NeuralAiController::with_search_value(genome, TRAINING_CONFIG, sc, vn),
                _ => NeuralAiController::with_search(genome, TRAINING_CONFIG, sc),
            };
            let mut hard = HardAi::hard();
            let mut rng = XorShift32::new(seed);
            for _ in 0..2 {
                let cur = g.current_player();
                if cur.0 == champ_seat as usize { champ.place_headquarters(&mut g, cur); }
                else { hard.place_headquarters(&mut g, cur); }
                g.change_turn();
            }
            let mut intents = [0u64; NUM_INTENTS];
            let mut decisions = 0u64;
            let mut device_built = false;
            let mut winner: Option<PlayerId> = None;
            let mut cause: Option<WinCause> = None;
            {
                let mut sink = |d: DecisionTrace| {
                    decisions += 1;
                    if d.chosen_intent < NUM_INTENTS { intents[d.chosen_intent] += 1; }
                };
                while g.live_players().len() > 1 && g.get_rounds_played() < cfg.cap {
                    let cur = g.current_player();
                    if cur.0 == champ_seat as usize {
                        champ.plan_turn(&mut g, cur, &mut rng, Some(&mut sink));
                    } else {
                        hard.plan_turn(&mut g, cur);
                    }
                    if !device_built && g.has_strange_device() { device_built = true; }
                    match g.end_turn() {
                        EndTurnOutcome::Win(p) => { winner = Some(p); cause = g.last_win_cause(); break; }
                        EndTurnOutcome::Tie => { break; }
                        _ => {}
                    }
                }
            }
            let total = g.get_tile_count().max(1) as f64;
            let champ_frac = g.get_tile_count_for_player(PlayerId(champ_seat as usize)) as f64 / total;
            let hard_frac = g.get_tile_count_for_player(PlayerId(1 - champ_seat as usize)) as f64 / total;
            // Last-standing fallback (someone was eliminated without an explicit Win event).
            let winner = winner.or_else(|| { let l = g.live_players(); if l.len() == 1 { Some(l[0]) } else { None } });

            let mut rec = GameRec {
                champ_seat, champ_won: false, hard_won: false, true_tie: false, by_tiebreak: false,
                cause, champ_frac, rounds: g.get_rounds_played(), device_built, intents, decisions,
            };
            match winner {
                Some(p) => { if p.0 == champ_seat as usize { rec.champ_won = true; } else { rec.hard_won = true; } }
                None => {
                    // Cap reached, both alive → tile-majority tiebreak (a win, not a draw).
                    rec.cause = None;
                    if champ_frac > hard_frac { rec.champ_won = true; rec.by_tiebreak = true; }
                    else if hard_frac > champ_frac { rec.hard_won = true; rec.by_tiebreak = true; }
                    else { rec.true_tie = true; }
                }
            }
            rec
        })
        .collect();

    let n = games.max(1);
    let mut r = BenchResult {
        n, win: 0.0, loss: 0.0, timeout: 0.0, tile_frac: 0.0,
        wins_seat0: 0, n_seat0: 0, wins_seat1: 0, n_seat1: 0,
        champ_cause: CauseTally::default(), hard_cause: CauseTally::default(), true_tie: 0,
        device_games: 0, device_wins: 0, intents: [0; NUM_INTENTS], decisions: 0,
        rounds_sum: [0.0; 5], rounds_cnt: [0; 5],
    };
    let mut wins = 0usize; let mut losses = 0usize; let mut ties = 0usize; let mut tf_sum = 0.0;
    for rec in &recs {
        tf_sum += rec.champ_frac;
        for k in 0..NUM_INTENTS { r.intents[k] += rec.intents[k]; }
        r.decisions += rec.decisions;
        if rec.device_built { r.device_games += 1; }
        if matches!(rec.cause, Some(WinCause::Device)) { r.device_wins += 1; }
        // by-seat
        if rec.champ_seat == 0 { r.n_seat0 += 1; if rec.champ_won { r.wins_seat0 += 1; } }
        else { r.n_seat1 += 1; if rec.champ_won { r.wins_seat1 += 1; } }
        // cause split + rounds-to-resolution
        let cause_idx = if rec.by_tiebreak { Some(4) } else {
            match rec.cause { Some(WinCause::Device) => Some(0), Some(WinCause::Domination) => Some(1),
                Some(WinCause::Conquest) => Some(2), Some(WinCause::Bankruptcy) => Some(3), None => Some(2) }
        };
        if rec.champ_won {
            wins += 1;
            if rec.by_tiebreak { r.champ_cause.tiebreak += 1; } else { r.champ_cause.add_natural(rec.cause); }
        } else if rec.hard_won {
            losses += 1;
            if rec.by_tiebreak { r.hard_cause.tiebreak += 1; } else { r.hard_cause.add_natural(rec.cause); }
        } else if rec.true_tie {
            ties += 1; r.true_tie += 1;
        }
        if !rec.true_tie {
            if let Some(ci) = cause_idx { r.rounds_sum[ci] += rec.rounds as f64; r.rounds_cnt[ci] += 1; }
        }
    }
    let nf = n as f64;
    r.win = wins as f64 / nf;
    r.loss = losses as f64 / nf;
    r.timeout = ties as f64 / nf; // "timeout" now == true non-decisive ties (target ≈ 0)
    r.tile_frac = tf_sum / nf;
    r
}

// --- game-replay recorder (dashboard "watch a game" viewer) ----------------
// Single short character code per building, for the compact replay frames.
fn building_code(k: BuildingType) -> char {
    match k {
        BuildingType::Farm => 'F',
        BuildingType::Mine => 'M',
        BuildingType::Village => 'V',
        BuildingType::Outpost => 'O',
        BuildingType::Hydro => 'H',
        BuildingType::Nuclear => 'N',
        BuildingType::StrangeDevice => 'D',
        BuildingType::Headquarters => 'Q',
        BuildingType::Mikontalo => 'K',
        _ => '?',
    }
}

/// One board snapshot as a compact JSON object: three equal-length strings
/// indexed in `get_tiles()` (column-major, index = x*height + y) order —
/// `own` (owner: '0' neutral / '1' seat0 / '2' seat1), `bld` (building code or
/// '.'), `sol` (soldier count digit, clamped 0–9, or '.'). `r` = round, `p` =
/// the seat whose turn just resolved.
fn capture_frame(g: &Game, round: i64, cur: usize) -> String {
    let tiles = g.get_tiles();
    let mut own = String::with_capacity(tiles.len());
    let mut bld = String::with_capacity(tiles.len());
    let mut sol = String::with_capacity(tiles.len());
    for t in tiles {
        own.push(match t.owner { None => '0', Some(p) => (b'1' + p.0 as u8) as char });
        bld.push(match &t.building { Some(b) => building_code(b.kind), None => '.' });
        let s = t.units.iter().filter(|&&u| g.units[u.0].kind == UnitType::Soldier).count();
        sol.push(if s == 0 { '.' } else { std::char::from_digit(s.min(9) as u32, 10).unwrap() });
    }
    format!("{{\"r\":{},\"p\":{},\"own\":\"{}\",\"bld\":\"{}\",\"sol\":\"{}\"}}", round, cur, own, bld, sol)
}

/// Play ONE game and serialize a per-turn board replay for the dashboard's animated
/// viewer. Seat 0 (blue) is always our champion. Seat 1 (red) is the HARD bot when
/// `vs_self` is false (`mode:"hard"`), or a SECOND copy of the champion when
/// `vs_self` is true (`mode:"self"`, AI-vs-AI self-play). Champion always takes
/// seat 0 so the viewer is consistent across iterations.
fn record_replay(genome: &Genome, value: Option<&ValueNet>, cfg: &Cfg, iter: usize, seed: u32, vs_self: bool) -> String {
    let leaf = value.is_some() && cfg.leaf_value;
    let sc = sc_for(cfg, leaf, seed ^ 0x12E_5AFE, false);
    let mut g = Game::new(cfg.width, cfg.height, &["P1", "P2"]);
    g.generate_map(cfg.width, cfg.height, seed);
    let mk = |s: SearchConfig| match value {
        Some(vn) if leaf => NeuralAiController::with_search_value(genome, TRAINING_CONFIG, s, vn),
        _ => NeuralAiController::with_search(genome, TRAINING_CONFIG, s),
    };
    let champ = mk(sc);
    // Seat-1 opponent: a second champion copy (self-play) or the HARD bot. The
    // self-play opponent uses a distinct search seed so the two sides aren't a
    // trivial mirror.
    let champ2 = mk(sc_for(cfg, leaf, seed ^ 0x5A17_C0DE, false));
    let mut hard = HardAi::hard();
    let mut rng = XorShift32::new(seed);
    for _ in 0..2 {
        let cur = g.current_player();
        if cur.0 == 0 { champ.place_headquarters(&mut g, cur); }
        else if vs_self { champ2.place_headquarters(&mut g, cur); }
        else { hard.place_headquarters(&mut g, cur); }
        g.change_turn();
    }
    // Static terrain (one char per tile, get_tiles() order) — captured once.
    let terrain: String = g
        .get_tiles()
        .iter()
        .map(|t| match t.tile_type {
            TileType::Grassland => 'g',
            TileType::Forest => 'f',
            TileType::AbundantForest => 'a',
            TileType::Mountain => 'm',
            TileType::River => 'r',
        })
        .collect();
    let mut frames: Vec<String> = vec![capture_frame(&g, g.get_rounds_played(), 9)]; // 9 = setup frame
    let mut winner: Option<PlayerId> = None;
    let mut cause: Option<WinCause> = None;
    // Stalemate cut, mirroring selfplay: a frozen game (no board change for
    // STALL_ROUNDS, no Device standing) ends now instead of recording hundreds of
    // Pass turns — so the dashboard viewer doesn't show misleading 300-round
    // freezes (the trainer's own games are already cut this way).
    let mut last_sig = cp_ai::selfplay::board_signature(&g, 2);
    let mut last_progress_round = g.get_rounds_played();
    while g.live_players().len() > 1 && g.get_rounds_played() < cfg.cap {
        let cur = g.current_player();
        let seat = cur.0;
        if seat == 0 { champ.plan_turn(&mut g, cur, &mut rng, None); }
        else if vs_self { champ2.plan_turn(&mut g, cur, &mut rng, None); }
        else { hard.plan_turn(&mut g, cur); }
        match g.end_turn() {
            EndTurnOutcome::Win(p) => { winner = Some(p); cause = g.last_win_cause(); frames.push(capture_frame(&g, g.get_rounds_played(), seat)); break; }
            EndTurnOutcome::Tie => { frames.push(capture_frame(&g, g.get_rounds_played(), seat)); break; }
            _ => { frames.push(capture_frame(&g, g.get_rounds_played(), seat)); }
        }
        let r = g.get_rounds_played();
        let sig = cp_ai::selfplay::board_signature(&g, 2);
        if sig != last_sig {
            last_sig = sig;
            last_progress_round = r;
        } else if r - last_progress_round >= cp_ai::selfplay::STALL_ROUNDS && !cp_ai::selfplay::device_on_board(&g) {
            break;
        }
    }
    let winner = winner.or_else(|| { let l = g.live_players(); if l.len() == 1 { Some(l[0]) } else { None } });
    let winner_seat: i64 = match winner { Some(p) => p.0 as i64, None => -1 };
    let cause_str = match cause {
        Some(WinCause::Device) => "device", Some(WinCause::Domination) => "domination",
        Some(WinCause::Conquest) => "conquest", Some(WinCause::Bankruptcy) => "bankruptcy",
        None => if winner.is_some() { "conquest" } else { "tiebreak/tie" },
    };
    format!(
        "{{\"iter\":{},\"seed\":{},\"mode\":\"{}\",\"width\":{},\"height\":{},\"champSeat\":0,\"terrain\":\"{}\",\
         \"result\":{{\"winnerSeat\":{},\"cause\":\"{}\",\"rounds\":{}}},\"frames\":[{}]}}",
        iter, seed, if vs_self { "self" } else { "hard" }, cfg.width, cfg.height, terrain, winner_seat, cause_str, g.get_rounds_played(), frames.join(","))
}

fn append_line(path: &PathBuf, line: &str) {
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(f, "{line}");
    }
}

fn main() {
    let cfg = parse_args();
    create_dir_all(&cfg.out).expect("create out dir");

    // Worker threads: explicit --threads, else auto = cores - 4 (leave headroom so
    // the desktop stays responsive while training).
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let threads = if cfg.threads > 0 { cfg.threads } else { cores.saturating_sub(4).max(1) };
    rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().ok();
    println!("alphazero: {threads} worker threads ({cores} cores detected, leaving headroom)");

    // Truncate dashboard artifacts at startup so a fresh run never mixes its
    // gen-series with a previous run's leftover lines (which jumbles the charts).
    let _ = std::fs::write(cfg.out.join("log.jsonl"), "");
    let _ = std::fs::write(cfg.out.join("benchmark-history.jsonl"), "");

    let start = Instant::now();
    let mut rng = SplitMix64(cfg.seed ^ 0x5EED);

    // Init nets (random — gradient training from scratch).
    // Exp I: the spatial policy has a different input dim than the shipped 63-dim
    // net, so it cannot warm-start — it COLD-STARTS from random (the value net
    // still warm-starts, giving MCTS a sane leaf signal during the cold policy's
    // early learning).
    let arch = if cfg.spatial_policy { DEFAULT_ARCH_SPATIAL.to_vec() } else { DEFAULT_ARCH.to_vec() };
    let init_genome = if cfg.spatial_policy {
        // WARM-START the spatial policy by transplanting a 63-dim champion (so it
        // begins at ~that champion's strength and only LEARNS the spatial features),
        // rather than cold-starting from random (which stalls ~5%).
        match &cfg.init_policy {
            Some(p) => {
                let base = Genome::from_file(&p.to_string_lossy()).expect("load --init-policy genome");
                if base.arch == DEFAULT_ARCH.to_vec() {
                    let g = cp_ai::policy_spatial::warmstart_spatial(&base);
                    println!("alphazero: WARM-START spatial policy from {} (transplant {:?} -> {:?}, 6 spatial weights = 0; init == base)", p.display(), base.arch, g.arch);
                    g
                } else {
                    println!("alphazero: --init-policy arch {:?} != [63,24,16,1]; COLD-START random spatial policy", base.arch);
                    random_genome(&arch, cfg.seed ^ 0x9001)
                }
            }
            None => {
                println!("alphazero: COLD-START random spatial policy arch {:?}", arch);
                random_genome(&arch, cfg.seed ^ 0x9001)
            }
        }
    } else {
        match &cfg.init_policy {
            Some(p) => {
                let g = Genome::from_file(&p.to_string_lossy()).expect("load --init-policy genome");
                println!("alphazero: warm-start policy from {} (arch {:?})", p.display(), g.arch);
                g
            }
            None => random_genome(&arch, cfg.seed ^ 0x9001),
        }
    };
    // KL trust-region anchor: freeze a copy of the warm-start as the reference
    // the policy is pulled toward each step (guards against the self-play drift
    // to passivity). Only meaningful with a warm-start init.
    let kl_ref = if cfg.kl_anchor > 0.0 { Some(init_genome.clone()) } else { None };
    let mut pol = PolicyTrainer::new(init_genome, cfg.lr_policy);
    pol.l2 = 1e-5;
    if let Some(r) = kl_ref {
        println!("alphazero: KL trust-region anchor λ={:.2} → policy pulled toward the frozen warm-start each step", cfg.kl_anchor);
        pol.ref_genome = Some(r);
        pol.kl_coeff = cfg.kl_anchor;
    }
    let init_value = if cfg.spatial_value {
        // Warm-start the 41-dim spatial value net from a compatible checkpoint if
        // given (avoids the random-value "rough start"); else from scratch.
        match &cfg.init_value {
            Some(p) => {
                let v = ValueNet::from_file(&p.to_string_lossy()).expect("load --init-value net");
                if v.arch.first() == VALUE_ARCH_SPATIAL.first() {
                    println!("alphazero: warm-start SPATIAL value net from {} (arch {:?})", p.display(), v.arch);
                    v
                } else {
                    println!("alphazero: --init-value dim mismatch ({:?} != spatial); random spatial value", v.arch);
                    ValueNet::random_arch(&VALUE_ARCH_SPATIAL, cfg.seed ^ 0x07A1)
                }
            }
            None => {
                println!("alphazero: SPATIAL value net {:?} (trains from scratch)", VALUE_ARCH_SPATIAL);
                ValueNet::random_arch(&VALUE_ARCH_SPATIAL, cfg.seed ^ 0x07A1)
            }
        }
    } else {
        match &cfg.init_value {
            Some(p) => ValueNet::from_file(&p.to_string_lossy()).expect("load --init-value net"),
            None => ValueNet::random(cfg.seed ^ 0x07A1),
        }
    };
    let mut val = ValueTrainer::new(init_value, cfg.lr_value);

    let mut pbuf: VecDeque<PolicyExample> = VecDeque::new();
    let mut vbuf: VecDeque<ValueExample> = VecDeque::new();

    let log_path = cfg.out.join("log.jsonl");
    let bench_hist = cfg.out.join("benchmark-history.jsonl");

    println!("alphazero: out={} iters={} games/iter={} sims={} bench every {} ({} games)",
        cfg.out.display(), cfg.iters, cfg.games, cfg.sims, cfg.bench_every, cfg.bench_games);
    if cfg.vs_hard_frac > 0.0 {
        let n_vs_hard = (cfg.games as f64 * cfg.vs_hard_frac).round() as usize;
        println!("alphazero: vs-hard-frac {:.2} → {}/{} games/iter vs the HARD bot (main exploiter)",
            cfg.vs_hard_frac, n_vs_hard, cfg.games);
    }
    if cfg.shaping > 0.0 {
        let mode = if cfg.combined_shaping { "COMBINED: territory + push-to-kill" }
            else if cfg.aggression { "AGGRESSION: push-to-kill" }
            else { "positional: lead/dom/econ/mil" };
        println!("alphazero: potential-based Φ shaping (z − λΦ, λ={:.2}) ({})", cfg.shaping, mode);
    }
    if cfg.dirichlet_eps > 0.0 && cfg.dirichlet_alpha > 0.0 {
        println!("alphazero: root exploration — Dirichlet(α={:.2}) ε={:.2} mixed into self-play root priors",
            cfg.dirichlet_alpha, cfg.dirichlet_eps);
    }
    if cfg.move_temp > 0.0 {
        println!("alphazero: played-move temperature τ={:.2} until round {} (then greedy)",
            cfg.move_temp, cfg.temp_until_round);
    }

    // Track the best benchmark so far so an unattended long run never loses its
    // peak to later drift (Exp E showed runs can regress). The best policy+value
    // are mirrored to champion-best.json / value-best.json.
    let mut best_win = -1.0f64;
    for iter in 0..cfg.iters {
        // --- self-play (parallel across cores; games are independent) -------
        let base = smix(cfg.seed ^ (iter as u64).wrapping_mul(0x9E3779B97F4A7C15));
        // Main-exploiter lever: the first `n_vs_hard` games of each iter pit our net
        // (seat 0) against the held-out HARD bot; the rest are pure self-play. Both
        // share the same seed stream, which is disjoint (different hash) from the
        // bench-vs-hard stream (`^0xBE0`), so the held-out eval stays honest.
        let n_vs_hard = (cfg.games as f64 * cfg.vs_hard_frac).round() as usize;
        let results: Vec<SelfPlayData> = (0..cfg.games)
            .into_par_iter()
            .map(|gi| {
                let seed = smix(base ^ gi as u64) as u32;
                let sc = sc_for(&cfg, cfg.leaf_value, seed, true);
                if gi < n_vs_hard {
                    play_one_game_vs_hard(seed, cfg.width, cfg.height,
                        &pol.genome, &TRAINING_CONFIG, &sc, Some(&val.net), cfg.cap, cfg.spatial_value,
                        cfg.timeout_penalty, cfg.win_speed)
                } else {
                    play_one_game(seed, cfg.width, cfg.height, cfg.players,
                        &pol.genome, &TRAINING_CONFIG, &sc, Some(&val.net), cfg.cap, cfg.spatial_value,
                        cfg.timeout_penalty, cfg.win_speed)
                }
            })
            .collect();
        let mut new_examples = 0usize;
        for data in results {
            new_examples += data.policy.len();
            for (pe, ve) in data.policy.into_iter().zip(data.value.into_iter()) {
                // POTENTIAL-BASED reward shaping (Ng, Harada & Russell 1999): with the
                // standard terminal convention Φ(s_T)≜0 and γ=1, the Monte-Carlo value
                // target telescopes to  V*(s) − Φ(s) = z − λ·Φ(s).  This rewards
                // INCREASING Φ per step (expand / push to the kill) and is provably
                // policy-invariant. The OLD blend (1−λ)z + λΦ had the Φ term with the
                // WRONG SIGN — it rewarded merely BEING in a high-Φ state, which created
                // the "sit on the lead" / draw-happy attractor we observed.
                // --combined = territory+push, --aggression = push-only, else positional.
                let z = if cfg.shaping > 0.0 {
                    let p = if cfg.combined_shaping { phi_combined(&ve.x) }
                        else if cfg.aggression { phi_aggression(&ve.x) }
                        else { phi(&ve.x) };
                    (ve.z - cfg.shaping * p).clamp(-1.0, 1.0)
                } else {
                    ve.z
                };
                pbuf.push_back(pe);
                vbuf.push_back(ValueExample { x: ve.x, z });
                if pbuf.len() > cfg.buffer { pbuf.pop_front(); vbuf.pop_front(); }
            }
        }

        // --- train ----------------------------------------------------------
        let n = pbuf.len();
        let mut ploss = 0.0; let mut vloss = 0.0; let mut steps = 0usize;
        if n >= cfg.batch {
            let pvec: Vec<&PolicyExample> = pbuf.iter().collect();
            let vvec: Vec<&ValueExample> = vbuf.iter().collect();
            for _ in 0..cfg.epochs {
                // shuffle index order (Fisher–Yates)
                let mut idx: Vec<usize> = (0..n).collect();
                for k in (1..n).rev() { idx.swap(k, rng.below(k + 1)); }
                let mut s = 0;
                while s + cfg.batch <= n {
                    let sl = &idx[s..s + cfg.batch];
                    let pb: Vec<PolicyExample> = sl.iter().map(|&k| pvec[k].clone()).collect();
                    let vb: Vec<ValueExample> = sl.iter().map(|&k| vvec[k].clone()).collect();
                    ploss += pol.step(&pb); vloss += val.step(&vb); steps += 1;
                    s += cfg.batch;
                }
            }
        }
        if steps > 0 { ploss /= steps as f64; vloss /= steps as f64; }

        // --- training-health metrics (§10.1 ★1 policy entropy, ★2 value calib) ---
        // Policy entropy: mean NORMALISED entropy (÷ln K) of the net's candidate
        // softmax over a sample of recent buffer states. 1 ≈ uniform/exploratory,
        // →0 ≈ collapsed / over-confident → the policy-freeze signal we caught only
        // by hand last arc. Value calibration: mean PREDICTED value bucketed by the
        // TRUE outcome — a healthy net drives win→+1, loss→−1, draw→~0; all three
        // collapsing toward 0 is the draw-collapse signature (read directly here
        // rather than inferred from valueLoss→0.1).
        let psample = pbuf.len().min(1024);
        let (mut ent_sum, mut ent_cnt) = (0.0f64, 0usize);
        for ex in pbuf.iter().rev().take(psample) {
            let k = ex.inputs.len();
            if k < 2 { continue; }
            let scores: Vec<f64> = ex.inputs.iter().map(|x| score(&pol.genome, x)).collect();
            let p = softmax(&scores);
            let h: f64 = p.iter().filter(|&&v| v > 0.0).map(|&v| -v * v.ln()).sum();
            ent_sum += h / (k as f64).ln();
            ent_cnt += 1;
        }
        let policy_entropy = if ent_cnt > 0 { ent_sum / ent_cnt as f64 } else { 0.0 };
        let vsample = vbuf.len().min(1024);
        let (mut vw, mut nw, mut vl, mut nl, mut vd, mut nd) = (0.0f64, 0usize, 0.0f64, 0usize, 0.0f64, 0usize);
        for ve in vbuf.iter().rev().take(vsample) {
            let pred = val.net.forward(&ve.x);
            if ve.z > 0.33 { vw += pred; nw += 1; }
            else if ve.z < -0.33 { vl += pred; nl += 1; }
            else { vd += pred; nd += 1; }
        }
        let jmean = |s: f64, c: usize| if c > 0 { format!("{:.4}", s / c as f64) } else { "null".to_string() };

        // --- log line (dashboard Koulutus tab) ------------------------------
        let elapsed = start.elapsed().as_secs_f64();
        let gps = if elapsed > 0.0 { ((iter + 1) * cfg.games) as f64 / elapsed } else { 0.0 };
        append_line(&log_path, &format!(
            "{{\"gen\":{},\"bestFit\":null,\"meanFit\":null,\"medianFit\":null,\"fitStd\":null,\
             \"policyLoss\":{:.5},\"valueLoss\":{:.5},\"bufferSize\":{},\"newExamples\":{},\
             \"policyEntropy\":{:.4},\"valPredWin\":{},\"valPredLoss\":{},\"valPredDraw\":{},\
             \"gamesPerSec\":{:.3},\"elapsedSec\":{:.1},\"winRateVsHeur\":null}}",
            iter, ploss, vloss, n, new_examples,
            policy_entropy, jmean(vw, nw), jmean(vl, nl), jmean(vd, nd),
            gps, elapsed));

        // --- periodic game replays (dashboard "watch a game" viewer) ---------
        // Every 5 iters, record REPLAY_GAMES FRESH games per source (each on a
        // DIFFERENT map seed) from the CURRENT net, written as a JSON array so the
        // dashboard can browse several games from this same checkpoint — not one
        // live game, and never stale ones. Seat 0 (blue) is always our champion;
        // seat 1 is the HARD bot (`replay.json`) or a 2nd champion (`replay_selfplay.json`).
        if iter % 5 == 0 || iter + 1 == cfg.iters {
            const REPLAY_GAMES: usize = 5;
            let mut hard_arr = Vec::with_capacity(REPLAY_GAMES);
            let mut self_arr = Vec::with_capacity(REPLAY_GAMES);
            for gi in 0..REPLAY_GAMES as u64 {
                let hseed = smix(cfg.seed ^ (iter as u64).wrapping_mul(0x9E37_79B1) ^ 0x9E_F00D ^ gi.wrapping_mul(0x2545_F491)) as u32;
                hard_arr.push(record_replay(&pol.genome, Some(&val.net), &cfg, iter, hseed, false));
                let sseed = smix(cfg.seed ^ (iter as u64).wrapping_mul(0x85EB_CA77) ^ 0x5E1F ^ gi.wrapping_mul(0x9E37_79B1)) as u32;
                self_arr.push(record_replay(&pol.genome, Some(&val.net), &cfg, iter, sseed, true));
            }
            let _ = std::fs::write(cfg.out.join("replay.json"), format!("[{}]", hard_arr.join(",")));
            let _ = std::fs::write(cfg.out.join("replay_selfplay.json"), format!("[{}]", self_arr.join(",")));
        }

        // --- periodic benchmark vs hard + checkpoint ------------------------
        if iter % cfg.bench_every == 0 || iter + 1 == cfg.iters {
            let br = bench_vs_hard(&pol.genome, Some(&val.net), &cfg, cfg.bench_games, cfg.seed as u32 ^ 0xBE0);
            // derived fields (null when the denominator is empty)
            let seat = |w: usize, m: usize| if m > 0 { format!("{:.4}", w as f64 / m as f64) } else { "null".to_string() };
            let dsurv = if br.device_games > 0 { format!("{:.4}", br.device_wins as f64 / br.device_games as f64) } else { "null".to_string() };
            let rmean = |i: usize| if br.rounds_cnt[i] > 0 { format!("{:.1}", br.rounds_sum[i] / br.rounds_cnt[i] as f64) } else { "null".to_string() };
            let mut intents_json = String::from("{");
            for k in 0..NUM_INTENTS {
                if k > 0 { intents_json.push(','); }
                intents_json.push_str(&format!("\"{}\":{}", INTENT_NAMES[k], br.intents[k]));
            }
            intents_json.push('}');
            append_line(&bench_hist, &format!(
                "{{\"gen\":{},\"winRate\":{:.4},\"lossRate\":{:.4},\"timeoutRate\":{:.4},\"tileFrac\":{:.4},\
                 \"nGames\":{},\"winSeat0\":{},\"winSeat1\":{},\
                 \"champWins\":{},\"hardWins\":{},\"trueTie\":{},\
                 \"deviceBuildRate\":{:.4},\"deviceSurvival\":{},\
                 \"roundsByCause\":{{\"device\":{},\"domination\":{},\"conquest\":{},\"bankruptcy\":{},\"tiebreak\":{}}},\
                 \"intents\":{},\"decisions\":{},\"ts\":\"{}\"}}",
                iter, br.win, br.loss, br.timeout, br.tile_frac,
                br.n, seat(br.wins_seat0, br.n_seat0), seat(br.wins_seat1, br.n_seat1),
                br.champ_cause.json(), br.hard_cause.json(), br.true_tie,
                br.device_games as f64 / br.n as f64, dsurv,
                rmean(0), rmean(1), rmean(2), rmean(3), rmean(4),
                intents_json, br.decisions, now_iso()));
            // checkpoint (latest)
            let _ = std::fs::write(cfg.out.join("champion.json"), pol.genome.to_json());
            let _ = val.net.to_file(&cfg.out.join("value.json").to_string_lossy());
            // checkpoint (best-so-far) — never lose the peak to later drift.
            let mut tag = "";
            if br.win > best_win {
                best_win = br.win;
                let _ = std::fs::write(cfg.out.join("champion-best.json"), pol.genome.to_json());
                let _ = val.net.to_file(&cfg.out.join("value-best.json").to_string_lossy());
                tag = " *BEST*";
            }
            let cc = &br.champ_cause;
            println!("iter {iter}: vs-hard win {:.1}% (loss {:.1}%, tie {:.1}%) | champ wins D{} Dom{} C{} B{} TB{} | dev {:.0}% built surv {} | H{:.2} | ploss {:.4} vloss {:.4} | buf {} | {:.0}s{}",
                br.win * 100.0, br.loss * 100.0, br.timeout * 100.0,
                cc.device, cc.domination, cc.conquest, cc.bankruptcy, cc.tiebreak,
                100.0 * br.device_games as f64 / br.n as f64, dsurv,
                policy_entropy, ploss, vloss, n, elapsed, tag);
        }
    }

    let _ = std::fs::write(cfg.out.join("champion.json"), pol.genome.to_json());
    let _ = val.net.to_file(&cfg.out.join("value.json").to_string_lossy());
    println!("alphazero: done in {:.0}s → {}", start.elapsed().as_secs_f64(), cfg.out.display());
}
