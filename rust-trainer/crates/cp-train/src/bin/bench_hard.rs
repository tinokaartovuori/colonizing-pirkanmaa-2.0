//! `bench_hard` — the pivotal vs-HARD-heuristic measurement.
//!
//! Seat 0 = the champion genome (no-search OR test-time MCTS), seat 1 = the
//! held-out HARD heuristic (`cp_ai::HardAi`, a faithful Rust port of
//! `src/managers/ai.ts` `PARAMS.hard`). This is the Rust analogue of
//! `training/benchmark.ts` (champion vs TS-hard) — the decisive measurement for
//! whether test-time search can reach a high win-rate vs the heuristic.
//!
//! Two curriculum modes:
//!   --curriculum bench   (default) mirrors `training/benchmark.ts`'s SIZES /
//!                        seed / round-cap distribution, so the no-search number
//!                        is directly comparable to the TS oracle.
//!   --curriculum train   reuses bench.rs's SplitMix64 curriculum (for parity
//!                        with the self-play A/B harness).
//!
//! Seat 0 mode:
//!   --search none        (default) plain policy argmax (the no-search baseline).
//!   --search rollout     rollout-MCTS (--sims, --rollout horizon).
//!   --search static      static-leaf MCTS.
//!   --search value       value-net-leaf MCTS (--value <path>).
//!
//! Usage:
//!   cargo run --release -p cp-train --bin bench_hard -- \
//!     --champion rust-trainer/checkpoints/champion.json \
//!     --search none --games 80 --seed 1
//!   cargo run --release -p cp-train --bin bench_hard -- \
//!     --search rollout --sims 120 --rollout 6 --games 24 --seed 1

use std::path::PathBuf;
use std::time::Instant;

use cp_ai::controller::NeuralAiController;
use cp_ai::mlp::Genome;
use cp_ai::policy::XorShift32;
use cp_ai::{HardAi, LeafEval, SearchConfig, ValueNet, TRAINING_CONFIG};
use cp_sim::{EndTurnOutcome, Game, PlayerId};

// ---------------------------------------------------------------------------
// Curricula.
// ---------------------------------------------------------------------------

struct GameSpec {
    seed: u32,
    width: i32,
    height: i32,
    cap: i64,
}

/// `training/benchmark.ts` curriculum. It draws from a `makeRng(baseSeed)`
/// xorshift32 (the SAME RNG the TS harness uses); we replicate it bit-for-bit so
/// the map/seed/cap stream MATCHES the TS oracle's, giving a directly comparable
/// no-search win-rate.
struct BenchRng {
    s: u32,
}
impl BenchRng {
    /// `makeRng(seed)` from training/harness.ts.
    fn new(seed: u32) -> Self {
        let mut s = seed.wrapping_mul(2654435761);
        if s == 0 {
            s = 0x9e3779b9;
        }
        BenchRng { s }
    }
    fn next_f64(&mut self) -> f64 {
        // Mirror the TS `makeRng` (training/harness.ts) bit-for-bit. CRUCIAL:
        // JS `>>` is a SIGNED (arithmetic) int32 shift, and at that point `s`
        // (after the prior `>>>= 0`) may have bit 31 set, so it is treated as a
        // negative int32 and sign-extends. A plain Rust `u32 >> 17` (logical)
        // would diverge. We replicate by casting to i32 for the `>> 17`.
        self.s ^= self.s << 13; // u32 wraps to 32 bits, == JS `<<` + `>>>=0`
        self.s = (self.s ^ (((self.s as i32) >> 17) as u32)) & 0xFFFF_FFFF;
        self.s ^= self.s << 5; // wraps to 32 bits, == JS `<<` + `>>>=0`
        (self.s as f64) / 4294967296.0
    }
}

/// The benchmark.ts SIZES table.
const BENCH_SIZES: [(i32, i32); 8] = [
    (12, 12),
    (12, 12),
    (12, 12),
    (14, 12),
    (14, 12),
    (16, 14),
    (18, 14),
    (20, 15),
];

/// Generate the FULL bench curriculum stream from one base seed, exactly as
/// `training/benchmark.ts main()` does (3 draws per game, in order).
fn bench_curriculum(base_seed: u32, games: usize) -> Vec<GameSpec> {
    let mut r = BenchRng::new(base_seed);
    let mut out = Vec::with_capacity(games);
    for _ in 0..games {
        let (w, h) = BENCH_SIZES[(r.next_f64() * BENCH_SIZES.len() as f64).floor() as usize];
        let seed = 1 + (r.next_f64() * 1000.0).floor() as u32;
        let cap = if r.next_f64() < 0.12 { 180 } else { 80 };
        out.push(GameSpec {
            seed,
            width: w,
            height: h,
            cap,
        });
    }
    out
}

// bench.rs's SplitMix64 curriculum (kept for cross-checking against the A/B bin).
struct SplitMix64 {
    state: u64,
}
impl SplitMix64 {
    fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn train_curriculum(master: u64, base_cap: i64, games: usize) -> Vec<GameSpec> {
    (0..games)
        .map(|i| {
            let mut r =
                SplitMix64::new(master ^ ((i as u64).wrapping_mul(0x165667B19E3779F9)));
            let size_roll = r.next_f64();
            let (width, height) = if size_roll < 0.5 {
                (12, 12)
            } else if size_roll < 0.8 {
                (16, 14)
            } else if size_roll < 0.93 {
                (18, 14)
            } else {
                (20, 15)
            };
            let cap = if r.next_f64() < 0.15 {
                base_cap.max(300)
            } else {
                base_cap
            };
            let seed = (r.next_u64() as u32) | 1;
            GameSpec {
                seed,
                width,
                height,
                cap,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// One game: seat 0 = champion (no-search or MCTS), seat 1 = HardAi.
// ---------------------------------------------------------------------------

struct GameOut {
    winner_num: Option<i64>,
    timeout: bool,
    seat0_tile_frac: f64,
    rounds: i64,
}

#[derive(Clone, Copy)]
enum Seat0Mode {
    NoSearch,
    Search(SearchConfig),
}

fn play_game(
    spec: &GameSpec,
    genome: &Genome,
    mode: Seat0Mode,
    value_net: Option<&ValueNet>,
) -> GameOut {
    let mut g = Game::new(spec.width, spec.height, &["P1", "P2"]);
    g.generate_map(spec.width, spec.height, spec.seed);

    let champ = match mode {
        Seat0Mode::NoSearch => NeuralAiController::new(genome, TRAINING_CONFIG),
        Seat0Mode::Search(sc) => match value_net {
            Some(vn) => NeuralAiController::with_search_value(genome, TRAINING_CONFIG, sc, vn),
            None => NeuralAiController::with_search(genome, TRAINING_CONFIG, sc),
        },
    };
    let mut hard = HardAi::hard();
    let mut rng = XorShift32::new(spec.seed);

    // HQ placement per seat.
    for _ in 0..2 {
        let cur = g.current_player();
        if cur.0 == 0 {
            champ.place_headquarters(&mut g, cur);
        } else {
            hard.place_headquarters(&mut g, cur);
        }
        g.change_turn();
    }

    let mut winner: Option<PlayerId> = None;
    let mut tie = false;
    while g.live_players().len() > 1 && g.get_rounds_played() < spec.cap {
        let cur = g.current_player();
        if cur.0 == 0 {
            champ.plan_turn(&mut g, cur, &mut rng, None);
        } else {
            hard.plan_turn(&mut g, cur);
        }
        match g.end_turn() {
            EndTurnOutcome::Win(p) => {
                winner = Some(p);
                break;
            }
            EndTurnOutcome::Tie => {
                tie = true;
                break;
            }
            _ => {}
        }
    }

    let total = g.get_tile_count().max(1);
    let seat0_tile_frac = g.get_tile_count_for_player(PlayerId(0)) as f64 / total as f64;
    let (winner_num, timeout) = if let Some(w) = winner {
        (Some(g.players[w.0].player_num), false)
    } else if tie {
        (None, false)
    } else if g.live_players().len() == 1 {
        let w = g.live_players()[0];
        (Some(g.players[w.0].player_num), false)
    } else {
        (None, true)
    };
    GameOut {
        winner_num,
        timeout,
        seat0_tile_frac,
        rounds: g.get_rounds_played(),
    }
}

// ---------------------------------------------------------------------------
// CLI.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum SearchMode {
    None,
    Rollout,
    Static,
    Value,
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum Curriculum {
    Bench,
    Train,
}

struct Config {
    champion: PathBuf,
    games: usize,
    sims: usize,
    seed: u64,
    cap: i64,
    rollout: Option<i64>,
    search: SearchMode,
    value: Option<PathBuf>,
    curriculum: Curriculum,
    /// Force a fixed NxN map for every game (caps per-game cost for the slow
    /// rollout leaf). None = curriculum sizes.
    map: Option<i32>,
    /// Clamp every game's round cap to at most this (bounds the worst-case
    /// rollout cost on long games). None = curriculum caps.
    maxcap: Option<i64>,
}

fn default_champion() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.join("checkpoints/champion.json"))
        .unwrap_or_else(|| PathBuf::from("rust-trainer/checkpoints/champion.json"))
}

fn parse_args() -> Result<Config, String> {
    let mut cfg = Config {
        champion: default_champion(),
        games: 80,
        sims: 120,
        seed: 1,
        cap: 200,
        rollout: None,
        search: SearchMode::None,
        value: None,
        curriculum: Curriculum::Bench,
        map: None,
        maxcap: None,
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut take = || it.next().ok_or_else(|| format!("{a} needs a value"));
        match a.as_str() {
            "--champion" => cfg.champion = PathBuf::from(take()?),
            "--games" => cfg.games = take()?.parse().map_err(|e| format!("--games: {e}"))?,
            "--sims" => cfg.sims = take()?.parse().map_err(|e| format!("--sims: {e}"))?,
            "--seed" => cfg.seed = take()?.parse().map_err(|e| format!("--seed: {e}"))?,
            "--cap" => cfg.cap = take()?.parse().map_err(|e| format!("--cap: {e}"))?,
            "--rollout" => {
                cfg.rollout = Some(take()?.parse().map_err(|e| format!("--rollout: {e}"))?)
            }
            "--search" => {
                cfg.search = match take()?.as_str() {
                    "none" => SearchMode::None,
                    "rollout" => SearchMode::Rollout,
                    "static" => SearchMode::Static,
                    "value" => SearchMode::Value,
                    o => return Err(format!("--search: none|rollout|static|value, got {o:?}")),
                }
            }
            "--value" => cfg.value = Some(PathBuf::from(take()?)),
            "--map" => cfg.map = Some(take()?.parse().map_err(|e| format!("--map: {e}"))?),
            "--maxcap" => {
                cfg.maxcap = Some(take()?.parse().map_err(|e| format!("--maxcap: {e}"))?)
            }
            "--curriculum" => {
                cfg.curriculum = match take()?.as_str() {
                    "bench" => Curriculum::Bench,
                    "train" => Curriculum::Train,
                    o => return Err(format!("--curriculum: bench|train, got {o:?}")),
                }
            }
            "-h" | "--help" => return Err(usage()),
            o => return Err(format!("unknown arg {o:?}\n{}", usage())),
        }
    }
    Ok(cfg)
}

fn usage() -> String {
    "bench_hard — champion (seat0) vs HARD heuristic (seat1)\n\
     \n\
     --champion <path>   genome JSON (default checkpoints/champion.json)\n\
     --search MODE       none (default) | rollout | static | value\n\
     --games N           number of games (default 80)\n\
     --sims S            MCTS sims/decision (default 120; ignored when --search none)\n\
     --rollout N         rollout horizon in rounds (default search default; rollout only)\n\
     --value <path>      value net JSON (required for --search value)\n\
     --seed N            base curriculum seed (default 1)\n\
     --cap N             base round cap for the train curriculum (default 200)\n\
     --curriculum C      bench (default, mirrors training/benchmark.ts) | train\n\
     --map N             force NxN map for every game (bounds slow-rollout cost)\n"
        .to_string()
}

fn main() {
    let cfg = match parse_args() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };

    let genome = match Genome::from_file(&cfg.champion.to_string_lossy()) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("bench_hard: failed to load champion {}: {}", cfg.champion.display(), e);
            std::process::exit(2);
        }
    };

    let default_horizon = match SearchConfig::default().leaf_eval {
        LeafEval::Rollout { horizon } => horizon,
        LeafEval::Static | LeafEval::Value => 10,
    };
    let value_net: Option<ValueNet> = if cfg.search == SearchMode::Value {
        let path = match &cfg.value {
            Some(p) => p.clone(),
            None => {
                eprintln!("bench_hard: --search value requires --value <path>");
                std::process::exit(2);
            }
        };
        match ValueNet::from_file(&path.to_string_lossy()) {
            Ok(vn) => Some(vn),
            Err(e) => {
                eprintln!("bench_hard: failed to load value net {}: {}", path.display(), e);
                std::process::exit(2);
            }
        }
    } else {
        None
    };

    let mode = match cfg.search {
        SearchMode::None => Seat0Mode::NoSearch,
        SearchMode::Rollout => Seat0Mode::Search(SearchConfig {
            n_sims: cfg.sims,
            leaf_eval: LeafEval::Rollout {
                horizon: cfg.rollout.unwrap_or(default_horizon),
            },
            ..Default::default()
        }),
        SearchMode::Static => Seat0Mode::Search(SearchConfig {
            n_sims: cfg.sims,
            leaf_eval: LeafEval::Static,
            ..Default::default()
        }),
        SearchMode::Value => Seat0Mode::Search(SearchConfig {
            n_sims: cfg.sims,
            leaf_eval: LeafEval::Value,
            ..Default::default()
        }),
    };

    let search_desc = match cfg.search {
        SearchMode::None => "none (policy argmax)".to_string(),
        SearchMode::Rollout => format!(
            "rollout-MCTS (sims={}, horizon={})",
            cfg.sims,
            cfg.rollout.unwrap_or(default_horizon)
        ),
        SearchMode::Static => format!("static-MCTS (sims={})", cfg.sims),
        SearchMode::Value => format!("value-MCTS (sims={})", cfg.sims),
    };

    let mut specs = match cfg.curriculum {
        Curriculum::Bench => bench_curriculum(cfg.seed as u32, cfg.games),
        Curriculum::Train => train_curriculum(cfg.seed, cfg.cap, cfg.games),
    };
    if let Some(sz) = cfg.map {
        for s in specs.iter_mut() {
            s.width = sz;
            s.height = sz;
        }
    }
    if let Some(mc) = cfg.maxcap {
        for s in specs.iter_mut() {
            if s.cap > mc {
                s.cap = mc;
            }
        }
    }

    println!(
        "bench_hard: champion {} (arch {:?}, {} params)\n\
         bench_hard: seat0 = {}, seat1 = HARD heuristic\n\
         bench_hard: {} games, curriculum {}, seed {}\n",
        cfg.champion.display(),
        genome.arch,
        genome.params.len(),
        search_desc,
        cfg.games,
        match cfg.curriculum {
            Curriculum::Bench => "bench (benchmark.ts)",
            Curriculum::Train => "train",
        },
        cfg.seed,
    );

    let mut wins = 0usize; // seat 0 (champion) wins
    let mut losses = 0usize; // seat 1 (hard) wins
    let mut ties = 0usize;
    let mut timeouts = 0usize;
    let mut tile_frac_sum = 0.0f64;
    let mut rounds_sum = 0i64;

    let t0 = Instant::now();
    for (i, spec) in specs.iter().enumerate() {
        let out = play_game(spec, &genome, mode, value_net.as_ref());
        tile_frac_sum += out.seat0_tile_frac;
        rounds_sum += out.rounds;
        if out.timeout {
            timeouts += 1;
        }
        match out.winner_num {
            Some(1) => wins += 1,
            Some(2) => losses += 1,
            Some(_) => {}
            None if out.timeout => {}
            None => ties += 1,
        }
        println!(
            "  game {:>3}/{:<3}  {}x{} cap{} seed{:<5}  winner={:<4} seat0_frac={:.3} rounds={}",
            i + 1,
            cfg.games,
            spec.width,
            spec.height,
            spec.cap,
            spec.seed,
            out.winner_num
                .map(|w| format!("P{w}"))
                .unwrap_or_else(|| if out.timeout { "TO".into() } else { "tie".into() }),
            out.seat0_tile_frac,
            out.rounds,
        );
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let n = cfg.games as f64;
    let gps = if elapsed > 0.0 { n / elapsed } else { 0.0 };
    println!(
        "\nbench_hard: champion ({}) vs HARD heuristic over {} games\n\
         bench_hard:   win-rate    {:>3}  ({:.1}%)\n\
         bench_hard:   loss-rate   {:>3}  ({:.1}%)\n\
         bench_hard:   tie-rate    {:>3}  ({:.1}%)\n\
         bench_hard:   timeout     {:>3}  ({:.1}%)\n\
         bench_hard:   avg seat0 tile-frac {:.3}\n\
         bench_hard:   avg game length     {:.1} rounds\n\
         bench_hard:   throughput  {:.3} games/s ({:.1}s total)",
        search_desc,
        cfg.games,
        wins,
        100.0 * wins as f64 / n,
        losses,
        100.0 * losses as f64 / n,
        ties,
        100.0 * ties as f64 / n,
        timeouts,
        100.0 * timeouts as f64 / n,
        tile_frac_sum / n,
        rounds_sum as f64 / n,
        gps,
        elapsed,
    );
}
