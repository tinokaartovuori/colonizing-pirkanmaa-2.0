//! `bench` — Stage-A MCTS head-to-head A/B harness.
//!
//! Plays a champion genome against ITSELF over a fixed curriculum:
//!   - seat 0 = the champion wrapped in test-time MCTS (`SearchConfig`),
//!   - seat 1 = the plain champion (no search, `policy` argmax).
//!
//! This isolates the question "does search help?" purely in Rust — the hard
//! heuristic is not in Rust, so this self-play A/B is the Stage-A signal. We
//! reuse the same curriculum dimensions (map sizes / round caps) as `train.rs`,
//! restricted to 2-seat games so the MCTS seat always faces exactly the plain
//! champion.
//!
//! Usage:
//!   cargo run -p cp-train --bin bench -- \
//!     --champion rust-trainer/checkpoints/champion.json \
//!     --games 30 --sims 200 --seed 1
//!
//! Reports seat-0 (MCTS) win-rate, loss-rate, timeout-rate, and avg tile-frac,
//! plus throughput (games/s).

use std::path::PathBuf;
use std::time::Instant;

use cp_ai::controller::NeuralAiController;
use cp_ai::mlp::Genome;
use cp_ai::policy::XorShift32;
use cp_ai::{LeafEval, SearchConfig, ValueNet, TRAINING_CONFIG};
use cp_sim::{EndTurnOutcome, Game, PlayerId};

// ---------------------------------------------------------------------------
// Deterministic per-game curriculum (a trimmed copy of train.rs's logic, fixed
// to 2 seats). SplitMix64 keeps seeds reproducible from one master `--seed`.
// ---------------------------------------------------------------------------

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

struct GameSpec {
    seed: u32,
    width: i32,
    height: i32,
    cap: i64,
}

/// Per-game spec for game `i`, derived from the master `seed`. Biases small/medium
/// maps like train.rs, with ~15% longer games. Always 2 seats.
fn game_spec(master: u64, base_cap: i64, i: usize) -> GameSpec {
    let mut r = SplitMix64::new(master ^ ((i as u64).wrapping_mul(0x165667B19E3779F9)));
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
}

// ---------------------------------------------------------------------------
// One head-to-head game: seat 0 = MCTS champion, seat 1 = plain champion.
// Mirrors `run_game`'s loop (HQ per seat, then plan_turn + end_turn per turn),
// but the two seats use different controllers.
// ---------------------------------------------------------------------------

struct GameOut {
    /// 1-based winner number, or None on tie/timeout.
    winner_num: Option<i64>,
    timeout: bool,
    /// Seat-0 final tile fraction.
    seat0_tile_frac: f64,
}

fn play_game(spec: &GameSpec, genome: &Genome, sc: &SearchConfig, value_net: Option<&ValueNet>) -> GameOut {
    let mut g = Game::new(spec.width, spec.height, &["P1", "P2"]);
    g.generate_map(spec.width, spec.height, spec.seed);

    let mcts = match value_net {
        Some(vn) => NeuralAiController::with_search_value(genome, TRAINING_CONFIG, *sc, vn),
        None => NeuralAiController::with_search(genome, TRAINING_CONFIG, *sc),
    };
    let plain = NeuralAiController::new(genome, TRAINING_CONFIG);
    let mut rng = XorShift32::new(spec.seed);

    // HQ placement per seat (both heuristics are identical / search-free).
    for _ in 0..2 {
        let cur = g.current_player();
        if cur.0 == 0 {
            mcts.place_headquarters(&mut g, cur);
        } else {
            plain.place_headquarters(&mut g, cur);
        }
        g.change_turn();
    }

    let mut winner: Option<PlayerId> = None;
    let mut tie = false;

    while g.live_players().len() > 1 && g.get_rounds_played() < spec.cap {
        let cur = g.current_player();
        if cur.0 == 0 {
            mcts.plan_turn(&mut g, cur, &mut rng, None);
        } else {
            plain.plan_turn(&mut g, cur, &mut rng, None);
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
    }
}

// ---------------------------------------------------------------------------
// CLI + main.
// ---------------------------------------------------------------------------

struct Config {
    champion: PathBuf,
    games: usize,
    sims: usize,
    seed: u64,
    cap: i64,
    /// Optional leaf-rollout horizon override (rounds). None = SearchConfig default.
    /// Ignored when `leaf == Static`.
    rollout: Option<i64>,
    /// Leaf-evaluation mode: rollout (default), static, or learned value net.
    leaf: LeafMode,
    /// Optional fixed map size override `WxH` (e.g. 12). None = curriculum sizes.
    map: Option<i32>,
    /// Path to a trained value net (`value.json`) — required when `leaf == Value`.
    value: Option<PathBuf>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LeafMode {
    Rollout,
    Static,
    Value,
}

fn default_champion() -> PathBuf {
    // CARGO_MANIFEST_DIR = rust-trainer/crates/cp-train; default champion lives at
    // rust-trainer/checkpoints/champion.json (two levels up + checkpoints).
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
        games: 30,
        sims: 200,
        seed: 1,
        cap: 200,
        rollout: None,
        leaf: LeafMode::Rollout,
        map: None,
        value: None,
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
            "--leaf" => {
                cfg.leaf = match take()?.as_str() {
                    "static" => LeafMode::Static,
                    "rollout" => LeafMode::Rollout,
                    "value" => LeafMode::Value,
                    other => {
                        return Err(format!("--leaf: expected static|rollout|value, got {other:?}"))
                    }
                }
            }
            "--value" => cfg.value = Some(PathBuf::from(take()?)),
            "--map" => cfg.map = Some(take()?.parse().map_err(|e| format!("--map: {e}"))?),
            "-h" | "--help" => return Err(usage()),
            other => return Err(format!("unknown arg {other:?}\n{}", usage())),
        }
    }
    Ok(cfg)
}

fn usage() -> String {
    "bench — Stage-A MCTS A/B (seat0=MCTS champion vs seat1=plain champion)\n\
     \n\
     --champion <path>  genome JSON (default rust-trainer/checkpoints/champion.json)\n\
     --games N          number of head-to-head games (default 30)\n\
     --sims S           MCTS simulations per decision (default 200)\n\
     --seed N           master curriculum seed (default 1)\n\
     --cap N            base round cap (default 200)\n\
     --leaf MODE        leaf eval: rollout (default) | static (no turns, fast) | value (learned net)\n\
     --value <path>     trained value net JSON (required when --leaf value)\n\
     --rollout N        leaf-rollout horizon in rounds (default 10; ignored if --leaf static/value)\n\
     --map N            force a fixed NxN map for every game (default: curriculum)\n"
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
            eprintln!("bench: failed to load champion {}: {}", cfg.champion.display(), e);
            std::process::exit(2);
        }
    };

    let default_horizon = match SearchConfig::default().leaf_eval {
        LeafEval::Rollout { horizon } => horizon,
        LeafEval::Static | LeafEval::Value => 10,
    };
    let leaf_eval = match cfg.leaf {
        LeafMode::Static => LeafEval::Static,
        LeafMode::Value => LeafEval::Value,
        LeafMode::Rollout => LeafEval::Rollout {
            horizon: cfg.rollout.unwrap_or(default_horizon),
        },
    };
    let sc = SearchConfig {
        n_sims: cfg.sims,
        leaf_eval,
        ..Default::default()
    };

    // Load the value net if --leaf value (required).
    let value_net: Option<ValueNet> = if cfg.leaf == LeafMode::Value {
        let path = match &cfg.value {
            Some(p) => p.clone(),
            None => {
                eprintln!("bench: --leaf value requires --value <path>");
                std::process::exit(2);
            }
        };
        match ValueNet::from_file(&path.to_string_lossy()) {
            Ok(vn) => Some(vn),
            Err(e) => {
                eprintln!("bench: failed to load value net {}: {}", path.display(), e);
                std::process::exit(2);
            }
        }
    } else {
        None
    };

    let leaf_desc = match leaf_eval {
        LeafEval::Static => "static".to_string(),
        LeafEval::Value => format!(
            "value({})",
            cfg.value.as_ref().map(|p| p.display().to_string()).unwrap_or_default()
        ),
        LeafEval::Rollout { horizon } => format!("rollout(h={horizon})"),
    };
    println!(
        "bench: champion {} (arch {:?}, {} params)\n\
         bench: {} games, {} sims/decision, leaf {}, seed {}, base cap {}\n",
        cfg.champion.display(),
        genome.arch,
        genome.params.len(),
        cfg.games,
        cfg.sims,
        leaf_desc,
        cfg.seed,
        cfg.cap
    );

    let mut wins = 0usize; // seat 0 (MCTS) wins
    let mut losses = 0usize; // seat 1 wins
    let mut ties = 0usize;
    let mut timeouts = 0usize;
    let mut tile_frac_sum = 0.0f64;

    let t0 = Instant::now();
    for i in 0..cfg.games {
        let mut spec = game_spec(cfg.seed, cfg.cap, i);
        if let Some(sz) = cfg.map {
            spec.width = sz;
            spec.height = sz;
        }
        let out = play_game(&spec, &genome, &sc, value_net.as_ref());
        tile_frac_sum += out.seat0_tile_frac;
        if out.timeout {
            timeouts += 1;
        }
        match out.winner_num {
            Some(1) => wins += 1,
            Some(2) => losses += 1,
            Some(_) => {} // 2-seat only; unreachable
            None if out.timeout => {}
            None => ties += 1,
        }
        println!(
            "  game {:>3}/{:<3}  {}x{} cap{} seed{:<10}  winner={:<4} seat0_frac={:.3}",
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
        );
    }
    let elapsed = t0.elapsed().as_secs_f64();

    let n = cfg.games as f64;
    let gps = if elapsed > 0.0 { n / elapsed } else { 0.0 };
    println!(
        "\nbench: seat0 (MCTS) vs seat1 (no-search) over {} games\n\
         bench:   wins      {:>3}  ({:.1}%)\n\
         bench:   losses    {:>3}  ({:.1}%)\n\
         bench:   ties      {:>3}  ({:.1}%)\n\
         bench:   timeouts  {:>3}  ({:.1}%)\n\
         bench:   avg seat0 tile-frac  {:.3}\n\
         bench:   throughput  {:.3} games/s ({:.1}s total, {} sims/decision)",
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
        gps,
        elapsed,
        cfg.sims,
    );
}
