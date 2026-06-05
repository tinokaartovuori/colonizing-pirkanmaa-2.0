//! `value_train` — Stage-B LEARNED VALUE NET data generator + regression trainer.
//!
//! Two phases, one binary:
//!
//! 1. **Data generation** (parallel via rayon). Three game-types, mixable via CLI
//!    so the value net can be trained on a BETTER distribution than the original
//!    no-search self-play (which suffered distribution mismatch vs MCTS / vs the
//!    hard opponent):
//!
//!    - **`selfplay-nosearch`** (the original): every seat plays the champion via
//!      the no-search policy argmax. Fast, but ON-distribution only for argmax.
//!    - **`selfplay-value`**: every seat plays the champion wrapped in value-MCTS
//!      (`--gen-sims`, bootstrapped by `--bootstrap-value`). ON-POLICY for the
//!      search agent — the states MCTS actually visits at play time.
//!    - **`vs-hard`**: seat 0 = the champion (no-search OR value-MCTS), seat 1 =
//!      the faithful Rust `HardAi`. States arising AGAINST the strong opponent.
//!      Only the CHAMPION seat's turns are recorded (labelled from its own
//!      outcome) — we are learning the value of the *search agent's* states, and
//!      HardAi's internal states are off-policy for the net we deploy.
//!
//!    The split is controlled by `--vs-hard-frac F` (fraction of games that are
//!    `vs-hard`; the rest are self-play). Self-play uses value-MCTS iff
//!    `--selfplay-search value`; the champion seat in `vs-hard` uses value-MCTS
//!    iff `--vs-hard-search value`.
//!
//!    At the START of each recorded player-turn we snapshot the 36-dim global
//!    feature vector for the to-move player. After the game ends we label every
//!    snapshot for seat `p` with `z(p) ∈ {+1 win, -1 loss/bankrupt/eliminated,
//!    0 tie/timeout}` — that seat's FINAL outcome from its own perspective. The
//!    buffer is written to disk as JSONL (one `{"x":[..36..],"z":..}` per line).
//!
//! 2. **Gradient regression** (hand-coded backprop, no autodiff). Train the value
//!    net (`[36,32,16,1]`, tanh hidden + tanh output) to predict `z` by MSE via
//!    mini-batch Adam. A held-out validation split reports train/val MSE per
//!    epoch. The trained net is saved to `--out` (default `value.json`).
//!
//! This NEVER touches the policy genome / candidates / features / policy numerics
//! or the parity path — the value net is a separate artifact. The 36-dim feature
//! extraction reuses `cp_ai::features::global_features` verbatim. `HardAi` is the
//! held-out benchmark opponent, also off the parity path.
//!
//! Usage (the v2 "better distribution" run):
//!   cargo run --release -p cp-train --bin value_train -- \
//!     --champion checkpoints/champion.json \
//!     --bootstrap-value checkpoints/value.json \
//!     --games 800 --vs-hard-frac 0.5 \
//!     --selfplay-search value --vs-hard-search value --gen-sims 64 \
//!     --cap 120 --epochs 60 --out checkpoints/value-v2.json
//!
//! Re-using an existing buffer (skip generation):
//!   cargo run --release -p cp-train --bin value_train -- \
//!     --buffer /tmp/value-buffer.jsonl --epochs 60 --out value.json

use std::io::Write as _;
use std::path::PathBuf;
use std::time::Instant;

use rayon::prelude::*;

use cp_ai::features::global_features;
use cp_ai::mlp::Genome;
use cp_ai::policy::XorShift32;
use cp_ai::{
    HardAi, LeafEval, NeuralAiController, SearchConfig, ValueExample, ValueNet, ValueTrainer,
    TRAINING_CONFIG,
};
use cp_sim::{EndTurnOutcome, Game, PlayerId};

/// Whether a generation game's champion seats play via value-MCTS or plain
/// no-search argmax. `Value` requires a bootstrap value net.
#[derive(Clone, Copy, PartialEq, Eq)]
enum GenSearch {
    NoSearch,
    Value,
}

// ---------------------------------------------------------------------------
// Deterministic RNG (SplitMix64) — same construction as bench.rs / train.rs.
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
    seats: usize,
}

/// Per-game spec for game `i`. Mirrors bench.rs's curriculum (small/medium maps),
/// but allows 2-4 seats so the value net sees multiplayer board states too.
fn game_spec(master: u64, base_cap: i64, i: usize, force_map: Option<i32>, force_seats: Option<usize>) -> GameSpec {
    let mut r = SplitMix64::new(master ^ ((i as u64).wrapping_mul(0x165667B19E3779F9)));
    let size_roll = r.next_f64();
    let (mut width, mut height) = if size_roll < 0.5 {
        (12, 12)
    } else if size_roll < 0.8 {
        (16, 14)
    } else if size_roll < 0.93 {
        (18, 14)
    } else {
        (20, 15)
    };
    if let Some(sz) = force_map {
        width = sz;
        height = sz;
    }
    let cap = if r.next_f64() < 0.15 {
        base_cap.max(300)
    } else {
        base_cap
    };
    // 2-4 seats (more multiplayer variety than bench's fixed 2).
    let seats = force_seats.unwrap_or_else(|| {
        let roll = r.next_f64();
        if roll < 0.5 {
            2
        } else if roll < 0.8 {
            3
        } else {
            4
        }
    });
    let seed = (r.next_u64() as u32) | 1;
    GameSpec {
        seed,
        width,
        height,
        cap,
        seats,
    }
}

// ---------------------------------------------------------------------------
// Phase 1: one self-play game → labelled (features, z) examples.
//
// Mirrors `run_game`'s loop (HQ per seat, then plan_turn + end_turn per turn)
// with EVERY seat on the no-search champion. At each turn START we snapshot the
// to-move player's 36-dim global features; after the game ends we label every
// snapshot with that seat's final z from its own perspective.
// ---------------------------------------------------------------------------

/// A value-MCTS controller config for generation games. Modest sims for speed;
/// other knobs match `SearchConfig::default()` except the value-net leaf.
fn gen_search_config(sims: usize) -> SearchConfig {
    SearchConfig {
        n_sims: sims,
        leaf_eval: LeafEval::Value,
        ..Default::default()
    }
}

/// SELF-PLAY game: every seat = the champion (no-search OR value-MCTS). Records
/// the to-move seat's features at every turn start, labels each by that seat's
/// final outcome. `bootstrap` is the value net used as the MCTS leaf evaluator
/// when `search == Value` (ignored for no-search).
fn play_and_label(
    spec: &GameSpec,
    genome: &Genome,
    search: GenSearch,
    gen_sims: usize,
    bootstrap: Option<&ValueNet>,
) -> Vec<ValueExample> {
    let n = spec.seats;
    let names: Vec<String> = (0..n).map(|i| format!("P{}", i + 1)).collect();
    let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    let mut g = Game::new(spec.width, spec.height, &name_refs);
    g.generate_map(spec.width, spec.height, spec.seed);

    let sc = gen_search_config(gen_sims);
    let make_ctrl = || match (search, bootstrap) {
        (GenSearch::Value, Some(vn)) => {
            NeuralAiController::with_search_value(genome, TRAINING_CONFIG, sc, vn)
        }
        _ => NeuralAiController::new(genome, TRAINING_CONFIG),
    };
    let ctrls: Vec<NeuralAiController> = (0..n).map(|_| make_ctrl()).collect();
    let mut rng = XorShift32::new(spec.seed);

    // HQ placement per seat.
    for _ in 0..n {
        let cur = g.current_player();
        ctrls[cur.0].place_headquarters(&mut g, cur);
        g.change_turn();
    }

    // (seat, features) snapshots, labelled with z at the end.
    let mut samples: Vec<(usize, Vec<f64>)> = Vec::new();

    let mut winner: Option<PlayerId> = None;
    let mut tie = false;

    while g.live_players().len() > 1 && g.get_rounds_played() < spec.cap {
        let cur = g.current_player();
        // Snapshot the to-move player's global features at turn start.
        let round = g.get_rounds_played();
        let gvec = global_features(&mut g, cur, round);
        samples.push((cur.0, gvec));

        ctrls[cur.0].plan_turn(&mut g, cur, &mut rng, None);
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

    label_samples(&g, samples, winner, tie)
}

/// VS-HARD game: seat 0 = champion (no-search OR value-MCTS), seat 1 = HardAi.
/// Records ONLY the champion seat's (seat 0) turns — the states the search agent
/// faces against the strong opponent — labelled by seat 0's final outcome.
fn play_and_label_vs_hard(
    spec: &GameSpec,
    genome: &Genome,
    search: GenSearch,
    gen_sims: usize,
    bootstrap: Option<&ValueNet>,
) -> Vec<ValueExample> {
    // vs-hard is strictly 2-seat (champion vs one hard opponent).
    let mut g = Game::new(spec.width, spec.height, &["P1", "P2"]);
    g.generate_map(spec.width, spec.height, spec.seed);

    let sc = gen_search_config(gen_sims);
    let champ = match (search, bootstrap) {
        (GenSearch::Value, Some(vn)) => {
            NeuralAiController::with_search_value(genome, TRAINING_CONFIG, sc, vn)
        }
        _ => NeuralAiController::new(genome, TRAINING_CONFIG),
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

    let mut samples: Vec<(usize, Vec<f64>)> = Vec::new();
    let mut winner: Option<PlayerId> = None;
    let mut tie = false;

    while g.live_players().len() > 1 && g.get_rounds_played() < spec.cap {
        let cur = g.current_player();
        let round = g.get_rounds_played();
        if cur.0 == 0 {
            // Record ONLY the champion seat's states.
            let gvec = global_features(&mut g, cur, round);
            samples.push((cur.0, gvec));
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

    label_samples(&g, samples, winner, tie)
}

/// Resolve final per-seat outcome z and attach it to each snapshot. A seat WINS
/// if it's the recorded winner or the sole survivor; ties/timeouts → 0;
/// everything else (dead / eliminated / bankrupt / a present-but-losing seat at
/// timeout) → -1.
fn label_samples(
    g: &Game,
    samples: Vec<(usize, Vec<f64>)>,
    winner: Option<PlayerId>,
    tie: bool,
) -> Vec<ValueExample> {
    let live: std::collections::HashSet<PlayerId> = g.live_players().iter().copied().collect();
    let winner_seat: Option<usize> = if let Some(w) = winner {
        Some(w.0)
    } else if !tie && g.live_players().len() == 1 {
        Some(g.live_players()[0].0)
    } else {
        None
    };

    let z_for = |seat: usize| -> f64 {
        if let Some(ws) = winner_seat {
            return if seat == ws { 1.0 } else { -1.0 };
        }
        if tie {
            // Tie verdict: survivors share a draw (0); dead seats lost (-1).
            return if live.contains(&PlayerId(seat)) { 0.0 } else { -1.0 };
        }
        // Timeout (round cap): no winner. Survivors → 0 (no decision),
        // eliminated seats → -1.
        if live.contains(&PlayerId(seat)) {
            0.0
        } else {
            -1.0
        }
    };

    samples
        .into_iter()
        .map(|(seat, x)| ValueExample { x, z: z_for(seat) })
        .collect()
}

// ---------------------------------------------------------------------------
// JSONL buffer I/O.
// ---------------------------------------------------------------------------

fn write_buffer(path: &PathBuf, data: &[ValueExample]) -> std::io::Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    for ex in data {
        writeln!(f, "{}", serde_json::to_string(ex).expect("example serialises"))?;
    }
    f.flush()
}

fn read_buffer(path: &PathBuf) -> std::io::Result<Vec<ValueExample>> {
    let s = std::fs::read_to_string(path)?;
    let mut out = Vec::new();
    for line in s.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let ex: ValueExample = serde_json::from_str(line)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        out.push(ex);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// CLI + main.
// ---------------------------------------------------------------------------

struct Config {
    champion: PathBuf,
    games: usize,
    seed: u64,
    cap: i64,
    map: Option<i32>,
    seats: Option<usize>,
    epochs: usize,
    batch: usize,
    lr: f64,
    val_frac: f64,
    out: PathBuf,
    /// Write the generated buffer here (default `<out-dir>/value-buffer.jsonl`).
    buffer_out: Option<PathBuf>,
    /// Load an existing buffer and SKIP generation.
    buffer_in: Option<PathBuf>,
    // --- v2 better-distribution knobs ---------------------------------------
    /// Fraction of generated games that are champion-vs-HardAi (rest self-play).
    vs_hard_frac: f64,
    /// Search mode for self-play seats.
    selfplay_search: GenSearch,
    /// Search mode for the champion seat in vs-hard games.
    vs_hard_search: GenSearch,
    /// MCTS sims/decision during generation (value-MCTS game types).
    gen_sims: usize,
    /// Bootstrap value net for value-MCTS during generation. Required if any
    /// game type uses value-MCTS.
    bootstrap_value: Option<PathBuf>,
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
        games: 800,
        seed: 1,
        cap: 120,
        map: None,
        seats: None,
        epochs: 60,
        batch: 256,
        lr: 1e-3,
        val_frac: 0.1,
        out: PathBuf::from("rust-trainer/checkpoints/value.json"),
        buffer_out: None,
        buffer_in: None,
        vs_hard_frac: 0.0,
        selfplay_search: GenSearch::NoSearch,
        vs_hard_search: GenSearch::NoSearch,
        gen_sims: 64,
        bootstrap_value: None,
    };
    let parse_search = |s: &str| -> Result<GenSearch, String> {
        match s {
            "none" => Ok(GenSearch::NoSearch),
            "value" => Ok(GenSearch::Value),
            o => Err(format!("expected none|value, got {o:?}")),
        }
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut take = || it.next().ok_or_else(|| format!("{a} needs a value"));
        match a.as_str() {
            "--champion" => cfg.champion = PathBuf::from(take()?),
            "--games" => cfg.games = take()?.parse().map_err(|e| format!("--games: {e}"))?,
            "--seed" => cfg.seed = take()?.parse().map_err(|e| format!("--seed: {e}"))?,
            "--cap" => cfg.cap = take()?.parse().map_err(|e| format!("--cap: {e}"))?,
            "--map" => cfg.map = Some(take()?.parse().map_err(|e| format!("--map: {e}"))?),
            "--seats" => cfg.seats = Some(take()?.parse().map_err(|e| format!("--seats: {e}"))?),
            "--epochs" => cfg.epochs = take()?.parse().map_err(|e| format!("--epochs: {e}"))?,
            "--batch" => cfg.batch = take()?.parse().map_err(|e| format!("--batch: {e}"))?,
            "--lr" => cfg.lr = take()?.parse().map_err(|e| format!("--lr: {e}"))?,
            "--val-frac" => cfg.val_frac = take()?.parse().map_err(|e| format!("--val-frac: {e}"))?,
            "--out" => cfg.out = PathBuf::from(take()?),
            "--buffer-out" => cfg.buffer_out = Some(PathBuf::from(take()?)),
            "--buffer" => cfg.buffer_in = Some(PathBuf::from(take()?)),
            "--vs-hard-frac" => {
                cfg.vs_hard_frac = take()?.parse().map_err(|e| format!("--vs-hard-frac: {e}"))?
            }
            "--selfplay-search" => cfg.selfplay_search = parse_search(take()?)?,
            "--vs-hard-search" => cfg.vs_hard_search = parse_search(take()?)?,
            "--gen-sims" => {
                cfg.gen_sims = take()?.parse().map_err(|e| format!("--gen-sims: {e}"))?
            }
            "--bootstrap-value" => cfg.bootstrap_value = Some(PathBuf::from(take()?)),
            "-h" | "--help" => return Err(usage()),
            other => return Err(format!("unknown arg {other:?}\n{}", usage())),
        }
    }
    Ok(cfg)
}

fn usage() -> String {
    "value_train — Stage-B value-net data gen + regression trainer\n\
     \n\
     --champion <path>  policy genome JSON for self-play (default checkpoints/champion.json)\n\
     --games N          self-play games to generate (default 800)\n\
     --seed N           master curriculum seed (default 1)\n\
     --cap N            base round cap (default 120)\n\
     --map N            force NxN maps (default: curriculum sizes)\n\
     --seats N          force seat count 2-4 (default: curriculum 2-4)\n\
     --epochs N         training epochs (default 60)\n\
     --batch N          mini-batch size (default 256)\n\
     --lr F             Adam learning rate (default 1e-3)\n\
     --val-frac F       validation split fraction (default 0.1)\n\
     --out <path>       trained value.json (default checkpoints/value.json)\n\
     --buffer-out <p>   where to write the generated JSONL buffer\n\
     --buffer <path>    load an existing buffer and SKIP generation\n\
     \n\
     v2 better-distribution knobs:\n\
     --vs-hard-frac F     fraction of games that are champion-vs-HardAi (default 0.0)\n\
     --selfplay-search M  none (default) | value  — self-play seats' search mode\n\
     --vs-hard-search M   none (default) | value  — champion seat's search mode vs hard\n\
     --gen-sims S         MCTS sims/decision during generation (default 64)\n\
     --bootstrap-value P  value net used as the MCTS leaf during generation\n"
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

    // --- Phase 1: data (generate or load) ----------------------------------
    let mut data: Vec<ValueExample> = if let Some(buf) = &cfg.buffer_in {
        println!("value_train: loading buffer {} (skipping generation)", buf.display());
        match read_buffer(buf) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("value_train: failed to read buffer: {e}");
                std::process::exit(2);
            }
        }
    } else {
        let genome = match Genome::from_file(&cfg.champion.to_string_lossy()) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("value_train: failed to load champion {}: {}", cfg.champion.display(), e);
                std::process::exit(2);
            }
        };

        // Load the bootstrap value net if any game type uses value-MCTS.
        let need_bootstrap = cfg.selfplay_search == GenSearch::Value
            || cfg.vs_hard_search == GenSearch::Value;
        let bootstrap: Option<ValueNet> = if need_bootstrap {
            let path = match &cfg.bootstrap_value {
                Some(p) => p.clone(),
                None => {
                    eprintln!("value_train: value-MCTS generation requires --bootstrap-value <path>");
                    std::process::exit(2);
                }
            };
            match ValueNet::from_file(&path.to_string_lossy()) {
                Ok(vn) => {
                    println!("value_train: bootstrap value net {} (arch {:?})", path.display(), vn.arch);
                    Some(vn)
                }
                Err(e) => {
                    eprintln!("value_train: failed to load bootstrap value net {}: {}", path.display(), e);
                    std::process::exit(2);
                }
            }
        } else {
            None
        };

        // Partition games: the first `n_vs_hard` are champion-vs-HardAi, the rest
        // are self-play. (Deterministic, contiguous split — keeps the seed→spec
        // mapping stable and lets us report exact composition.)
        let n_vs_hard = ((cfg.games as f64) * cfg.vs_hard_frac).round() as usize;
        let n_vs_hard = n_vs_hard.min(cfg.games);
        let n_selfplay = cfg.games - n_vs_hard;
        let sp_desc = match cfg.selfplay_search {
            GenSearch::NoSearch => "no-search",
            GenSearch::Value => "value-MCTS",
        };
        let vh_desc = match cfg.vs_hard_search {
            GenSearch::NoSearch => "no-search",
            GenSearch::Value => "value-MCTS",
        };
        println!(
            "value_train: generating {} games (champion {}, arch {:?})\n\
             value_train:   self-play {} ({}, {}-seat), vs-hard {} ({} vs HardAi, gen-sims {})",
            cfg.games,
            cfg.champion.display(),
            genome.arch,
            n_selfplay,
            sp_desc,
            cfg.seats.map(|s| s.to_string()).unwrap_or_else(|| "2-4".into()),
            n_vs_hard,
            vh_desc,
            cfg.gen_sims,
        );
        let t0 = Instant::now();
        let bootstrap_ref = bootstrap.as_ref();
        let data: Vec<ValueExample> = (0..cfg.games)
            .into_par_iter()
            .flat_map_iter(|i| {
                if i < n_vs_hard {
                    // vs-hard games: force 2 seats. Use the bench-comparable map
                    // distribution by reusing game_spec with seats forced to 2.
                    let spec = game_spec(cfg.seed ^ 0x4841_5244_u64, cfg.cap, i, cfg.map, Some(2));
                    play_and_label_vs_hard(
                        &spec,
                        &genome,
                        cfg.vs_hard_search,
                        cfg.gen_sims,
                        bootstrap_ref,
                    )
                } else {
                    let spec = game_spec(cfg.seed, cfg.cap, i, cfg.map, cfg.seats);
                    play_and_label(
                        &spec,
                        &genome,
                        cfg.selfplay_search,
                        cfg.gen_sims,
                        bootstrap_ref,
                    )
                }
            })
            .collect();
        let dt = t0.elapsed().as_secs_f64();
        println!(
            "value_train: generated {} examples from {} games in {:.1}s ({:.2} games/s)",
            data.len(),
            cfg.games,
            dt,
            cfg.games as f64 / dt.max(1e-9)
        );
        // Persist the buffer.
        let buf_path = cfg.buffer_out.clone().unwrap_or_else(|| {
            cfg.out
                .parent()
                .map(|p| p.join("value-buffer.jsonl"))
                .unwrap_or_else(|| PathBuf::from("value-buffer.jsonl"))
        });
        if let Err(e) = write_buffer(&buf_path, &data) {
            eprintln!("value_train: WARN failed to write buffer {}: {e}", buf_path.display());
        } else {
            println!("value_train: wrote buffer → {}", buf_path.display());
        }
        data
    };

    if data.is_empty() {
        eprintln!("value_train: no training data; aborting");
        std::process::exit(2);
    }

    // Label balance report.
    let (mut npos, mut nneg, mut nzero) = (0usize, 0usize, 0usize);
    for ex in &data {
        if ex.z > 0.5 {
            npos += 1;
        } else if ex.z < -0.5 {
            nneg += 1;
        } else {
            nzero += 1;
        }
    }
    println!(
        "value_train: label balance  +1: {} ({:.1}%)  -1: {} ({:.1}%)  0: {} ({:.1}%)",
        npos,
        100.0 * npos as f64 / data.len() as f64,
        nneg,
        100.0 * nneg as f64 / data.len() as f64,
        nzero,
        100.0 * nzero as f64 / data.len() as f64,
    );

    // --- Deterministic shuffle + train/val split ----------------------------
    let mut rng = SplitMix64::new(cfg.seed.wrapping_mul(0x9E3779B97F4A7C15) ^ 0xA5A5_A5A5);
    // Fisher-Yates.
    for i in (1..data.len()).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        data.swap(i, j);
    }
    let val_n = ((data.len() as f64 * cfg.val_frac).round() as usize).clamp(1, data.len() - 1);
    let (val, train) = data.split_at(val_n);
    println!(
        "value_train: split  train={}  val={}  (val_frac={:.2})",
        train.len(),
        val.len(),
        cfg.val_frac
    );

    // --- Phase 2: regression -------------------------------------------------
    let net = ValueNet::random(cfg.seed ^ 0xC0FFEE);
    let mut trainer = ValueTrainer::new(net, cfg.lr);
    println!(
        "value_train: training value net arch {:?} ({} params), Adam lr={}, batch={}, epochs={}",
        trainer.net.arch,
        trainer.net.params.len(),
        cfg.lr,
        cfg.batch,
        cfg.epochs
    );

    let mut idx: Vec<usize> = (0..train.len()).collect();
    let mut best_val = f64::INFINITY;
    let mut best_params = trainer.net.params.clone();

    for epoch in 0..cfg.epochs {
        // Shuffle the training index each epoch.
        for i in (1..idx.len()).rev() {
            let j = (rng.next_u64() % (i as u64 + 1)) as usize;
            idx.swap(i, j);
        }
        let mut epoch_loss = 0.0;
        let mut nbatches = 0usize;
        let mut start = 0usize;
        while start < idx.len() {
            let end = (start + cfg.batch).min(idx.len());
            let batch: Vec<ValueExample> =
                idx[start..end].iter().map(|&k| train[k].clone()).collect();
            epoch_loss += trainer.step(&batch);
            nbatches += 1;
            start = end;
        }
        let train_mse = epoch_loss / nbatches.max(1) as f64;
        let val_mse = trainer.mse(val);
        if val_mse < best_val {
            best_val = val_mse;
            best_params = trainer.net.params.clone();
        }
        println!(
            "  epoch {:>3}/{:<3}  train_mse {:.5}  val_mse {:.5}{}",
            epoch + 1,
            cfg.epochs,
            train_mse,
            val_mse,
            if val_mse <= best_val { "  *" } else { "" }
        );
    }

    // Restore best-val params (early-stopping by validation MSE).
    trainer.net.params = best_params;
    let final_train = trainer.mse(train);
    let final_val = trainer.mse(val);
    println!(
        "value_train: best val_mse {:.5}  (final train_mse {:.5}, val_mse {:.5})",
        best_val, final_train, final_val
    );

    if let Err(e) = trainer.net.to_file(&cfg.out.to_string_lossy()) {
        eprintln!("value_train: failed to write {}: {e}", cfg.out.display());
        std::process::exit(2);
    }
    println!("value_train: saved value net → {}", cfg.out.display());
}
