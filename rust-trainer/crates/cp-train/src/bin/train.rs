//! `train` binary — Milestone 6 neuroevolution self-play trainer.
//!
//! Mutation-only evolution strategy with elitism (NO crossover — uniform weight
//! crossover hits the NN permutation problem, so it is deliberately omitted).
//! Each generation:
//!   1. Evaluate every genome over `GAMES` self-play games whose opponents are
//!      drawn ONLY from {current population} ∪ {Hall-of-Fame} (the heuristic AI
//!      is NEVER used in training — that is a held-out M7 benchmark).
//!   2. Sort the population by mean game fitness (desc), keep the top `ELITE`
//!      unchanged, and refill the rest by cloning a random elite and mutating
//!      every param by `+= sigma * gaussian()` (Box–Muller).
//!   3. Anneal sigma linearly from `sigma0` (gen 0) to `sigma1` (final gen).
//!   4. Snapshot the best genome into the Hall of Fame every `HOF_EVERY` gens.
//!
//! Determinism: a single master `--seed` derives every per-game seed, every
//! opponent sample, and every mutation draw via SplitMix64 stream-splitting, so a
//! run is reproducible regardless of how rayon schedules the games.
//!
//! This binary does NOT touch `training/checkpoints/` (the real champion the TS
//! game + golden traces depend on). Outputs go to `--out`
//! (default `rust-trainer/checkpoints/`): `champion.json`, `hof.json`, `log.jsonl`.
//!
//! Resuming: `--resume <DIR>` lets a long run span multiple sessions and keep
//! improving instead of restarting from noise. It loads `<DIR>/champion.json` and
//! `<DIR>/hof.json` (validating arch/param-count), seeds the initial population
//! from the champion (genome 0 unchanged, the rest mutated copies at `sigma0`),
//! pre-loads the HoF as immediate self-play opponents, and APPENDS to `log.jsonl`
//! with the `gen` counter offset by the existing line count so the dashboard
//! x-axis stays continuous. `best_overall` is pre-seeded with the loaded champion
//! at fitness `NEG_INFINITY`, so the on-disk champion can never regress below the
//! one we resumed from even after a noisy generation. The same `--seed` makes a
//! resumed run reproducible. Typical usage: `--resume D --out D`.

use std::io::Write as _;
use std::path::PathBuf;
use std::time::Instant;

use rayon::prelude::*;

use cp_ai::mlp::{param_count, Genome};
use cp_ai::{run_game_telemetry, GameTelemetry, SeatTelemetry, TRAINING_CONFIG};
use cp_train::{fitness_v2, RewardConfig};

// ===========================================================================
// Deterministic RNG — SplitMix64. Cheap, stream-splittable, no `rand` dep.
// ===========================================================================

#[derive(Clone)]
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
    /// Uniform in [0, 1).
    fn next_f64(&mut self) -> f64 {
        // 53-bit mantissa.
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn next_usize(&mut self, bound: usize) -> usize {
        debug_assert!(bound > 0);
        (self.next_u64() % bound as u64) as usize
    }
    /// Standard normal via Box–Muller.
    fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Derive an independent stream seed from a master seed + integer labels. Mixing
/// each label through SplitMix64 keeps the streams decorrelated and order-free.
fn derive(master: u64, a: u64, b: u64, c: u64) -> u64 {
    let mut m = SplitMix64::new(master ^ 0xD1B54A32D192ED03);
    m.state = m.state.wrapping_add(a.wrapping_mul(0x9E3779B97F4A7C15));
    m.next_u64();
    m.state = m.state.wrapping_add(b.wrapping_mul(0xC2B2AE3D27D4EB4F));
    m.next_u64();
    m.state = m.state.wrapping_add(c.wrapping_mul(0x165667B19E3779F9));
    m.next_u64()
}

// ===========================================================================
// Per-game RNG adapter so mutation/sampling/game-seeds are all reproducible.
// Game seeds are u32 (cp-sim API); derive deterministically from the master.
// ===========================================================================

// ===========================================================================
// Config / CLI.
// ===========================================================================

struct Config {
    gens: usize,
    pop: usize,
    elite: usize,
    games: usize,
    seed: u64,
    cap: i64,
    hof_every: usize,
    hof_max: usize,
    sigma0: f64,
    sigma1: f64,
    out: PathBuf,
    /// When `Some`, resume from a previous run's checkpoint dir: seed the initial
    /// population from `<DIR>/champion.json`, load `<DIR>/hof.json`, and append to
    /// `log.jsonl` (instead of random init + truncate).
    resume: Option<PathBuf>,
    /// Reward shaping used by `fitness_v2` (v3 shaping). Loaded from
    /// `--reward <FILE>` (JSON), else the built-in default (== `rewards/v3-default.json`).
    reward: RewardConfig,
    /// PFSP-style opponent prioritization (AlphaStar Nature'19). `0.0` = uniform
    /// sampling (the exact legacy behavior, byte-reproducible). `> 0.0` is the
    /// softmax temperature `beta` over opponent *strength* (previous-gen fitness):
    /// `weight ∝ exp(beta*(strength - max))`, so stronger opponents — the ones
    /// most likely to beat you, the PFSP intent — are sampled more often.
    pfsp: f64,
    /// Novelty-search weight (Lehman & Stanley). `0.0` = off (selection by raw
    /// fitness, byte-reproducible). `> 0.0` adds `novelty * w` to the SELECTION
    /// score (elites/parents), where novelty is the mean distance to the
    /// `novelty_k` nearest behaviors among {population ∪ archive} in a 5-dim
    /// behavior space. The shipped champion is still chosen by raw fitness. This
    /// is the decisive diagnostic: if it unsticks expansion/fitness the cause was
    /// exploration; if not, it confirms the representation ceiling.
    novelty: f64,
    /// k for the novelty kNN distance (default 5).
    novelty_k: usize,
    /// Max behavior-archive size (oldest dropped). Default 200.
    novelty_archive_max: usize,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            gens: 120,
            pop: 48,
            elite: 8,
            games: 24,
            seed: 0xC0FFEE,
            cap: 200,
            hof_every: 5,
            hof_max: 10,
            sigma0: 0.18,
            sigma1: 0.05,
            out: PathBuf::from("rust-trainer/checkpoints"),
            resume: None,
            reward: RewardConfig::default(),
            pfsp: 0.0,
            novelty: 0.0,
            novelty_k: 5,
            novelty_archive_max: 200,
        }
    }
}

fn parse_args() -> Result<Config, String> {
    let mut cfg = Config::default();
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut take = || -> Result<String, String> {
            args.next().ok_or_else(|| format!("missing value for {flag}"))
        };
        match flag.as_str() {
            "--gens" => cfg.gens = take()?.parse().map_err(|e| format!("--gens: {e}"))?,
            "--pop" => cfg.pop = take()?.parse().map_err(|e| format!("--pop: {e}"))?,
            "--elite" => cfg.elite = take()?.parse().map_err(|e| format!("--elite: {e}"))?,
            "--games" => cfg.games = take()?.parse().map_err(|e| format!("--games: {e}"))?,
            "--seed" => cfg.seed = take()?.parse().map_err(|e| format!("--seed: {e}"))?,
            "--cap" => cfg.cap = take()?.parse().map_err(|e| format!("--cap: {e}"))?,
            "--hof-every" => {
                cfg.hof_every = take()?.parse().map_err(|e| format!("--hof-every: {e}"))?
            }
            "--hof-max" => cfg.hof_max = take()?.parse().map_err(|e| format!("--hof-max: {e}"))?,
            "--sigma0" => cfg.sigma0 = take()?.parse().map_err(|e| format!("--sigma0: {e}"))?,
            "--sigma1" => cfg.sigma1 = take()?.parse().map_err(|e| format!("--sigma1: {e}"))?,
            "--out" => cfg.out = PathBuf::from(take()?),
            "--resume" => cfg.resume = Some(PathBuf::from(take()?)),
            "--reward" => {
                let path = PathBuf::from(take()?);
                cfg.reward = RewardConfig::from_file(&path)
                    .map_err(|e| format!("--reward: {e}"))?;
            }
            "--pfsp" => cfg.pfsp = take()?.parse().map_err(|e| format!("--pfsp: {e}"))?,
            "--novelty" => cfg.novelty = take()?.parse().map_err(|e| format!("--novelty: {e}"))?,
            "--novelty-k" => {
                cfg.novelty_k = take()?.parse().map_err(|e| format!("--novelty-k: {e}"))?
            }
            "--novelty-archive-max" => {
                cfg.novelty_archive_max = take()?
                    .parse()
                    .map_err(|e| format!("--novelty-archive-max: {e}"))?
            }
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unknown flag {other:?} (try --help)")),
        }
    }
    if cfg.pop < 2 {
        return Err("--pop must be >= 2 (need an opponent)".into());
    }
    if cfg.elite == 0 {
        return Err("--elite must be >= 1".into());
    }
    // Clamp elite to pop (the default 8 can exceed a tiny smoke-test pop).
    if cfg.elite > cfg.pop {
        eprintln!(
            "cp-train: --elite ({}) > --pop ({}); clamping elite to {}",
            cfg.elite, cfg.pop, cfg.pop
        );
        cfg.elite = cfg.pop;
    }
    if cfg.games == 0 {
        return Err("--games must be >= 1".into());
    }
    if cfg.pfsp < 0.0 {
        return Err("--pfsp must be >= 0".into());
    }
    if cfg.novelty < 0.0 {
        return Err("--novelty must be >= 0".into());
    }
    if cfg.novelty_k == 0 {
        return Err("--novelty-k must be >= 1".into());
    }
    Ok(cfg)
}

fn print_help() {
    eprintln!(
        "cp-train — neuroevolution self-play trainer\n\n\
         USAGE: train [OPTIONS]\n\n\
         OPTIONS:\n\
         \x20 --gens <N>        generations (default 120)\n\
         \x20 --pop <N>         population size (default 48)\n\
         \x20 --elite <N>       elites kept each gen (default 8)\n\
         \x20 --games <N>       games per genome per gen (default 24)\n\
         \x20 --seed <N>        master RNG seed (default 0xC0FFEE)\n\
         \x20 --cap <N>         base round cap (default 200; ~15%% of games at 300+)\n\
         \x20 --hof-every <N>   snapshot champion to HoF every N gens (default 5)\n\
         \x20 --hof-max <N>     max HoF size (default 10)\n\
         \x20 --sigma0 <F>      initial mutation sigma (default 0.18)\n\
         \x20 --sigma1 <F>      final mutation sigma, held after 60%% of gens (default 0.05)\n\
         \x20 --reward <FILE>   reward-shaping JSON for fitness_v2 (default: built-in v3)\n\
         \x20 --pfsp <BETA>     PFSP opponent prioritization: softmax temp over opponent\n\
         \x20                   strength (prev-gen fitness). 0=uniform (default, legacy).\n\
         \x20                   Try 2-4. Stronger opponents sampled more often.\n\
         \x20 --novelty <W>     novelty-search weight added to the SELECTION score\n\
         \x20                   (0=off, default). Champion still shipped by raw fitness.\n\
         \x20                   Diagnostic: unsticks fitness => exploration was the cause.\n\
         \x20 --novelty-k <N>   kNN for novelty distance (default 5)\n\
         \x20 --novelty-archive-max <N>  behavior-archive cap (default 200)\n\
         \x20 --out <DIR>       output dir (default rust-trainer/checkpoints)\n\
         \x20 --resume <DIR>    resume: seed pop from <DIR>/champion.json, load\n\
         \x20                   <DIR>/hof.json, and APPEND to log.jsonl (continuing\n\
         \x20                   the gen counter). Typically used with --resume D --out D.\n\
         \x20                   Without it: random init + truncate log (default)."
    );
}

// ===========================================================================
// Curriculum: per-game map size / player count / round cap. Deterministic.
// ===========================================================================

struct GameSpec {
    seed: u32,
    width: i32,
    height: i32,
    players: usize,
    cap: i64,
}

/// Build the spec for genome `gi`'s game `game_idx` in generation `gen`. Biases
/// small/medium maps, mostly 2 players, with ~15% long (180-round) games.
fn game_spec(cfg: &Config, gen: usize, gi: usize, game_idx: usize) -> GameSpec {
    let mut r = SplitMix64::new(derive(cfg.seed, 0x5EED, gen as u64, (gi as u64) << 16 | game_idx as u64));

    // Map size: bias small/medium. Sizes between 12x12 and 20x15.
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

    // Player count: mostly 2, some 3, few 4.
    let pc_roll = r.next_f64();
    let players = if pc_roll < 0.7 {
        2
    } else if pc_roll < 0.92 {
        3
    } else {
        4
    };

    // Round cap: ~15% extra-long games, else the base cap.
    let cap = if r.next_f64() < 0.15 { cfg.cap.max(300) } else { cfg.cap };

    // Per-game world seed — derived independently of map/player rolls.
    let seed = (derive(cfg.seed, 0x60A1, gen as u64, (gi as u64) << 20 | game_idx as u64) as u32)
        | 1; // avoid 0

    GameSpec {
        seed,
        width,
        height,
        players,
        cap,
    }
}

// ===========================================================================
// Fitness v1 — per game, for the evaluated seat (index 0 of the genome list).
// ===========================================================================

/// Pick the evaluated seat's telemetry from a game where it played seat 0
/// (player num 1).
fn eval_seat(t: &GameTelemetry) -> &SeatTelemetry {
    &t.seats[0]
}

// ===========================================================================
// Per-genome evaluation results (aggregated over its games).
// ===========================================================================

#[derive(Clone, Default)]
struct GenomeEval {
    fitness: f64,
    // Telemetry aggregates for logging (over this genome's games).
    avg_game_len: f64,
    bankrupt_games: f64,
    sum_tile_frac: f64,
    sum_net_income: f64,
    // Behavior-descriptor accumulators (means over games) for novelty search.
    // All normalized to roughly [0,1] in `behavior_descriptor`.
    sum_conquered_frac: f64,
    sum_productive_area: f64,
    sum_military_lead: f64, // ∈ [-1,1] before normalization
    sum_survived_frac: f64,
    games: usize,
}

/// Dimensionality of the novelty behavior descriptor (see `behavior_descriptor`).
const BD_DIM: usize = 5;

/// Map a genome's averaged telemetry to a 5-dim behavior descriptor in ~[0,1]:
/// [expansion, aggression, economy-style, military-lead, longevity]. This is the
/// "what it DID" fingerprint novelty search rewards being different on — chosen to
/// capture exactly the axes the AI currently fails on (it barely expands/fights).
fn behavior_descriptor(e: &GenomeEval) -> [f64; BD_DIM] {
    [
        e.sum_tile_frac.clamp(0.0, 1.0),
        e.sum_conquered_frac.clamp(0.0, 1.0),
        e.sum_productive_area.clamp(0.0, 1.0),
        (0.5 * (e.sum_military_lead + 1.0)).clamp(0.0, 1.0),
        e.sum_survived_frac.clamp(0.0, 1.0),
    ]
}

fn bd_dist(a: &[f64; BD_DIM], b: &[f64; BD_DIM]) -> f64 {
    let mut s = 0.0;
    for i in 0..BD_DIM {
        let d = a[i] - b[i];
        s += d * d;
    }
    s.sqrt()
}

/// Weighted index pick over `weights` (sum `total`), consuming one f64 from `r`.
/// Falls back to the last index for float drift. Used by PFSP opponent sampling.
fn weighted_pick(weights: &[f64], total: f64, r: &mut SplitMix64) -> usize {
    let mut x = r.next_f64() * total;
    for (i, &w) in weights.iter().enumerate() {
        x -= w;
        if x <= 0.0 {
            return i;
        }
    }
    weights.len().saturating_sub(1)
}

// ===========================================================================
// Mutation.
// ===========================================================================

fn mutate(parent: &Genome, sigma: f64, rng: &mut SplitMix64) -> Genome {
    let params: Vec<f64> = parent
        .params
        .iter()
        .map(|&p| p + sigma * rng.next_gaussian())
        .collect();
    Genome {
        arch: parent.arch.clone(),
        params,
    }
}

// ===========================================================================
// Diversity: mean L2 distance of params from the population centroid.
// ===========================================================================

fn population_diversity(pop: &[Genome]) -> f64 {
    let n = pop.len();
    if n < 2 {
        return 0.0;
    }
    let dim = pop[0].params.len();
    let mut centroid = vec![0.0f64; dim];
    for g in pop {
        for (c, &p) in centroid.iter_mut().zip(g.params.iter()) {
            *c += p;
        }
    }
    for c in centroid.iter_mut() {
        *c /= n as f64;
    }
    let mut sum = 0.0f64;
    for g in pop {
        let mut d2 = 0.0f64;
        for (&p, &c) in g.params.iter().zip(centroid.iter()) {
            let d = p - c;
            d2 += d * d;
        }
        sum += d2.sqrt();
    }
    sum / n as f64
}

// ===========================================================================
// Initial population: small random genomes (Box–Muller * 0.5, matching the
// deterministic-genome init used elsewhere) seeded from the master.
// ===========================================================================

fn init_genome(arch: &[usize], rng: &mut SplitMix64) -> Genome {
    let n = param_count(arch);
    let params: Vec<f64> = (0..n).map(|_| rng.next_gaussian() * 0.5).collect();
    Genome {
        arch: arch.to_vec(),
        params,
    }
}

// ===========================================================================
// Resume: load a previous run's champion + HoF from a checkpoint dir.
// ===========================================================================

struct ResumeState {
    champion: Genome,
    hof: Vec<Genome>,
}

/// Validate that a genome matches the expected architecture and parameter count.
fn validate_genome(g: &Genome, what: &str, arch: &[usize], pcount: usize) -> Result<(), String> {
    if g.arch != arch {
        return Err(format!(
            "{what}: arch {:?} != expected DEFAULT_ARCH {:?}",
            g.arch, arch
        ));
    }
    if g.params.len() != pcount {
        return Err(format!(
            "{what}: param count {} != expected {}",
            g.params.len(),
            pcount
        ));
    }
    Ok(())
}

/// Load `<dir>/champion.json` (required) and `<dir>/hof.json` (optional → empty),
/// validating each genome against the trainer's architecture.
fn load_resume(dir: &std::path::Path, arch: &[usize], pcount: usize) -> Result<ResumeState, String> {
    let champ_path = dir.join("champion.json");
    if !champ_path.exists() {
        return Err(format!(
            "champion.json not found in {} (cannot resume)",
            dir.display()
        ));
    }
    let champ_str = std::fs::read_to_string(&champ_path)
        .map_err(|e| format!("reading {}: {e}", champ_path.display()))?;
    let champion = Genome::from_json(&champ_str)
        .map_err(|e| format!("parsing {}: {e}", champ_path.display()))?;
    validate_genome(&champion, "champion.json", arch, pcount)?;

    let hof_path = dir.join("hof.json");
    let hof: Vec<Genome> = if hof_path.exists() {
        let hof_str = std::fs::read_to_string(&hof_path)
            .map_err(|e| format!("reading {}: {e}", hof_path.display()))?;
        let v: Vec<Genome> = serde_json::from_str(&hof_str)
            .map_err(|e| format!("parsing {}: {e}", hof_path.display()))?;
        for (i, g) in v.iter().enumerate() {
            validate_genome(g, &format!("hof.json[{i}]"), arch, pcount)?;
        }
        v
    } else {
        Vec::new()
    };

    eprintln!(
        "cp-train: resuming from {} (champion + HoF size {})",
        dir.display(),
        hof.len()
    );
    Ok(ResumeState { champion, hof })
}

/// Count newline-terminated lines in an existing file (0 if absent/unreadable).
fn count_lines(path: &std::path::Path) -> usize {
    match std::fs::read_to_string(path) {
        Ok(s) => s.lines().count(),
        Err(_) => 0,
    }
}

// ===========================================================================
// Main.
// ===========================================================================

fn main() {
    let cfg = match parse_args() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("cp-train: {e}");
            std::process::exit(2);
        }
    };

    let arch: Vec<usize> = cp_ai::DEFAULT_ARCH.to_vec();
    let pcount = param_count(&arch);

    if let Err(e) = std::fs::create_dir_all(&cfg.out) {
        eprintln!("cp-train: cannot create {}: {e}", cfg.out.display());
        std::process::exit(2);
    }

    eprintln!(
        "cp-train: gens={} pop={} elite={} games={} seed={} cap={} arch={:?} ({} params) out={}",
        cfg.gens, cfg.pop, cfg.elite, cfg.games, cfg.seed, cfg.cap, arch, pcount, cfg.out.display()
    );

    // --- resume state (loaded BEFORE we would overwrite champion/hof/log) ------
    // When `--resume <DIR>` is given we load the previous champion + HoF here,
    // seed the population from the champion, and append to (not truncate) the log.
    let resumed = match &cfg.resume {
        Some(dir) => match load_resume(dir, &arch, pcount) {
            Ok(state) => Some(state),
            Err(e) => {
                eprintln!("cp-train: --resume {}: {e}", dir.display());
                std::process::exit(2);
            }
        },
        None => None,
    };

    // --- initial population ---------------------------------------------------
    let mut pop: Vec<Genome> = match &resumed {
        // Continue from the trained champion: genome 0 is the champion unchanged,
        // the remaining pop-1 slots are mutated copies (gen-0 sigma0, deterministic).
        Some(state) => {
            let mut r = SplitMix64::new(derive(cfg.seed, 0x1417, 0, 0));
            let mut v: Vec<Genome> = Vec::with_capacity(cfg.pop);
            v.push(state.champion.clone());
            while v.len() < cfg.pop {
                v.push(mutate(&state.champion, cfg.sigma0, &mut r));
            }
            v
        }
        // Fresh: small random genomes seeded from the master.
        None => {
            let mut r = SplitMix64::new(derive(cfg.seed, 0x1417, 0, 0));
            (0..cfg.pop).map(|_| init_genome(&arch, &mut r)).collect()
        }
    };

    // Hall of Fame — pre-loaded on resume so self-play opponents include past champions.
    let mut hof: Vec<Genome> = match &resumed {
        Some(state) => state.hof.clone(),
        None => Vec::new(),
    };

    // log.jsonl. On resume (when the file exists) append and offset the gen counter
    // by the existing line count so the dashboard x-axis stays continuous; otherwise
    // truncate/create.
    let log_path = cfg.out.join("log.jsonl");
    let gen_offset = if resumed.is_some() {
        count_lines(&log_path)
    } else {
        0
    };
    let log_open = if resumed.is_some() && log_path.exists() {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
    } else {
        std::fs::File::create(&log_path)
    };
    let mut log_file = match log_open {
        Ok(f) => f,
        Err(e) => {
            eprintln!("cp-train: cannot open {}: {e}", log_path.display());
            std::process::exit(2);
        }
    };

    // Pre-seed best_overall with the loaded champion at NEG_INFINITY fitness so it
    // is re-evaluated fairly (any positive gen-0 best replaces it) yet the on-disk
    // champion never regresses below the one we resumed from: if every generation
    // this session underperforms, the per-gen checkpoint keeps re-writing the same
    // loaded champion rather than a worse genome.
    let mut best_overall: Option<(f64, Genome)> = match &resumed {
        Some(state) => Some((f64::NEG_INFINITY, state.champion.clone())),
        None => None,
    };

    // PFSP opponent strength, aligned 1:1 with `pop` (the previous gen's fitness;
    // 0.0 at gen 0 → uniform). `hof_strength` parallels `hof`. Only consulted when
    // `cfg.pfsp > 0.0`. Resumed HoF entries get a neutral mid strength.
    let mut pop_strength: Vec<f64> = vec![0.0; pop.len()];
    let mut hof_strength: Vec<f64> = vec![0.5; hof.len()];
    // Novelty behavior archive (only grown when `cfg.novelty > 0.0`).
    let mut archive: Vec<[f64; BD_DIM]> = Vec::new();

    let t_start = Instant::now();

    for gen in 0..cfg.gens {
        let gen_start = Instant::now();
        // Slow sigma anneal: linearly decay sigma0 -> sigma1 over the FIRST 60%
        // of generations, then HOLD at sigma1 for the remainder.
        let anneal_gens = (0.6 * cfg.gens as f64).max(1.0);
        let frac = (gen as f64 / anneal_gens).min(1.0);
        let sigma = cfg.sigma0 + (cfg.sigma1 - cfg.sigma0) * frac;
        let w_t = 0.6 * (1.0 - gen as f64 / (0.6 * cfg.gens as f64)).max(0.0);

        // --- evaluate every genome over `games` games, in parallel -----------
        // The opponent pool is {population} ∪ {HoF}; sampled deterministically.
        let pool_len = pop.len() + hof.len();

        // PFSP weights over the pool (pop strengths then HoF strengths), softmax of
        // strength at temperature `cfg.pfsp`. Empty when PFSP is off → uniform path.
        let pool_weights: Vec<f64> = if cfg.pfsp > 0.0 {
            let strengths = pop_strength.iter().copied().chain(hof_strength.iter().copied());
            let max_s = strengths
                .clone()
                .fold(f64::NEG_INFINITY, f64::max);
            let max_s = if max_s.is_finite() { max_s } else { 0.0 };
            strengths.map(|s| (cfg.pfsp * (s - max_s)).exp()).collect()
        } else {
            Vec::new()
        };
        let total_weight: f64 = pool_weights.iter().sum();
        let pfsp_on = cfg.pfsp > 0.0 && total_weight > 0.0 && pool_weights.len() == pool_len;
        let pool_weights = &pool_weights;

        let evals: Vec<GenomeEval> = (0..pop.len())
            .into_par_iter()
            .map(|gi| {
                let mut acc = GenomeEval::default();
                for game_idx in 0..cfg.games {
                    let spec = game_spec(&cfg, gen, gi, game_idx);

                    // Sample opponents for the remaining seats from the pool.
                    let mut osel = SplitMix64::new(derive(
                        cfg.seed,
                        0x0FF0,
                        gen as u64,
                        (gi as u64) << 24 | (game_idx as u64) << 4,
                    ));
                    // Build the genome list: seat 0 = evaluated genome, rest sampled.
                    let mut genomes: Vec<Genome> = Vec::with_capacity(spec.players);
                    genomes.push(pop[gi].clone());
                    for _ in 1..spec.players {
                        // PFSP: prioritize stronger opponents; else uniform (legacy,
                        // byte-reproducible — uses the exact same RNG draw count).
                        let idx = if pfsp_on {
                            weighted_pick(pool_weights, total_weight, &mut osel)
                        } else {
                            osel.next_usize(pool_len)
                        };
                        let opp = if idx < pop.len() {
                            &pop[idx]
                        } else {
                            &hof[idx - pop.len()]
                        };
                        genomes.push(opp.clone());
                    }

                    let t: GameTelemetry = run_game_telemetry(
                        spec.seed,
                        spec.width,
                        spec.height,
                        &genomes,
                        &TRAINING_CONFIG,
                        spec.cap,
                    );

                    let s = eval_seat(&t);
                    acc.fitness +=
                        fitness_v2(s, &cfg.reward, gen, cfg.gens, spec.cap, t.total_tiles);
                    acc.avg_game_len += t.rounds as f64;
                    acc.bankrupt_games += if s.bankrupt { 1.0 } else { 0.0 };
                    acc.sum_tile_frac += s.tile_frac;
                    acc.sum_net_income += s.net_money_per_round;
                    // Behavior-descriptor components (means over games).
                    acc.sum_conquered_frac +=
                        s.enemy_tiles_conquered as f64 / t.total_tiles.max(1) as f64;
                    acc.sum_productive_area += s.mean_productive_area;
                    acc.sum_military_lead += s.mean_military_lead;
                    acc.sum_survived_frac += s.survived_rounds as f64 / spec.cap.max(1) as f64;
                    acc.games += 1;
                }
                let g = acc.games.max(1) as f64;
                GenomeEval {
                    fitness: acc.fitness / g,
                    avg_game_len: acc.avg_game_len / g,
                    bankrupt_games: acc.bankrupt_games,
                    sum_tile_frac: acc.sum_tile_frac / g,
                    sum_net_income: acc.sum_net_income / g,
                    sum_conquered_frac: acc.sum_conquered_frac / g,
                    sum_productive_area: acc.sum_productive_area / g,
                    sum_military_lead: acc.sum_military_lead / g,
                    sum_survived_frac: acc.sum_survived_frac / g,
                    games: acc.games,
                }
            })
            .collect();

        // --- novelty (Lehman & Stanley) --------------------------------------
        // Behavior descriptors for this gen, then per-genome novelty = mean kNN
        // distance among {this gen ∪ archive}. Only when --novelty > 0; otherwise
        // novelty is all-zero and `score == fitness` (legacy ordering preserved).
        let bds: Vec<[f64; BD_DIM]> = evals.iter().map(behavior_descriptor).collect();
        let novelty: Vec<f64> = if cfg.novelty > 0.0 {
            // Reference set: this gen's BDs (indices 0..pop) followed by the archive.
            let mut refs: Vec<[f64; BD_DIM]> = bds.clone();
            refs.extend(archive.iter().copied());
            let k = cfg.novelty_k.max(1);
            (0..pop.len())
                .map(|i| {
                    let mut ds: Vec<f64> = Vec::with_capacity(refs.len());
                    for (j, o) in refs.iter().enumerate() {
                        if j == i {
                            continue; // skip self (self lives at index i in the bds prefix)
                        }
                        ds.push(bd_dist(&bds[i], o));
                    }
                    ds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let kk = k.min(ds.len()).max(1);
                    ds.iter().take(kk).sum::<f64>() / kk as f64
                })
                .collect()
        } else {
            vec![0.0; pop.len()]
        };
        // Selection score = raw fitness + novelty bonus (== fitness when novelty off).
        let score: Vec<f64> = (0..pop.len())
            .map(|i| evals[i].fitness + cfg.novelty * novelty[i])
            .collect();

        // --- rank by SELECTION score (drives elites/parents) -----------------
        let mut order: Vec<usize> = (0..pop.len()).collect();
        order.sort_by(|&a, &b| {
            score[b]
                .partial_cmp(&score[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // --- stats -----------------------------------------------------------
        // The SHIPPED champion is the best by RAW fitness (not selection score),
        // so novelty/PFSP can steer evolution without ever shipping a merely-novel
        // genome. With novelty off this equals `order[0]`.
        // First maximal index by raw fitness (strict `>`), matching the legacy
        // stable descending sort's `order[0]` on ties → baseline stays identical.
        let mut best_perf_idx = 0;
        for i in 1..pop.len() {
            if evals[i].fitness > evals[best_perf_idx].fitness {
                best_perf_idx = i;
            }
        }
        let fits: Vec<f64> = (0..pop.len()).map(|i| evals[i].fitness).collect();
        let best_fit = evals[best_perf_idx].fitness;
        let mean_fit = fits.iter().sum::<f64>() / fits.len() as f64;
        let median_fit = {
            let mut s = fits.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = s.len();
            if n % 2 == 1 {
                s[n / 2]
            } else {
                0.5 * (s[n / 2 - 1] + s[n / 2])
            }
        };
        let fit_std = {
            let var = fits.iter().map(|f| (f - mean_fit).powi(2)).sum::<f64>() / fits.len() as f64;
            var.sqrt()
        };

        let best_idx = best_perf_idx;
        let avg_game_len = evals.iter().map(|e| e.avg_game_len).sum::<f64>() / evals.len() as f64;
        let bankrupt_games: f64 = evals.iter().map(|e| e.bankrupt_games).sum();
        let bankrupt_rate = bankrupt_games / (pop.len() * cfg.games) as f64;
        let champ_tile_frac = evals[best_idx].sum_tile_frac;
        let champ_net_income = evals[best_idx].sum_net_income;
        let diversity = population_diversity(&pop);
        let novelty_mean = if cfg.novelty > 0.0 {
            novelty.iter().sum::<f64>() / novelty.len() as f64
        } else {
            0.0
        };

        let elapsed = t_start.elapsed().as_secs_f64();
        let gen_secs = gen_start.elapsed().as_secs_f64();
        let games_per_sec = if gen_secs > 0.0 {
            (pop.len() * cfg.games) as f64 / gen_secs
        } else {
            0.0
        };

        // Track best-ever genome (champion).
        let champ = pop[best_idx].clone();
        match &best_overall {
            Some((bf, _)) if *bf >= best_fit => {}
            _ => best_overall = Some((best_fit, champ.clone())),
        }

        // --- log line --------------------------------------------------------
        let line = format!(
            "{{\"gen\":{},\"bestFit\":{},\"meanFit\":{},\"medianFit\":{},\"fitStd\":{},\
             \"sigma\":{},\"wT\":{},\"avgGameLen\":{},\"bankruptRate\":{},\
             \"championTileFrac\":{},\"championNetIncome\":{},\"populationDiversity\":{},\
             \"noveltyMean\":{},\"gamesPerSec\":{},\"elapsedSec\":{},\"winRateVsHeur\":null}}",
            gen + gen_offset,
            j(best_fit),
            j(mean_fit),
            j(median_fit),
            j(fit_std),
            j(sigma),
            j(w_t),
            j(avg_game_len),
            j(bankrupt_rate),
            j(champ_tile_frac),
            j(champ_net_income),
            j(diversity),
            j(novelty_mean),
            j(games_per_sec),
            j(elapsed),
        );
        if let Err(e) = writeln!(log_file, "{line}") {
            eprintln!("cp-train: log write failed: {e}");
        }
        let _ = log_file.flush();

        eprintln!(
            "gen {:>3}/{}  best={:+.4} mean={:+.4} med={:+.4} std={:.4} tf={:.3} nov={:.3} \
             sigma={:.4} wT={:.3} gLen={:.1} bankrupt={:.2} div={:.3} {:.1} g/s",
            gen + gen_offset,
            cfg.gens + gen_offset,
            best_fit,
            mean_fit,
            median_fit,
            fit_std,
            champ_tile_frac,
            novelty_mean,
            sigma,
            w_t,
            avg_game_len,
            bankrupt_rate,
            diversity,
            games_per_sec,
        );

        // --- Hall of Fame snapshot ------------------------------------------
        if cfg.hof_every > 0 && gen % cfg.hof_every == 0 {
            hof.push(champ.clone());
            hof_strength.push(best_fit); // PFSP: HoF entries are past champions → strong.
            if hof.len() > cfg.hof_max {
                // Drop oldest (keep the strength vector aligned).
                hof.remove(0);
                hof_strength.remove(0);
            }
        }

        // --- novelty archive: record the most novel behavior this gen --------
        if cfg.novelty > 0.0 {
            let most_novel = (0..pop.len())
                .max_by(|&a, &b| {
                    novelty[a]
                        .partial_cmp(&novelty[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);
            archive.push(bds[most_novel]);
            if archive.len() > cfg.novelty_archive_max {
                archive.remove(0);
            }
        }

        // --- checkpoint champion + HoF every generation (crash/stop safe) ----
        // A long run can be interrupted at any time; the latest best-ever
        // genome is always on disk. The final write below is then redundant
        // but harmless.
        if let Some((_, ref g)) = best_overall {
            let cp = cfg.out.join("champion.json");
            if let Err(e) = std::fs::write(&cp, g.to_json()) {
                eprintln!("cp-train: champion checkpoint write failed: {e}");
            }
        }
        if let Ok(hj) = serde_json::to_string(&hof) {
            let _ = std::fs::write(cfg.out.join("hof.json"), hj);
        }

        // --- build the next generation (skip on the last gen) ----------------
        if gen + 1 == cfg.gens {
            break;
        }
        let mut next: Vec<Genome> = Vec::with_capacity(pop.len());
        // `next_strength` parallels `next` so PFSP can prioritize next gen's
        // opponents by their (inherited) strength estimate: an elite keeps its own
        // fitness; a child inherits its parent's fitness as a prior.
        let mut next_strength: Vec<f64> = Vec::with_capacity(pop.len());
        // Elites carry over unchanged.
        for &i in order.iter().take(cfg.elite) {
            next.push(pop[i].clone());
            next_strength.push(evals[i].fitness);
        }
        // Fill the rest by mutating a random elite parent.
        let mut mrng = SplitMix64::new(derive(cfg.seed, 0x4D07A7E, gen as u64, 0));
        while next.len() < pop.len() {
            let parent_rank = mrng.next_usize(cfg.elite);
            let parent_idx = order[parent_rank];
            next.push(mutate(&pop[parent_idx], sigma, &mut mrng));
            next_strength.push(evals[parent_idx].fitness);
        }
        pop = next;
        pop_strength = next_strength;
    }

    // --- write champion + HoF ------------------------------------------------
    let champion = best_overall
        .map(|(_, g)| g)
        .unwrap_or_else(|| pop[0].clone());
    let champ_path = cfg.out.join("champion.json");
    if let Err(e) = std::fs::write(&champ_path, champion.to_json()) {
        eprintln!("cp-train: cannot write {}: {e}", champ_path.display());
        std::process::exit(2);
    }

    let hof_path = cfg.out.join("hof.json");
    let hof_json = serde_json::to_string(&hof).expect("hof serialises");
    if let Err(e) = std::fs::write(&hof_path, hof_json) {
        eprintln!("cp-train: cannot write {}: {e}", hof_path.display());
        std::process::exit(2);
    }

    eprintln!(
        "cp-train: done in {:.1}s — wrote {}, {} (HoF size {}), {}",
        t_start.elapsed().as_secs_f64(),
        champ_path.display(),
        hof_path.display(),
        hof.len(),
        log_path.display(),
    );
}

// JSON-safe float formatter: emit `null` for non-finite values so log.jsonl
// always parses (NaN/Infinity are not valid JSON).
fn j(x: f64) -> String {
    if x.is_finite() {
        // Enough precision to round-trip; trims trailing zeros via {}.
        format!("{x}")
    } else {
        "null".to_string()
    }
}
