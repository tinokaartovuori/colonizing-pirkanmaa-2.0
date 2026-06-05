//! `parity` — Milestone 5 conformance gate.
//!
//! Replays every golden trace (`rust-trainer/golden/trace-*.json`, exported from
//! the authoritative TS engine by `training/export-golden.ts`) against the Rust
//! `cp-sim` + `cp-ai` stack and asserts bit-close parity at EVERY checkpoint:
//!
//!   - the worldgen map (tile grid + per-seat HQ placement),
//!   - every NN decision (36-dim globalVec, candidate list w/ 10-dim locals,
//!     per-candidate scores, chosen index/intent),
//!   - the state fingerprint before/after each turn AND after each end_turn,
//!   - the final game result (winner, rounds, reason).
//!
//! It mirrors `cp_ai::run::run_game` / `export-golden.ts runGame` exactly: HQ
//! placement per seat (one `change_turn` each), then one player-turn + `end_turn`
//! per outer iteration, until one player remains or the round cap is hit. The
//! single shared `XorShift32(seed)` RNG is never consumed at temperature=0 /
//! blunder=0, but is constructed identically for faithfulness.
//!
//! On the first divergence it reports trace file, game-round, seat, decision
//! index, the field, and expected-vs-actual, then fails that trace. Exit code is
//! non-zero if ANY trace fails.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use cp_ai::controller::{DecisionTrace, NeuralAiController};
use cp_ai::mlp::Genome;
use cp_ai::policy::XorShift32;
use cp_ai::tiers::{TierConfig, TRAINING_CONFIG};
use cp_sim::resources::BasicResource;
use cp_sim::{EndTurnOutcome, Game, PlayerId, UnitType};

// Float comparison tolerance. The TS and Rust both do f64 math in the same
// operation order, so values should match bit-for-bit; this guards only against
// genuine order-independent last-ULP rounding. Empirically every value matches
// exactly (max observed |diff| == 0.0), so this never actually relaxes a check.
const FEPS: f64 = 1e-9;

// ---------------------------------------------------------------------------
// Golden-trace deserialization (mirrors training/export-golden.ts schema v1).
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct Trace {
    #[allow(dead_code)]
    #[serde(rename = "schemaVersion")]
    schema_version: i64,
    seed: u32,
    #[serde(rename = "mapWidth")]
    map_width: i32,
    #[serde(rename = "mapHeight")]
    map_height: i32,
    #[serde(rename = "playerCount")]
    player_count: usize,
    #[serde(rename = "roundCap")]
    round_cap: i64,
    config: GoldenConfig,
    #[serde(rename = "genomeSource")]
    genome_source: String,
    #[serde(rename = "genomeArch")]
    genome_arch: Vec<usize>,
    #[serde(rename = "genomeParamCount")]
    genome_param_count: usize,
    #[serde(rename = "hqPlacementTileIndex")]
    hq_placement_tile_index: Vec<i64>,
    map: Vec<MapTile>,
    rounds: Vec<RoundRecord>,
    result: GoldenResult,
}

#[derive(Deserialize)]
struct GoldenConfig {
    budget: i64,
    temperature: f64,
    reserve: i64,
    blunder: f64,
    experts: bool,
    military: bool,
    nuclear: bool,
    // Added in the Strange-Device arc; `serde(default)` lets pre-device golden
    // traces (no `device` field) still decode as `false`.
    #[serde(default)]
    device: bool,
}

#[derive(Deserialize)]
struct MapTile {
    #[allow(dead_code)]
    x: i32,
    #[allow(dead_code)]
    y: i32,
    #[serde(rename = "type")]
    ty: String,
    building: String,
}

#[derive(Deserialize)]
struct RoundRecord {
    round: i64,
    turns: Vec<TurnRecord>,
    #[serde(rename = "afterEndTurn")]
    after_end_turn: Fingerprint,
}

#[derive(Deserialize)]
struct TurnRecord {
    #[allow(dead_code)]
    round: i64,
    #[serde(rename = "currentPlayerNum")]
    current_player_num: i64,
    before: Fingerprint,
    #[serde(rename = "afterTurn")]
    after_turn: Fingerprint,
    decisions: Vec<DecisionRecord>,
}

#[derive(Deserialize)]
struct DecisionRecord {
    #[allow(dead_code)]
    round: i64,
    #[serde(rename = "globalVec")]
    global_vec: Vec<f64>,
    candidates: Vec<CandidateRecord>,
    scores: Vec<f64>,
    #[serde(rename = "chosenCandidateIndex")]
    chosen_candidate_index: usize,
    #[serde(rename = "chosenIntent")]
    chosen_intent: usize,
}

#[derive(Deserialize)]
struct CandidateRecord {
    intent: usize,
    local: Vec<f64>,
    label: String,
}

#[derive(Deserialize)]
struct Fingerprint {
    players: Vec<PlayerFingerprint>,
    tiles: Vec<TileFingerprint>,
}

#[derive(Deserialize)]
struct PlayerFingerprint {
    num: i64,
    alive: bool,
    money: i64,
    wood: i64,
    stone: i64,
    metal: i64,
}

#[derive(Deserialize)]
struct TileFingerprint {
    o: i64,
    b: String,
    u: String,
    c: String,
}

#[derive(Deserialize)]
struct GoldenResult {
    #[serde(rename = "winnerNum")]
    winner_num: Option<i64>,
    reason: String,
    rounds: i64,
    #[allow(dead_code)]
    crashed: bool,
}

// ---------------------------------------------------------------------------
// Rust-side fingerprinting (mirrors export-golden.ts `fingerprint`).
// ---------------------------------------------------------------------------

fn unit_codes(g: &Game, units: &[cp_sim::UnitId]) -> String {
    units
        .iter()
        .map(|&u| {
            let unit = &g.units[u.0];
            let seat = unit.owner.map(|p| g.players[p.0].player_num).unwrap_or(0);
            let code = match unit.kind {
                UnitType::BasicWorker => 'W',
                UnitType::Expert => 'E',
                UnitType::Soldier => 'S',
            };
            format!("{}:{}", seat, code)
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn building_type_str(b: Option<&cp_sim::Building>) -> String {
    match b {
        None => String::new(),
        Some(b) => b.kind.as_str().to_string(),
    }
}

struct RustTile {
    o: i64,
    b: String,
    u: String,
    c: String,
}

struct RustPlayer {
    num: i64,
    alive: bool,
    money: i64,
    wood: i64,
    stone: i64,
    metal: i64,
}

struct RustFingerprint {
    players: Vec<RustPlayer>,
    tiles: Vec<RustTile>,
}

fn fingerprint(g: &Game, n_players: usize) -> RustFingerprint {
    let alive: HashSet<PlayerId> = g.live_players().iter().copied().collect();
    let players = (0..n_players)
        .map(|i| {
            let pid = PlayerId(i);
            let r = &g.players[i].resources;
            RustPlayer {
                num: g.players[i].player_num,
                alive: alive.contains(&pid),
                money: r.get(BasicResource::Money).unwrap_or(0),
                wood: r.get(BasicResource::Wood).unwrap_or(0),
                stone: r.get(BasicResource::Stone).unwrap_or(0),
                metal: r.get(BasicResource::Metal).unwrap_or(0),
            }
        })
        .collect();
    let tiles = g
        .get_tiles()
        .iter()
        .map(|t| RustTile {
            o: t.owner.map(|p| g.players[p.0].player_num).unwrap_or(0),
            b: building_type_str(t.building.as_ref()),
            u: unit_codes(g, &t.units),
            c: unit_codes(g, &t.conquering_units),
        })
        .collect();
    RustFingerprint { players, tiles }
}

// ---------------------------------------------------------------------------
// Parity assertions. Each returns Err(message) on the first divergence so the
// caller can fail the trace precisely.
// ---------------------------------------------------------------------------

type PResult = Result<(), String>;

fn float_eq(a: f64, b: f64) -> bool {
    if a == b {
        return true;
    }
    if a.is_nan() && b.is_nan() {
        return true;
    }
    let diff = (a - b).abs();
    if diff <= FEPS {
        return true;
    }
    let scale = a.abs().max(b.abs());
    diff <= FEPS * scale
}

fn check_fingerprint(label: &str, exp: &Fingerprint, got: &RustFingerprint) -> PResult {
    if exp.players.len() != got.players.len() {
        return Err(format!(
            "{}: player count {} != {}",
            label,
            got.players.len(),
            exp.players.len()
        ));
    }
    for (i, (e, g)) in exp.players.iter().zip(got.players.iter()).enumerate() {
        if e.num != g.num
            || e.alive != g.alive
            || e.money != g.money
            || e.wood != g.wood
            || e.stone != g.stone
            || e.metal != g.metal
        {
            return Err(format!(
                "{}: player[{}] expected {{num:{} alive:{} money:{} wood:{} stone:{} metal:{}}} \
                 got {{num:{} alive:{} money:{} wood:{} stone:{} metal:{}}}",
                label,
                i,
                e.num,
                e.alive,
                e.money,
                e.wood,
                e.stone,
                e.metal,
                g.num,
                g.alive,
                g.money,
                g.wood,
                g.stone,
                g.metal,
            ));
        }
    }
    if exp.tiles.len() != got.tiles.len() {
        return Err(format!(
            "{}: tile count {} != {}",
            label,
            got.tiles.len(),
            exp.tiles.len()
        ));
    }
    for (i, (e, g)) in exp.tiles.iter().zip(got.tiles.iter()).enumerate() {
        if e.o != g.o || e.b != g.b || e.u != g.u || e.c != g.c {
            return Err(format!(
                "{}: tile[{}] expected {{o:{} b:{:?} u:{:?} c:{:?}}} got {{o:{} b:{:?} u:{:?} c:{:?}}}",
                label, i, e.o, e.b, e.u, e.c, g.o, g.b, g.u, g.c
            ));
        }
    }
    Ok(())
}

fn check_decision(
    label: &str,
    exp: &DecisionRecord,
    got: &DecisionTrace,
) -> PResult {
    // globalVec
    if exp.global_vec.len() != got.global_vec.len() {
        return Err(format!(
            "{}: globalVec len {} != {}",
            label,
            got.global_vec.len(),
            exp.global_vec.len()
        ));
    }
    for (i, (e, g)) in exp.global_vec.iter().zip(got.global_vec.iter()).enumerate() {
        if !float_eq(*e, *g) {
            return Err(format!(
                "{}: globalVec[{}] expected {} got {} (|diff|={:e})",
                label,
                i,
                e,
                g,
                (e - g).abs()
            ));
        }
    }
    // candidates
    if exp.candidates.len() != got.candidates.len() {
        return Err(format!(
            "{}: candidate count {} != {} (exp intents {:?}, got intents {:?})",
            label,
            got.candidates.len(),
            exp.candidates.len(),
            exp.candidates.iter().map(|c| c.intent).collect::<Vec<_>>(),
            got.candidates.iter().map(|c| c.intent).collect::<Vec<_>>(),
        ));
    }
    for (i, (e, g)) in exp.candidates.iter().zip(got.candidates.iter()).enumerate() {
        if e.intent != g.intent {
            return Err(format!(
                "{}: candidate[{}] intent expected {} ({}) got {} ({})",
                label, i, e.intent, e.label, g.intent, g.label
            ));
        }
        if e.local.len() != g.local.len() {
            return Err(format!(
                "{}: candidate[{}] ({}) local len {} != {}",
                label,
                i,
                e.label,
                g.local.len(),
                e.local.len()
            ));
        }
        for (j, (le, lg)) in e.local.iter().zip(g.local.iter()).enumerate() {
            if !float_eq(*le, *lg) {
                return Err(format!(
                    "{}: candidate[{}] ({}) local[{}] expected {} got {} (|diff|={:e})",
                    label,
                    i,
                    e.label,
                    j,
                    le,
                    lg,
                    (le - lg).abs()
                ));
            }
        }
    }
    // scores
    if exp.scores.len() != got.scores.len() {
        return Err(format!(
            "{}: scores len {} != {}",
            label,
            got.scores.len(),
            exp.scores.len()
        ));
    }
    for (i, (e, g)) in exp.scores.iter().zip(got.scores.iter()).enumerate() {
        if !float_eq(*e, *g) {
            return Err(format!(
                "{}: scores[{}] (intent {}) expected {} got {} (|diff|={:e})",
                label,
                i,
                exp.candidates[i].intent,
                e,
                g,
                (e - g).abs()
            ));
        }
    }
    // chosen index + intent
    if exp.chosen_candidate_index != got.chosen_candidate_index {
        return Err(format!(
            "{}: chosenCandidateIndex expected {} got {}",
            label, exp.chosen_candidate_index, got.chosen_candidate_index
        ));
    }
    if exp.chosen_intent != got.chosen_intent {
        return Err(format!(
            "{}: chosenIntent expected {} got {}",
            label, exp.chosen_intent, got.chosen_intent
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Counters for the summary.
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Counts {
    decisions: usize,
    fingerprints: usize,
}

// ---------------------------------------------------------------------------
// Replay one trace and assert parity.
// ---------------------------------------------------------------------------

fn replay_trace(path: &Path, genome_dir: &Path) -> Result<Counts, String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| format!("read {}: {}", path.display(), e))?;
    let trace: Trace = serde_json::from_str(&raw)
        .map_err(|e| format!("parse {}: {}", path.display(), e))?;

    // --- config sanity (these traces all use TRAINING_CONFIG) ----------------
    let cfg = TierConfig {
        budget: trace.config.budget,
        temperature: trace.config.temperature,
        reserve: trace.config.reserve,
        blunder: trace.config.blunder,
        experts: trace.config.experts,
        military: trace.config.military,
        nuclear: trace.config.nuclear,
        device: trace.config.device,
    };
    if !same_config(&cfg, &TRAINING_CONFIG) {
        return Err(format!(
            "config mismatch: trace cfg differs from TRAINING_CONFIG (budget {} temp {} reserve {} blunder {} experts {} military {} nuclear {})",
            cfg.budget, cfg.temperature, cfg.reserve, cfg.blunder, cfg.experts, cfg.military, cfg.nuclear
        ));
    }

    // --- genome ---------------------------------------------------------------
    let genome = load_genome(&trace, genome_dir)?;
    if genome.arch != trace.genome_arch {
        return Err(format!(
            "genome arch {:?} != trace arch {:?}",
            genome.arch, trace.genome_arch
        ));
    }
    if genome.params.len() != trace.genome_param_count {
        return Err(format!(
            "genome param count {} != trace {}",
            genome.params.len(),
            trace.genome_param_count
        ));
    }

    let mut counts = Counts::default();

    // --- construct + worldgen -------------------------------------------------
    let n = trace.player_count;
    let names: Vec<String> = (0..n).map(|i| format!("P{}", i + 1)).collect();
    let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    let mut g = Game::new(trace.map_width, trace.map_height, &name_refs);
    g.generate_map(trace.map_width, trace.map_height, trace.seed);

    // Map parity: tile *types* are fixed at worldgen and never change, so they
    // are checked here against the freshly generated grid. (The `map` field's
    // `building` column reflects the FINAL board — export-golden evaluates
    // `mapSnapshot(om)` at game end — so buildings are checked after the replay
    // against the final Rust state, below.)
    if g.get_tiles().len() != trace.map.len() {
        return Err(format!(
            "map tile count {} != {}",
            g.get_tiles().len(),
            trace.map.len()
        ));
    }
    for (i, (e, t)) in trace.map.iter().zip(g.get_tiles().iter()).enumerate() {
        let got_ty = t.tile_type.as_str();
        if e.ty != got_ty {
            return Err(format!(
                "map tile[{}] type expected {:?} got {:?}",
                i, e.ty, got_ty
            ));
        }
    }

    // --- controllers + RNG ----------------------------------------------------
    let ctrls: Vec<NeuralAiController> = (0..n)
        .map(|_| NeuralAiController::new(&genome, cfg))
        .collect();
    let mut rng = XorShift32::new(trace.seed);

    // --- round 0: HQ placement per seat --------------------------------------
    let mut hq_indices: Vec<i64> = Vec::with_capacity(n);
    for _ in 0..n {
        let cur = g.current_player();
        ctrls[cur.0].place_headquarters(&mut g, cur);
        let hq = g.get_hq_tile(cur);
        hq_indices.push(hq.map(|t| t.0 as i64).unwrap_or(-1));
        g.change_turn();
    }
    if hq_indices != trace.hq_placement_tile_index {
        return Err(format!(
            "HQ placement indices expected {:?} got {:?}",
            trace.hq_placement_tile_index, hq_indices
        ));
    }

    // --- main loop ------------------------------------------------------------
    // Mirror export-golden: one RoundRecord per (seat) outer iteration.
    let mut winner: Option<PlayerId> = None;
    let mut tie = false;
    let mut rec_iter = trace.rounds.iter();

    while g.live_players().len() > 1 && g.get_rounds_played() < trace.round_cap {
        let rec = match rec_iter.next() {
            Some(r) => r,
            None => {
                return Err(format!(
                    "Rust ran more game-rounds than the trace has ({} records); \
                     diverged earlier without being caught",
                    trace.rounds.len()
                ));
            }
        };

        let round = g.get_rounds_played();
        if rec.round != round {
            return Err(format!(
                "round number: trace record says {} but engine is at {}",
                rec.round, round
            ));
        }
        let cur = g.current_player();
        let cur_num = g.players[cur.0].player_num;

        // The exporter records exactly one turn per RoundRecord.
        if rec.turns.len() != 1 {
            return Err(format!(
                "round {}: expected 1 turn record, found {}",
                round,
                rec.turns.len()
            ));
        }
        let turn = &rec.turns[0];
        if turn.current_player_num != cur_num {
            return Err(format!(
                "round {}: turn currentPlayerNum expected {} got {}",
                round, turn.current_player_num, cur_num
            ));
        }

        // before fingerprint (at turn start, before scaffold)
        let before = fingerprint(&g, n);
        check_fingerprint(
            &format!("[{}] round {} seat P{} before", file_name(path), round, cur_num),
            &turn.before,
            &before,
        )?;
        counts.fingerprints += 1;

        // Drive the real controller with the trace sink, checking each decision
        // against the golden record as it is produced.
        let mut decisions: Vec<DecisionTrace> = Vec::new();
        ctrls[cur.0].plan_turn(&mut g, cur, &mut rng, Some(&mut |d| decisions.push(d)));

        if decisions.len() != turn.decisions.len() {
            return Err(format!(
                "[{}] round {} seat P{}: decision count expected {} got {}",
                file_name(path),
                round,
                cur_num,
                turn.decisions.len(),
                decisions.len()
            ));
        }
        for (di, (e, g_dec)) in turn.decisions.iter().zip(decisions.iter()).enumerate() {
            check_decision(
                &format!(
                    "[{}] round {} seat P{} decision {}",
                    file_name(path),
                    round,
                    cur_num,
                    di
                ),
                e,
                g_dec,
            )?;
            counts.decisions += 1;
        }

        // afterTurn fingerprint (after scaffold + decision loop, before endTurn)
        let after_turn = fingerprint(&g, n);
        check_fingerprint(
            &format!("[{}] round {} seat P{} afterTurn", file_name(path), round, cur_num),
            &turn.after_turn,
            &after_turn,
        )?;
        counts.fingerprints += 1;

        // endTurn + afterEndTurn fingerprint
        match g.end_turn() {
            EndTurnOutcome::Win(p) => {
                winner = Some(p);
            }
            EndTurnOutcome::Tie => {
                tie = true;
            }
            EndTurnOutcome::Continue | EndTurnOutcome::PlayersLost(_) => {}
        }
        let after_end = fingerprint(&g, n);
        check_fingerprint(
            &format!("[{}] round {} seat P{} afterEndTurn", file_name(path), round, cur_num),
            &rec.after_end_turn,
            &after_end,
        )?;
        counts.fingerprints += 1;

        if winner.is_some() || tie {
            break;
        }
    }

    // The trace should not have more records than we replayed (unless game ended).
    if winner.is_none() && !tie {
        if let Some(_extra) = rec_iter.next() {
            return Err(format!(
                "[{}] engine stopped at round {} but trace has additional RoundRecords",
                file_name(path),
                g.get_rounds_played()
            ));
        }
    }

    // Final-board parity: the trace `map.building` column is a snapshot of the
    // board at game end. Verify the Rust final buildings match it tile-for-tile.
    for (i, (e, t)) in trace.map.iter().zip(g.get_tiles().iter()).enumerate() {
        let got_b = building_type_str(t.building.as_ref());
        if e.building != got_b {
            return Err(format!(
                "[{}] final map tile[{}] building expected {:?} got {:?}",
                file_name(path),
                i,
                e.building,
                got_b
            ));
        }
    }

    // --- final result ---------------------------------------------------------
    let total = g.get_tile_count();
    let (winner_num, reason) = if let Some(w) = winner {
        let owned = g.get_tile_count_for_player(w);
        let r = if total > 0 && (owned * 100) / total >= 70 {
            "domination"
        } else {
            "last-standing"
        };
        (Some(g.players[w.0].player_num), r.to_string())
    } else if tie {
        (None, "tie".to_string())
    } else if g.live_players().len() == 1 {
        let w = g.live_players()[0];
        (Some(g.players[w.0].player_num), "last-standing".to_string())
    } else {
        (None, "timeout".to_string())
    };

    check_result(&trace.result, winner_num, &reason, g.get_rounds_played())?;

    Ok(counts)
}

fn check_result(exp: &GoldenResult, winner_num: Option<i64>, reason: &str, rounds: i64) -> PResult {
    if exp.winner_num != winner_num {
        return Err(format!(
            "result winnerNum expected {:?} got {:?}",
            exp.winner_num, winner_num
        ));
    }
    if exp.reason != reason {
        return Err(format!(
            "result reason expected {:?} got {:?}",
            exp.reason, reason
        ));
    }
    if exp.rounds != rounds {
        return Err(format!(
            "result rounds expected {} got {}",
            exp.rounds, rounds
        ));
    }
    Ok(())
}

fn same_config(a: &TierConfig, b: &TierConfig) -> bool {
    a.budget == b.budget
        && a.temperature == b.temperature
        && a.reserve == b.reserve
        && a.blunder == b.blunder
        && a.experts == b.experts
        && a.military == b.military
        && a.nuclear == b.nuclear
        && a.device == b.device
}

// ---------------------------------------------------------------------------
// Genome loading: match the exporter's choice exactly.
// ---------------------------------------------------------------------------

fn load_genome(trace: &Trace, repo_root: &Path) -> Result<Genome, String> {
    if trace.genome_source == "training/checkpoints/champion.json" {
        let p = repo_root.join("training/checkpoints/champion.json");
        return Genome::from_file(&p.to_string_lossy())
            .map_err(|e| format!("load champion {}: {}", p.display(), e));
    }
    if let Some(rest) = trace.genome_source.strip_prefix("deterministic-lcg:seed=") {
        let seed = parse_seed(rest)?;
        return Ok(deterministic_genome(seed, &trace.genome_arch));
    }
    Err(format!("unknown genomeSource {:?}", trace.genome_source))
}

fn parse_seed(s: &str) -> Result<u32, String> {
    let s = s.trim();
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u32::from_str_radix(hex, 16).map_err(|e| format!("bad seed {:?}: {}", s, e))
    } else {
        s.parse::<u32>().map_err(|e| format!("bad seed {:?}: {}", s, e))
    }
}

/// Replicates `export-golden.ts deterministicGenome`: inline LCG → Box–Muller
/// normals * 0.5. Used only if a trace was made without a champion checkpoint.
fn deterministic_genome(seed: u32, arch: &[usize]) -> Genome {
    let n = cp_ai::mlp::param_count(arch);
    let mut params = vec![0.0f64; n];
    let mut s: u32 = if seed == 0 { 1 } else { seed };
    let mut next = || -> f64 {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        (s as f64) / 4294967296.0
    };
    for p in params.iter_mut() {
        let u1 = next().max(1e-9);
        let u2 = next();
        *p = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos() * 0.5;
    }
    Genome {
        arch: arch.to_vec(),
        params,
    }
}

// ---------------------------------------------------------------------------
// Discovery + main.
// ---------------------------------------------------------------------------

fn file_name(p: &Path) -> String {
    p.file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Resolve the repo root (where `training/checkpoints/champion.json` lives) and
/// the golden dir from CARGO_MANIFEST_DIR. CARGO_MANIFEST_DIR points at
/// `rust-trainer/crates/cp-train`; repo root is three levels up.
fn resolve_paths() -> (PathBuf, PathBuf) {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // crates/cp-train -> crates -> rust-trainer -> <repo root>
    let repo_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| manifest.clone());
    let golden = repo_root.join("rust-trainer/golden");
    (repo_root, golden)
}

fn discover_traces(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut traces: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| format!("read golden dir {}: {}", dir.display(), e))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            let name = file_name(p);
            name.starts_with("trace-") && name.ends_with(".json")
        })
        .collect();
    traces.sort();
    Ok(traces)
}

fn main() {
    let (repo_root, default_golden) = resolve_paths();

    // CLI: optional golden dir argument.
    let args: Vec<String> = std::env::args().skip(1).collect();
    let golden_dir = if let Some(a) = args.first() {
        PathBuf::from(a)
    } else {
        default_golden
    };

    let traces = match discover_traces(&golden_dir) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("parity: {}", e);
            std::process::exit(2);
        }
    };

    if traces.is_empty() {
        eprintln!("parity: no trace-*.json found in {}", golden_dir.display());
        std::process::exit(2);
    }

    println!(
        "parity: checking {} golden trace(s) in {}\n",
        traces.len(),
        golden_dir.display()
    );

    let mut passed = 0usize;
    let mut failed = 0usize;
    let mut total_decisions = 0usize;
    let mut total_fingerprints = 0usize;

    for path in &traces {
        match replay_trace(path, &repo_root) {
            Ok(counts) => {
                passed += 1;
                total_decisions += counts.decisions;
                total_fingerprints += counts.fingerprints;
                println!(
                    "  PASS  {:<18} {} decisions, {} fingerprints",
                    file_name(path),
                    counts.decisions,
                    counts.fingerprints
                );
            }
            Err(msg) => {
                failed += 1;
                println!("  FAIL  {}", file_name(path));
                println!("        {}", msg);
            }
        }
    }

    println!(
        "\nparity: {}/{} traces PASS  ({} decisions, {} fingerprints checked)",
        passed,
        traces.len(),
        total_decisions,
        total_fingerprints
    );

    if failed == 0 {
        println!("parity: ALL TRACES PASS — Rust sim + AI reproduce the TS engine exactly.");
        std::process::exit(0);
    } else {
        eprintln!("parity: {} trace(s) FAILED — see first divergence above.", failed);
        std::process::exit(1);
    }
}
