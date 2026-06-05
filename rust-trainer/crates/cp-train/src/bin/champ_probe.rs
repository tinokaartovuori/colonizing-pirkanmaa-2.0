//! Behavioural probe: run a champion (seat 0, MCTS) vs the HARD bot (seat 1) and
//! report — SPLIT BY OUTCOME — how much territory the champion holds, how many
//! soldiers it fields, and how many buildings it owns at game end. Answers the
//! question "how does it win N% while holding so few tiles?": are its wins
//! territorial conquests, narrow HQ-snipes, or opponent blunders? And does it
//! actually build military + expand, or just hoard economy?
//!
//! Usage:
//!   cargo run --release -p cp-train --bin champ_probe -- \
//!     --champion checkpoints-az8/champion.json --value checkpoints-az8/value.json \
//!     --sims 96 --games 200 --cap 120 --seed 7 --threads 4

use std::path::PathBuf;

use cp_ai::{DecisionTrace, Genome, HardAi, LeafEval, NeuralAiController, SearchConfig, ValueNet, XorShift32, TRAINING_CONFIG};
use cp_sim::model::{BuildingType, ObjId};
use cp_sim::resources::BasicResource;
use cp_sim::{EndTurnOutcome, Game, PlayerId};
use rayon::prelude::*;

/// Owns a building of `kind`?
fn owns_building(g: &Game, p: PlayerId, kind: BuildingType) -> bool {
    g.owned_tiles(p).into_iter().any(|t| matches!(&g.tiles[t.0].building, Some(b) if b.kind == kind))
}

/// DIAGNOSTIC army-forcing scaffold (opt-in, NOT on the parity path): each seat-0
/// turn, build the Mine→Outpost military chain and fill the soldier cap, using the
/// same public Game APIs the controller uses. This hands the NN an actual army so
/// its (already-expressive) Attack candidates can mass & assault — testing whether
/// army-building (problem ①) is the binding constraint, independent of the net's
/// inability to LEARN the long-horizon build chain.
fn ensure_military(g: &mut Game, p: PlayerId) {
    // 1. Mine on an owned mountain (metal source) if we have none.
    if !owns_building(g, p, BuildingType::Mine) {
        if let Some(t) = g.owned_tiles(p).into_iter().find(|&t|
            g.tiles[t.0].building.is_none() && g.buildable_buildings(t).contains(&"Mine")) {
            g.ai_build_building("Mine", t);
        }
    }
    // 2. Outpost (raises soldier cap by 3) if we have none.
    if !owns_building(g, p, BuildingType::Outpost) {
        if let Some(t) = g.owned_tiles(p).into_iter().find(|&t|
            g.tiles[t.0].building.is_none() && g.buildable_buildings(t).contains(&"Outpost")) {
            g.ai_build_building("Outpost", t);
        }
    }
    // 3. Fill the free soldier cap, placing on owned tiles with space.
    let mut guard = 0;
    while g.free_soldier_amount(p) > 0 && guard < 12 {
        guard += 1;
        let tile = g.owned_tiles(p).into_iter().find(|&t| g.tiles[t.0].has_space_for_units());
        match tile {
            Some(t) => { if !g.ai_buy_and_place_unit("Soldier", t) { break; } }
            None => break,
        }
    }
}

const RES: [BasicResource; 4] = [BasicResource::Money, BasicResource::Wood, BasicResource::Stone, BasicResource::Metal];
/// Did `p` go negative on any resource while still holding >= 8 tiles (i.e. a
/// self-inflicted bankruptcy, not a consequence of being cut down)?
fn bankrupt_while_healthy(g: &Game, p: PlayerId) -> bool {
    let neg = RES.iter().any(|&r| g.players[p.0].resources.get(r).unwrap_or(0) < 0);
    let tiles = g.players[p.0].objects.iter().filter(|o| matches!(o, ObjId::Tile(_))).count();
    neg && tiles >= 8
}

struct Cfg {
    champion: Option<PathBuf>, value: Option<PathBuf>,
    sims: usize, games: usize, cap: i64, width: i32, height: i32, seed: u32, threads: usize,
    force_military: bool,
    spatial_policy: bool,
}
impl Default for Cfg {
    fn default() -> Self {
        Cfg { champion: None, value: None, sims: 96, games: 200, cap: 120, width: 12, height: 12, seed: 7, threads: 4,
            force_military: false, spatial_policy: false }
    }
}

/// Per-game record. code: 0 = champ(seat0) win, 1 = hard(seat1) win, 2 = timeout/tie.
struct Rec {
    code: u8, rounds: i64,
    frac0: f64, frac1: f64,   // tile fractions of ALL tiles
    sol0: i64, sol1: i64,     // soldiers on the board
    bld0: i64, bld1: i64,     // buildings owned
    // Champion decision-making: how often each intent was CHOSEN, and how often
    // Expand/HireSoldier/Attack candidates were even AVAILABLE to choose.
    chosen: [u64; 11], decisions: u64,
    exp_avail: u64, hire_avail: u64, atk_avail: u64,
    // Benchmark integrity: did HARD (seat1) self-bankrupt while still healthy
    // during this game? If so a champ win here is tainted (not really our doing).
    hard_self_bankrupt: bool,
}

const INTENT_NAMES: [&str; 11] = [
    "BuildFarm", "BuildMine", "BuildVillage", "BuildOutpost", "BuildHydro",
    "BuildNuclear", "Expand", "HireSoldier", "Attack", "StackProducer", "Pass",
];

fn buildings_of(g: &Game, p: PlayerId) -> i64 {
    g.get_tiles().iter().filter(|t| t.owner == Some(p) && t.building.is_some()).count() as i64
}

fn play(seed: u32, genome: &Genome, value: Option<&ValueNet>, sc: &SearchConfig, cfg: &Cfg) -> Rec {
    let mut g = Game::new(cfg.width, cfg.height, &["P1", "P2"]);
    g.generate_map(cfg.width, cfg.height, seed);
    let champ = match value {
        Some(vn) => NeuralAiController::with_search_value(genome, TRAINING_CONFIG, *sc, vn),
        None => NeuralAiController::with_search(genome, TRAINING_CONFIG, *sc),
    };
    let mut hard = HardAi::hard();
    let mut rng = XorShift32::new(seed);
    for _ in 0..2 {
        let cur = g.current_player();
        if cur.0 == 0 { champ.place_headquarters(&mut g, cur); } else { hard.place_headquarters(&mut g, cur); }
        g.change_turn();
    }
    let mut winner: Option<PlayerId> = None;
    let mut decided = false;
    let mut chosen = [0u64; 11];
    let (mut decisions, mut exp_avail, mut hire_avail, mut atk_avail) = (0u64, 0u64, 0u64, 0u64);
    let mut hard_self_bankrupt = false;
    {
        let mut sink = |d: DecisionTrace| {
            decisions += 1;
            if d.chosen_intent < 11 { chosen[d.chosen_intent] += 1; }
            if d.candidates.iter().any(|c| c.intent == 6) { exp_avail += 1; }
            if d.candidates.iter().any(|c| c.intent == 7) { hire_avail += 1; }
            if d.candidates.iter().any(|c| c.intent == 8) { atk_avail += 1; }
        };
        while g.live_players().len() > 1 && g.get_rounds_played() < cfg.cap {
            let cur = g.current_player();
            if cur.0 == 0 {
                champ.plan_turn(&mut g, cur, &mut rng, Some(&mut sink));
                if cfg.force_military { ensure_military(&mut g, cur); }
            } else {
                hard.plan_turn(&mut g, cur);
            }
            let out = g.end_turn();
            if !hard_self_bankrupt && bankrupt_while_healthy(&g, PlayerId(1)) { hard_self_bankrupt = true; }
            match out {
                EndTurnOutcome::Win(p) => { winner = Some(p); decided = true; break; }
                EndTurnOutcome::Tie => { decided = true; break; }
                _ => {}
            }
        }
    }
    let winner = winner.or_else(|| { let l = g.live_players(); if l.len() == 1 { Some(l[0]) } else { None } });
    let code = match winner { Some(p) if p.0 == 0 => 0, Some(_) => 1, None => { let _ = decided; 2 } };
    let total = g.get_tile_count().max(1) as f64;
    Rec {
        code, rounds: g.get_rounds_played(),
        frac0: g.get_tile_count_for_player(PlayerId(0)) as f64 / total,
        frac1: g.get_tile_count_for_player(PlayerId(1)) as f64 / total,
        sol0: g.current_soldier_amount(PlayerId(0)), sol1: g.current_soldier_amount(PlayerId(1)),
        bld0: buildings_of(&g, PlayerId(0)), bld1: buildings_of(&g, PlayerId(1)),
        chosen, decisions, exp_avail, hire_avail, atk_avail, hard_self_bankrupt,
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
            "--champion" => cfg.champion = Some(PathBuf::from(v!())),
            "--value" => cfg.value = Some(PathBuf::from(v!())),
            "--sims" => cfg.sims = v!().parse().unwrap_or(cfg.sims),
            "--games" => cfg.games = v!().parse().unwrap_or(cfg.games),
            "--cap" => cfg.cap = v!().parse().unwrap_or(cfg.cap),
            "--width" => cfg.width = v!().parse().unwrap_or(cfg.width),
            "--height" => cfg.height = v!().parse().unwrap_or(cfg.height),
            "--seed" => cfg.seed = v!().parse().unwrap_or(cfg.seed),
            "--threads" => cfg.threads = v!().parse().unwrap_or(cfg.threads),
            "--force-military" => cfg.force_military = true,
            "--spatial-policy" => cfg.spatial_policy = true,
            _ => {}
        }
        i += 1;
    }
    rayon::ThreadPoolBuilder::new().num_threads(cfg.threads.max(1)).build_global().ok();

    let genome = Genome::from_file(&cfg.champion.as_ref().expect("--champion required").to_string_lossy())
        .expect("load champion");
    let value = cfg.value.as_ref().map(|p| ValueNet::from_file(&p.to_string_lossy()).expect("load value"));
    let leaf = if value.is_some() { LeafEval::Value } else { LeafEval::Static };
    let sc = SearchConfig { n_sims: cfg.sims, leaf_eval: leaf, seed: cfg.seed ^ 0xB17_C0DE, spatial_policy: cfg.spatial_policy, ..Default::default() };

    println!("champ_probe: {} games, champion={} value={} sims={} leaf={:?} on {}x{} cap {}",
        cfg.games, cfg.champion.as_ref().unwrap().display(),
        cfg.value.as_ref().map(|p| p.display().to_string()).unwrap_or("none".into()),
        cfg.sims, leaf, cfg.width, cfg.height, cfg.cap);

    let recs: Vec<Rec> = (0..cfg.games).into_par_iter()
        .map(|gi| play(cfg.seed.wrapping_add((gi as u32).wrapping_mul(2_654_435_761)), &genome, value.as_ref(), &sc, &cfg))
        .collect();

    let n = cfg.games.max(1) as f64;
    let (mut w, mut l, mut t) = (0usize, 0usize, 0usize);
    for r in &recs { match r.code { 0 => w += 1, 1 => l += 1, _ => t += 1 } }
    println!("\noutcomes: champ WIN {} ({:.1}%) | LOSS {} ({:.1}%) | TIMEOUT {} ({:.1}%)",
        w, 100.0 * w as f64 / n, l, 100.0 * l as f64 / n, t, 100.0 * t as f64 / n);

    // TILE-MAJORITY TIEBREAK (benchmark convention only — the sim/parity path is
    // untouched): score each TIMEOUT game by who holds more tiles at the cap.
    // The original game has no cap (it ends at 70% domination or elimination), so
    // this is purely a training/benchmark metric for the draw-prone capped games.
    let (mut tb_champ, mut tb_hard, mut tb_draw) = (0usize, 0usize, 0usize);
    for r in &recs {
        if r.code == 2 {
            if r.frac0 > r.frac1 { tb_champ += 1; }
            else if r.frac1 > r.frac0 { tb_hard += 1; }
            else { tb_draw += 1; }
        }
    }
    let win_tb = w + tb_champ;
    println!("tile-majority tiebreak on {} TIMEOUT games: champ-leads {} | hard-leads {} | even {}",
        t, tb_champ, tb_hard, tb_draw);
    println!("  >>> win-rate WITH tile-tiebreak: {}/{} = {:.1}%   (natural {:.1}%, gain +{:.1}pp)",
        win_tb, cfg.games, 100.0 * win_tb as f64 / n, 100.0 * w as f64 / n,
        100.0 * (win_tb as f64 - w as f64) / n);

    // Benchmark integrity: strip games where HARD self-bankrupted while healthy
    // (its economic suicide, not our doing) and report the honest win-rate.
    let tainted: usize = recs.iter().filter(|r| r.hard_self_bankrupt).count();
    let tainted_wins: usize = recs.iter().filter(|r| r.hard_self_bankrupt && r.code == 0).count();
    let clean_games = cfg.games - tainted;
    let legit_wins = w - tainted_wins;
    println!("benchmark integrity: hard self-bankrupted-while-healthy in {} games ({} of them we 'won').", tainted, tainted_wins);
    if clean_games > 0 {
        println!("  LEGITIMATE win-rate (tainted games removed): {}/{} = {:.1}%  (raw {:.1}%)",
            legit_wins, clean_games, 100.0 * legit_wins as f64 / clean_games as f64, 100.0 * w as f64 / n);
    }

    // Per-outcome behavioural means for the CHAMPION (seat 0) vs HARD (seat 1).
    println!("\nper-outcome means (champion seat0 vs hard seat1):");
    println!("  {:<10} {:>5} {:>8} | {:>11} {:>11} | {:>9} {:>9} | {:>7} {:>7}",
        "bucket", "n", "rounds", "champ tile%", "hard tile%", "champ sol", "hard sol", "champ b", "hard b");
    for (name, code) in [("WIN", 0u8), ("LOSS", 1), ("TIMEOUT", 2)] {
        let bucket: Vec<&Rec> = recs.iter().filter(|r| r.code == code).collect();
        if bucket.is_empty() { println!("  {:<10} {:>5}  (none)", name, 0); continue; }
        let m = bucket.len() as f64;
        let mean = |f: &dyn Fn(&Rec) -> f64| bucket.iter().map(|r| f(r)).sum::<f64>() / m;
        println!("  {:<10} {:>5} {:>8.1} | {:>10.1}% {:>10.1}% | {:>9.1} {:>9.1} | {:>7.1} {:>7.1}",
            name, bucket.len(),
            mean(&|r| r.rounds as f64),
            100.0 * mean(&|r| r.frac0), 100.0 * mean(&|r| r.frac1),
            mean(&|r| r.sol0 as f64), mean(&|r| r.sol1 as f64),
            mean(&|r| r.bld0 as f64), mean(&|r| r.bld1 as f64));
    }

    // What the champion actually CHOOSES, over every discretionary decision.
    let total_dec: u64 = recs.iter().map(|r| r.decisions).sum();
    let mut chosen_tot = [0u64; 11];
    for r in &recs { for k in 0..11 { chosen_tot[k] += r.chosen[k]; } }
    println!("\nchampion intent choices over {} decisions ({} games):", total_dec, cfg.games);
    let mut idx: Vec<usize> = (0..11).collect();
    idx.sort_by(|&a, &b| chosen_tot[b].cmp(&chosen_tot[a]));
    for k in idx {
        let pct = if total_dec > 0 { 100.0 * chosen_tot[k] as f64 / total_dec as f64 } else { 0.0 };
        println!("  {:<13} {:>7} ({:>5.1}%)", INTENT_NAMES[k], chosen_tot[k], pct);
    }
    // Availability vs choice for the expansion/military intents: were they even on
    // the menu, and how often picked WHEN available?
    let (exp_av, hire_av, atk_av): (u64, u64, u64) = recs.iter()
        .fold((0, 0, 0), |a, r| (a.0 + r.exp_avail, a.1 + r.hire_avail, a.2 + r.atk_avail));
    println!("\nexpansion/military availability (of {} decisions):", total_dec);
    let avail_line = |name: &str, avail: u64, chosen: u64| {
        let av_pct = if total_dec > 0 { 100.0 * avail as f64 / total_dec as f64 } else { 0.0 };
        let take_pct = if avail > 0 { 100.0 * chosen as f64 / avail as f64 } else { 0.0 };
        println!("  {:<12} available {:>6} ({:>5.1}% of decisions) | chosen when available: {:>5.1}%",
            name, avail, av_pct, take_pct);
    };
    avail_line("Expand", exp_av, chosen_tot[6]);
    avail_line("HireSoldier", hire_av, chosen_tot[7]);
    avail_line("Attack", atk_av, chosen_tot[8]);
}
