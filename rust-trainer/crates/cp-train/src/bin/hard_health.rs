//! Diagnostic: why does the HARD bot sometimes (a) never enter the game (0 tiles
//! after HQ placement) or (b) self-bankrupt in a handful of rounds with no enemy
//! pressure? Mirrors the replay setup: neural champ (seat 0, value-MCTS) vs HARD
//! (seat 1). Scans a seed range and flags the broken games; `--trace <seed>`
//! prints HARD's per-round tiles + resources so we can see the bleed.
//!
//!   cargo run --release -p cp-train --bin hard_health -- \
//!     --champion checkpoints-az/champion.json --value checkpoints-az/value.json \
//!     --width 14 --height 12 --cap 120 --scan 1000 [--trace 240260166]

use std::path::PathBuf;

use cp_ai::{Genome, HardAi, LeafEval, NeuralAiController, SearchConfig, ValueNet, XorShift32, TRAINING_CONFIG};
use cp_sim::resources::BasicResource;
use cp_sim::{EndTurnOutcome, Game, PlayerId};

const RES: [BasicResource; 4] = [BasicResource::Money, BasicResource::Wood, BasicResource::Stone, BasicResource::Metal];

struct Args { champion: PathBuf, value: Option<PathBuf>, width: i32, height: i32, cap: i64, scan: u32, trace: Option<u32> }

fn parse() -> Args {
    let a: Vec<String> = std::env::args().collect();
    let args = if let Some(p) = a.iter().position(|x| x == "--") { a[p + 1..].to_vec() } else { a[1..].to_vec() };
    let mut champion = PathBuf::from("checkpoints-az/champion.json");
    let (mut value, mut width, mut height, mut cap, mut scan, mut trace) = (None, 14, 12, 120i64, 500u32, None);
    let mut i = 0;
    while i < args.len() {
        let k = args[i].clone();
        macro_rules! v { () => {{ i += 1; args.get(i).cloned().unwrap_or_default() }} }
        match k.as_str() {
            "--champion" => champion = PathBuf::from(v!()),
            "--value" => value = Some(PathBuf::from(v!())),
            "--width" => width = v!().parse().unwrap_or(14),
            "--height" => height = v!().parse().unwrap_or(12),
            "--cap" => cap = v!().parse().unwrap_or(120),
            "--scan" => scan = v!().parse().unwrap_or(500),
            "--trace" => trace = v!().parse().ok(),
            _ => {}
        }
        i += 1;
    }
    Args { champion, value, width, height, cap, scan, trace }
}

fn hard_tiles(g: &Game) -> i64 { g.get_tile_count_for_player(PlayerId(1)) }
fn res(g: &Game, r: BasicResource) -> i64 { g.players[1].resources.get(r).unwrap_or(0) }

/// Returns (hard_tiles_after_placement, bankrupt_round_or_-1, final_rounds, hard_eliminated).
fn play(seed: u32, genome: &Genome, value: Option<&ValueNet>, a: &Args, trace: bool) -> (i64, i64, i64, bool) {
    let sc = SearchConfig { n_sims: 32, leaf_eval: if value.is_some() { LeafEval::Value } else { LeafEval::Static }, seed: seed ^ 0xB17_C0DE, ..Default::default() };
    let mut g = Game::new(a.width, a.height, &["P1", "P2"]);
    g.generate_map(a.width, a.height, seed);
    let champ = match value { Some(vn) => NeuralAiController::with_search_value(genome, TRAINING_CONFIG, sc, vn), None => NeuralAiController::with_search(genome, TRAINING_CONFIG, sc) };
    let mut hard = HardAi::hard();
    let mut rng = XorShift32::new(seed);
    for _ in 0..2 {
        let cur = g.current_player();
        if cur.0 == 0 { champ.place_headquarters(&mut g, cur); } else { hard.place_headquarters(&mut g, cur); }
        g.change_turn();
    }
    let tiles_after_placement = hard_tiles(&g);
    if trace {
        println!("seed {seed}: after placement HARD tiles={} | {}", tiles_after_placement,
            RES.iter().map(|&r| format!("{:?}={}", r, res(&g, r))).collect::<Vec<_>>().join(" "));
    }
    let mut bankrupt_round = -1i64;
    let mut eliminated = false;
    while g.live_players().len() > 1 && g.get_rounds_played() < a.cap {
        let cur = g.current_player();
        if cur.0 == 0 { champ.plan_turn(&mut g, cur, &mut rng, None); } else { hard.plan_turn(&mut g, cur); }
        if cur.0 == 1 && trace {
            let p1 = PlayerId(1);
            let workers = g.current_basic_worker_amount(p1);
            let experts = g.current_expert_amount(p1);
            let soldiers = g.current_soldier_amount(p1);
            let salary = workers * 5 + experts * 25 + soldiers * 30;
            // building + staffing composition
            let mut farms = 0; let mut farms_staffed = 0; let mut forests_worked = 0; let mut villages = 0; let mut outposts = 0; let mut mines = 0;
            for &t in &g.owned_tiles(p1) {
                let staffed = g.tile_units(t).iter().any(|&u| g.units[u.0].kind == cp_sim::model::UnitType::BasicWorker);
                match g.tiles[t.0].building.as_ref().map(|b| b.kind) {
                    Some(cp_sim::BuildingType::Farm) => { farms += 1; if staffed { farms_staffed += 1; } }
                    Some(cp_sim::BuildingType::Village) => villages += 1,
                    Some(cp_sim::BuildingType::Outpost) => outposts += 1,
                    Some(cp_sim::BuildingType::Mine) => mines += 1,
                    _ => { if g.tiles[t.0].tile_type == cp_sim::TileType::Forest && staffed { forests_worked += 1; } }
                }
            }
            println!("  r{:>3} tiles={:>2} money={:>4} wood={:>4} | W{} S{} salary={} | farms {}/{} forestW {} mines {} vil {} outp {}",
                g.get_rounds_played(), hard_tiles(&g), res(&g, BasicResource::Money), res(&g, BasicResource::Wood),
                workers, soldiers, salary, farms_staffed, farms, forests_worked, mines, villages, outposts);
        }
        if bankrupt_round < 0 && RES.iter().any(|&r| res(&g, r) < 0) { bankrupt_round = g.get_rounds_played(); }
        match g.end_turn() {
            EndTurnOutcome::Win(p) => { eliminated = p.0 == 0; break; }
            EndTurnOutcome::Tie => break,
            _ => {}
        }
    }
    (tiles_after_placement, bankrupt_round, g.get_rounds_played(), eliminated)
}

fn main() {
    let a = parse();
    rayon::ThreadPoolBuilder::new().num_threads(3).build_global().ok();
    let genome = Genome::from_file(&a.champion.to_string_lossy()).expect("load champion");
    let value = a.value.as_ref().map(|p| ValueNet::from_file(&p.to_string_lossy()).expect("load value"));

    if let Some(s) = a.trace {
        println!("=== TRACE seed {s} (neural champ seat0 vs HARD seat1) ===");
        let (t, b, r, e) = play(s, &genome, value.as_ref(), &a, true);
        println!("result: hard_tiles_after_placement={t} bankrupt_round={b} final_rounds={r} hard_eliminated={e}");
        return;
    }

    println!("scanning seeds 0..{} (14x12)…", a.scan);
    let (mut no_hq, mut early_death, mut sample) = (0u32, 0u32, Vec::<u32>::new());
    for s in 0..a.scan {
        let (tiles, _bank, r, e) = play(s, &genome, value.as_ref(), &a, false);
        if tiles == 0 { no_hq += 1; if no_hq <= 8 { println!("  NO-HQ    seed {s}: hard placed 0 tiles"); } }
        else if e && r <= 25 { early_death += 1; if sample.len() < 8 { sample.push(s); println!("  EARLYWIN seed {s}: champ 'won' at round {r} (hard died fast, ~self-destruct)"); } }
    }
    println!("\n{} / {} seeds ({:.1}%): HARD never placed an HQ (0 tiles → instant loss)", no_hq, a.scan, 100.0 * no_hq as f64 / a.scan as f64);
    println!("{} / {} seeds ({:.1}%): HARD eliminated by round 25 (likely self-bankrupt, not our doing)", early_death, a.scan, 100.0 * early_death as f64 / a.scan as f64);
}
