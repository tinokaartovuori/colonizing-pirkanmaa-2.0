//! TUNING HARNESS (transient, not parity-relevant) for the STRONG_ARMY yardstick.
//!
//! Constructs a candidate STRONG_ARMY `AiParams` from CLI flags and runs the key
//! fitness checks WITHOUT recompiling per param set:
//!   - h2h win-rate vs hard / rusher / device / fortress (both seat orders),
//!   - commit diagnostic: mean/median peak soldiers, front-open rate,
//!   - self-bankruptcy rate under real pressure (vs hard).
//!
//! All opponents use the SHIPPED constructors (HardAi::hard/rusher/device_rush/
//! fortress), so this measures the candidate against the real league. It never
//! mutates the shipped consts.
//!
//! Run e.g.:
//!   cargo run -p cp-train --bin sa_tune --release -- \
//!     --reserve 200 --strike 12 --assaults 10 --outposts 6 \
//!     --ready-soldiers 4 --ready-net 0 --seeds 60 --cap 200

use cp_ai::{AiParams, HardAi, HARD_PARAMS};
use cp_sim::resources::BasicResource;
use cp_sim::{EndTurnOutcome, Game, PlayerId, UnitType, WinCause};
use rayon::prelude::*;

const RES: [BasicResource; 4] = [
    BasicResource::Money,
    BasicResource::Wood,
    BasicResource::Stone,
    BasicResource::Metal,
];

#[derive(Clone, Copy)]
enum Opp {
    Hard,
    Rusher,
    Device,
    Fortress,
}
impl Opp {
    fn make(self) -> HardAi {
        match self {
            Opp::Hard => HardAi::hard(),
            Opp::Rusher => HardAi::rusher(),
            Opp::Device => HardAi::device_rush(),
            Opp::Fortress => HardAi::fortress(),
        }
    }
    fn name(self) -> &'static str {
        match self {
            Opp::Hard => "hard",
            Opp::Rusher => "rusher",
            Opp::Device => "device",
            Opp::Fortress => "fortress",
        }
    }
}

fn res(g: &Game, seat: usize, r: BasicResource) -> i64 {
    g.players[seat].resources.get(r).unwrap_or(0)
}

struct Out {
    sa_score: f64,    // from STRONG_ARMY's perspective
    sa_bankrupt: bool,
    peak_soldiers: i64,
    peak_cap: i64,
    peak_outposts: i64,
    peak_farms: i64,
    rounds: i64,
    opened_front: bool,
    reached8: bool,
}

/// Play candidate STRONG_ARMY (params `sap`) at `sa_seat` vs `opp`.
fn play(sap: AiParams, opp: Opp, sa_seat: usize, seed: u32, w: i32, h: i32, cap: i64) -> Out {
    let mut g = Game::new(w, h, &["P0", "P1"]);
    g.generate_map(w, h, seed);
    let mut sa = HardAi::new(sap);
    let mut other = opp.make();
    let placer = HardAi::hard();
    for _ in 0..2 {
        let cur = g.current_player();
        placer.place_headquarters(&mut g, cur);
        g.change_turn();
    }
    let sa_pid = PlayerId(sa_seat);
    let mut bankrupt = false;
    let mut peak = 0i64;
    let mut peak_cap = 0i64;
    let mut peak_outposts = 0i64;
    let mut peak_farms = 0i64;
    let mut opened = false;
    let mut natural: Option<(u8, Option<WinCause>)> = None;
    while g.live_players().len() > 1 && g.get_rounds_played() < cap {
        let cur = g.current_player();
        if cur.0 == sa_seat {
            sa.plan_turn(&mut g, cur);
        } else {
            other.plan_turn(&mut g, cur);
        }
        let s = g.current_soldier_amount(sa_pid);
        peak = peak.max(s);
        peak_cap = peak_cap.max(g.max_soldier_amount(sa_pid));
        let ops = g
            .owned_tiles(sa_pid)
            .iter()
            .filter(|&&t| g.tiles[t.0].building.as_ref().map(|b| b.kind) == Some(cp_sim::BuildingType::Outpost))
            .count() as i64;
        peak_outposts = peak_outposts.max(ops);
        let fms = g
            .owned_tiles(sa_pid)
            .iter()
            .filter(|&&t| g.tiles[t.0].building.as_ref().map(|b| b.kind) == Some(cp_sim::BuildingType::Farm))
            .count() as i64;
        peak_farms = peak_farms.max(fms);
        // front-open: a conquering SA soldier on an enemy-owned tile
        if !opened {
            for t in g.get_tiles() {
                let o = t.owner;
                if o.is_none() || o == Some(sa_pid) {
                    continue;
                }
                if g
                    .tile_conquering_units(t.id)
                    .iter()
                    .any(|&u| g.units[u.0].owner == Some(sa_pid) && g.units[u.0].kind == UnitType::Soldier)
                {
                    opened = true;
                    break;
                }
            }
        }
        if RES.iter().any(|&r| res(&g, sa_seat, r) < 0) {
            bankrupt = true;
        }
        match g.end_turn() {
            EndTurnOutcome::Win(p) => {
                natural = Some((p.0 as u8, g.last_win_cause()));
                break;
            }
            EndTurnOutcome::Tie => {
                natural = Some((2, None));
                break;
            }
            _ => {}
        }
    }
    if natural.is_none() {
        let live = g.live_players();
        if live.len() == 1 {
            natural = Some((live[0].0 as u8, g.last_win_cause()));
        }
    }
    let sa_score = match natural {
        Some((p, _)) if p as usize == sa_seat => 1.0,
        Some((2, _)) => 0.5,
        Some(_) => 0.0,
        None => {
            let t_sa = g.get_tile_count_for_player(sa_pid);
            let t_op = g.get_tile_count_for_player(PlayerId(1 - sa_seat));
            if t_sa > t_op {
                1.0
            } else if t_op > t_sa {
                0.0
            } else {
                0.5
            }
        }
    };
    Out {
        sa_score,
        sa_bankrupt: bankrupt,
        peak_soldiers: peak,
        peak_cap,
        peak_outposts,
        peak_farms,
        rounds: g.get_rounds_played(),
        opened_front: opened,
        reached8: peak >= 8,
    }
}

struct Agg {
    score: f64,
    games: u32,
    bankrupt: u32,
    peak_sum: i64,
    peaks: Vec<i64>,
    cap_sum: i64,
    outpost_sum: i64,
    farm_sum: i64,
    rounds_sum: i64,
    opened: u32,
    reached8: u32,
}

fn eval(sap: AiParams, opp: Opp, seeds: u32, w: i32, h: i32, cap: i64) -> Agg {
    let outs: Vec<Out> = (0..seeds)
        .into_par_iter()
        .flat_map(|s| {
            vec![
                play(sap, opp, 0, s, w, h, cap),
                play(sap, opp, 1, s ^ 0x5EED, w, h, cap),
            ]
        })
        .collect();
    let mut a = Agg {
        score: 0.0,
        games: 0,
        bankrupt: 0,
        peak_sum: 0,
        peaks: Vec::new(),
        cap_sum: 0,
        outpost_sum: 0,
        farm_sum: 0,
        rounds_sum: 0,
        opened: 0,
        reached8: 0,
    };
    for o in outs {
        a.score += o.sa_score;
        a.games += 1;
        if o.sa_bankrupt {
            a.bankrupt += 1;
        }
        a.peak_sum += o.peak_soldiers;
        a.peaks.push(o.peak_soldiers);
        a.cap_sum += o.peak_cap;
        a.outpost_sum += o.peak_outposts;
        a.farm_sum += o.peak_farms;
        a.rounds_sum += o.rounds;
        if o.opened_front {
            a.opened += 1;
        }
        if o.reached8 {
            a.reached8 += 1;
        }
    }
    a
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let argv = if let Some(p) = args.iter().position(|x| x == "--") {
        args[p + 1..].to_vec()
    } else {
        args[1..].to_vec()
    };
    // Start from HARD_PARAMS, override per flags.
    let mut p = HARD_PARAMS;
    let (mut seeds, mut cap, mut w, mut h, mut threads) = (60u32, 200i64, 14, 12, 12usize);
    let mut i = 0;
    while i < argv.len() {
        let k = argv[i].clone();
        macro_rules! v {
            () => {{
                i += 1;
                argv.get(i).cloned().unwrap_or_default()
            }};
        }
        match k.as_str() {
            "--reserve" => p.reserve = v!().parse().unwrap(),
            "--max-actions" => p.max_actions = v!().parse().unwrap(),
            "--garrison" => p.garrison = v!().parse().unwrap(),
            "--expand" => p.expand = v!().parse().unwrap(),
            "--outposts" => p.max_outposts = v!().parse().unwrap(),
            "--strike" => p.strike_force = v!().parse().unwrap(),
            "--assaults" => p.assaults_per_turn = v!().parse().unwrap(),
            "--ready-soldiers" => p.attack_ready_soldiers = v!().parse().unwrap(),
            "--ready-net" => p.econ_ready_net = v!().parse().unwrap(),
            "--nuclear" => p.nuclear = v!().parse::<i64>().unwrap() != 0,
            "--experts" => p.experts = v!().parse::<i64>().unwrap() != 0,
            "--warmonger" => p.warmonger = v!().parse::<i64>().unwrap() != 0,
            "--cut" => p.cut_priority = v!().parse::<i64>().unwrap() != 0,
            "--army-builder" => p.army_builder = v!().parse::<i64>().unwrap() != 0,
            "--device" => p.device = v!().parse::<i64>().unwrap() != 0,
            "--seeds" => seeds = v!().parse().unwrap(),
            "--cap" => cap = v!().parse().unwrap(),
            "--width" => w = v!().parse().unwrap(),
            "--height" => h = v!().parse().unwrap(),
            "--threads" => threads = v!().parse().unwrap(),
            _ => {}
        }
        i += 1;
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .ok();

    println!(
        "CANDIDATE STRONG_ARMY: reserve={} max_actions={} garrison={} expand={} outposts={} strike={} assaults={} ready_soldiers={} ready_net={} army_builder={} nuclear={} warmonger={} cut={} device={}",
        p.reserve, p.max_actions, p.garrison, p.expand, p.max_outposts, p.strike_force,
        p.assaults_per_turn, p.attack_ready_soldiers, p.econ_ready_net, p.army_builder,
        p.nuclear, p.warmonger, p.cut_priority, p.device
    );
    println!("seeds={} (x2 orders = {} games/pairing) cap={}\n", seeds, seeds * 2, cap);

    let opps = [Opp::Hard, Opp::Rusher, Opp::Device, Opp::Fortress];
    for opp in opps {
        let a = eval(p, opp, seeds, w, h, cap);
        let wr = a.score / a.games as f64 * 100.0;
        let mean_peak = a.peak_sum as f64 / a.games as f64;
        let mut peaks = a.peaks.clone();
        peaks.sort_unstable();
        let median = peaks[peaks.len() / 2];
        let opened = a.opened as f64 / a.games as f64 * 100.0;
        let _ = a.reached8;
        let reach6 = a.peaks.iter().filter(|&&p| p >= 6).count() as f64 / a.games as f64 * 100.0;
        let maxpeak = *a.peaks.iter().max().unwrap_or(&0);
        let bank = a.bankrupt as f64 / a.games as f64 * 100.0;
        let mean_cap = a.cap_sum as f64 / a.games as f64;
        let mean_ops = a.outpost_sum as f64 / a.games as f64;
        let mean_farms = a.farm_sum as f64 / a.games as f64;
        let mean_rounds = a.rounds_sum as f64 / a.games as f64;
        println!(
            "vs {:<9} win={:>5.1}%  peak(mean={:>4.1} med={:>2} max={:>2})  cap={:>4.1} ops={:>4.1} farms={:>4.1} rounds={:>5.0}  front-open={:>5.1}%  reach6={:>5.1}%  SA-bankrupt={:>4.1}%",
            opp.name(), wr, mean_peak, median, maxpeak, mean_cap, mean_ops, mean_farms, mean_rounds, opened, reach6, bank
        );
    }
}
