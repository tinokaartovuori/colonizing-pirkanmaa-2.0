//! ADVERSARIAL LEAGUE REVIEW (2026-06-06) — head-to-head + behavioral probes for the
//! 4 rebuilt scripted league bots (rusher / fortress / device / strong_army) + hard.
//!
//! The `league_health` no-op harness only proves SELF-SOLVENCY (no bankruptcy under
//! zero pressure). This binary proves STRENGTH/DEFENSE vs REAL opponents:
//!   1. a full win-rate matrix (rows beat columns), both seat orders, N seeds/pairing,
//!   2. per-archetype assertions (rusher homing, fortress wall, device teacher,
//!      strong_army yardstick) measured against real attackers/defenders.
//!
//! REVIEW-ONLY: reads `HardAi` constructors, never mutates AiParams. Off the parity
//! path. Run:
//!   cargo run -p cp-train --bin league_h2h --release -- --seeds 120 --cap 200

use cp_ai::HardAi;
use cp_sim::resources::BasicResource;
use cp_sim::{BuildingType, EndTurnOutcome, Game, PlayerId, TileType, UnitType, WinCause};
use rayon::prelude::*;

const RES: [BasicResource; 4] = [
    BasicResource::Money,
    BasicResource::Wood,
    BasicResource::Stone,
    BasicResource::Metal,
];

#[derive(Clone, Copy, PartialEq, Eq)]
enum Bot {
    Rusher,
    Fortress,
    Device,
    StrongArmy,
    Hard,
    Passive, // econ-only opponent (HARD with attack/military neutered-ish): use EXPERT preset
}

impl Bot {
    fn make(self) -> HardAi {
        match self {
            Bot::Rusher => HardAi::rusher(),
            Bot::Fortress => HardAi::fortress(),
            Bot::Device => HardAi::device_rush(),
            Bot::StrongArmy => HardAi::strong_army(),
            Bot::Hard => HardAi::hard(),
            Bot::Passive => HardAi::econ_expert(), // pure-econ, no offensive army
        }
    }
    fn name(self) -> &'static str {
        match self {
            Bot::Rusher => "rusher",
            Bot::Fortress => "fortress",
            Bot::Device => "device",
            Bot::StrongArmy => "strong_army",
            Bot::Hard => "hard",
            Bot::Passive => "passive",
        }
    }
}

fn res(g: &Game, seat: usize, r: BasicResource) -> i64 {
    g.players[seat].resources.get(r).unwrap_or(0)
}

/// Outcome of one game from the perspective of seat 0.
/// score: 1.0 = seat0 win, 0.0 = seat0 loss, 0.5 = true tie.
struct Out {
    score0: f64,
    rounds: i64,
    cause: Option<WinCause>,
    tie: bool,
    timeout: bool, // resolved by tile-majority tiebreak at the cap (no natural terminal)
    bankrupt0: bool,
    bankrupt1: bool,
}

/// Play `a` (seat0) vs `b` (seat1) on `seed`. Tile-majority tiebreak at the cap.
fn play(a: Bot, b: Bot, seed: u32, width: i32, height: i32, cap: i64) -> Out {
    let mut g = Game::new(width, height, &["P0", "P1"]);
    g.generate_map(width, height, seed);
    let mut p0 = a.make();
    let mut p1 = b.make();
    let placer = HardAi::hard();
    for _ in 0..2 {
        let cur = g.current_player();
        placer.place_headquarters(&mut g, cur);
        g.change_turn();
    }
    let mut bankrupt0 = false;
    let mut bankrupt1 = false;
    let mut natural: Option<(u8, Option<WinCause>)> = None;
    while g.live_players().len() > 1 && g.get_rounds_played() < cap {
        let cur = g.current_player();
        if cur.0 == 0 {
            p0.plan_turn(&mut g, cur);
        } else {
            p1.plan_turn(&mut g, cur);
        }
        if RES.iter().any(|&r| res(&g, 0, r) < 0) {
            bankrupt0 = true;
        }
        if RES.iter().any(|&r| res(&g, 1, r) < 0) {
            bankrupt1 = true;
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
    // Sole-survivor fallback.
    if natural.is_none() {
        let live = g.live_players();
        if live.len() == 1 {
            natural = Some((live[0].0 as u8, g.last_win_cause()));
        }
    }
    let (score0, cause, tie, timeout) = match natural {
        Some((0, c)) => (1.0, c, false, false),
        Some((1, c)) => (0.0, c, false, false),
        Some((2, _)) => (0.5, None, true, false),
        _ => {
            // Tile-majority tiebreak at the cap (a WIN, not a draw — per house rules).
            let t0 = g.get_tile_count_for_player(PlayerId(0));
            let t1 = g.get_tile_count_for_player(PlayerId(1));
            if t0 > t1 {
                (1.0, None, false, true)
            } else if t1 > t0 {
                (0.0, None, false, true)
            } else {
                (0.5, None, true, true)
            }
        }
    };
    Out {
        score0,
        rounds: g.get_rounds_played(),
        cause,
        tie,
        timeout,
        bankrupt0,
        bankrupt1,
    }
}

/// Aggregate a symmetric pairing (both seat orders) into A's win-rate vs B.
struct PairStat {
    a_wins: f64, // sum of A's score over all games (both orders), A treated as the row
    games: u32,
    ties: u32,
    timeouts: u32,
    a_bankrupt: u32,
    b_bankrupt: u32,
}

fn pairing(a: Bot, b: Bot, seeds: u32, width: i32, height: i32, cap: i64) -> PairStat {
    // Run both orders: A as seat0, then A as seat1 (B as seat0) — cancels first-move bias.
    let outs: Vec<(f64, bool, bool, bool, bool)> = (0..seeds)
        .into_par_iter()
        .flat_map(|s| {
            let o1 = play(a, b, s, width, height, cap); // A=seat0
            let o2 = play(b, a, s ^ 0x5EED, width, height, cap); // A=seat1
            // From A's perspective:
            let a_score_1 = o1.score0;
            let a_score_2 = 1.0 - o2.score0;
            vec![
                (a_score_1, o1.tie, o1.timeout, o1.bankrupt0, o1.bankrupt1),
                (a_score_2, o2.tie, o2.timeout, o2.bankrupt1, o2.bankrupt0),
            ]
        })
        .collect();
    let mut st = PairStat {
        a_wins: 0.0,
        games: 0,
        ties: 0,
        timeouts: 0,
        a_bankrupt: 0,
        b_bankrupt: 0,
    };
    for (score, tie, timeout, abk, bbk) in outs {
        st.a_wins += score;
        st.games += 1;
        if tie {
            st.ties += 1;
        }
        if timeout {
            st.timeouts += 1;
        }
        if abk {
            st.a_bankrupt += 1;
        }
        if bbk {
            st.b_bankrupt += 1;
        }
    }
    st
}

// ============================================================================
// Per-archetype behavioral probes (instrumented single games).
// ============================================================================

fn enemy_seat(seat: PlayerId) -> PlayerId {
    PlayerId(1 - seat.0)
}

/// Did `seat` stage a conquering soldier ON or ADJACENT-TO the enemy HQ this turn?
fn staged_on_enemy_hq(g: &Game, seat: PlayerId) -> bool {
    let enemy = enemy_seat(seat);
    let Some(hq) = g.get_hq_tile(enemy) else {
        return false;
    };
    // On the HQ tile itself (conquering units staged on it).
    let on_hq = g
        .tile_conquering_units(hq)
        .iter()
        .any(|&u| g.units[u.0].owner == Some(seat) && g.units[u.0].kind == UnitType::Soldier);
    if on_hq {
        return true;
    }
    // Adjacent: an owned soldier sitting on a tile next to the enemy HQ.
    for n in g.neighbour_tiles(hq) {
        if g.tiles[n.0].owner == Some(seat)
            && g
                .tile_units(n)
                .iter()
                .any(|&u| g.units[u.0].owner == Some(seat) && g.units[u.0].kind == UnitType::Soldier)
        {
            return true;
        }
    }
    false
}

/// RUSHER probe vs a passive opponent: does it reach + stage on/adjacent the enemy HQ?
/// Also tracks bridge builds + tile growth (river-crossing proxy).
struct RusherProbe {
    staged_on_hq: bool,
    bridges_built: i64,
    initial_tiles: i64,
    max_tiles: i64,
    fielded_soldier: bool,
    won: bool,
    won_by_conquest: bool,
}

fn probe_rusher(seed: u32, width: i32, height: i32, cap: i64) -> RusherProbe {
    let mut g = Game::new(width, height, &["P0", "P1"]);
    g.generate_map(width, height, seed);
    let rusher_seat = PlayerId(0);
    let mut rusher = HardAi::rusher();
    let mut passive = HardAi::econ_expert(); // pure-econ, no offensive
    let placer = HardAi::hard();
    for _ in 0..2 {
        let cur = g.current_player();
        placer.place_headquarters(&mut g, cur);
        g.change_turn();
    }
    let initial_tiles = g.get_tile_count_for_player(rusher_seat);
    let mut p = RusherProbe {
        staged_on_hq: false,
        bridges_built: 0,
        initial_tiles,
        max_tiles: initial_tiles,
        fielded_soldier: false,
        won: false,
        won_by_conquest: false,
    };
    while g.live_players().len() > 1 && g.get_rounds_played() < cap {
        let cur = g.current_player();
        if cur.0 == 0 {
            rusher.plan_turn(&mut g, cur);
        } else {
            passive.plan_turn(&mut g, cur);
        }
        if cur == rusher_seat {
            if staged_on_enemy_hq(&g, rusher_seat) {
                p.staged_on_hq = true;
            }
            if g.current_soldier_amount(rusher_seat) >= 1 {
                p.fielded_soldier = true;
            }
            let bridges = g
                .owned_tiles(rusher_seat)
                .iter()
                .filter(|&&t| g.tiles[t.0].building.as_ref().map(|b| b.kind) == Some(BuildingType::Bridge))
                .count() as i64;
            p.bridges_built = p.bridges_built.max(bridges);
            p.max_tiles = p.max_tiles.max(g.get_tile_count_for_player(rusher_seat));
        }
        match g.end_turn() {
            EndTurnOutcome::Win(w) => {
                if w == rusher_seat {
                    p.won = true;
                    if g.last_win_cause() == Some(WinCause::Conquest) {
                        p.won_by_conquest = true;
                    }
                }
                break;
            }
            EndTurnOutcome::Tie => break,
            _ => {}
        }
    }
    p
}

/// Count owned river tiles for a seat (to detect river-bounded spawns).
fn owned_rivers(g: &Game, seat: PlayerId) -> i64 {
    g.owned_tiles(seat)
        .iter()
        .filter(|&&t| g.tiles[t.0].tile_type == TileType::River)
        .count() as i64
}

/// Total river tiles on the map (geography indicator).
fn map_rivers(g: &Game) -> i64 {
    g.get_tiles()
        .iter()
        .filter(|t| t.tile_type == TileType::River)
        .count() as i64
}

/// FORTRESS probe: vs an attacker, did the fortress HQ fall? If so, how many soldiers
/// did the attacker have staged at the moment of the fall (min over the game)?
struct FortressProbe {
    hq_fell: bool,                  // fortress HQ no longer owned by fortress (any cause)
    hq_conquered: bool,            // HQ tile flipped to the ATTACKER (a real assault conquest)
    attacker_soldiers_at_fall: i64, // attacker's soldier count the turn the HQ flipped, -1
    fortress_outposts: i64,         // max outposts the fortress held
    fortress_won: bool,
    loss_cause: Option<WinCause>,  // why the fortress lost (if it lost)
    fortress_lost: bool,
}

fn probe_fortress(attacker: Bot, seed: u32, width: i32, height: i32, cap: i64) -> FortressProbe {
    let mut g = Game::new(width, height, &["P0", "P1"]);
    g.generate_map(width, height, seed);
    let fort_seat = PlayerId(0);
    let atk_seat = PlayerId(1);
    let mut fort = HardAi::fortress();
    let mut atk = attacker.make();
    let placer = HardAi::hard();
    for _ in 0..2 {
        let cur = g.current_player();
        placer.place_headquarters(&mut g, cur);
        g.change_turn();
    }
    let mut p = FortressProbe {
        hq_fell: false,
        hq_conquered: false,
        attacker_soldiers_at_fall: -1,
        fortress_outposts: 0,
        fortress_won: false,
        loss_cause: None,
        fortress_lost: false,
    };
    let fort_hq = g.get_hq_tile(fort_seat);
    while g.live_players().len() > 1 && g.get_rounds_played() < cap {
        let cur = g.current_player();
        if cur.0 == 0 {
            fort.plan_turn(&mut g, cur);
        } else {
            atk.plan_turn(&mut g, cur);
        }
        let ops = g
            .owned_tiles(fort_seat)
            .iter()
            .filter(|&&t| g.tiles[t.0].building.as_ref().map(|b| b.kind) == Some(BuildingType::Outpost))
            .count() as i64;
        p.fortress_outposts = p.fortress_outposts.max(ops);
        // ASSAULT FORCE measurement — conquest of the HQ resolves DURING `end_turn` (the
        // staged conquering units flip ownership then), so the meaningful "how big was the
        // cracking force" must be sampled JUST BEFORE end_turn: the attacker's soldiers
        // staged ON the fortress HQ tile plus those sitting ADJACENT (the units that can
        // resolve the conquest this turn). We snapshot it every half-turn and use the last
        // snapshot before the HQ flips. (The old probe sampled the global soldier count AT
        // TERMINAL, after resolution consumed the conquerors — undercounting to ~1.)
        let assault_force = if let Some(hq) = fort_hq {
            let on = g
                .tile_conquering_units(hq)
                .iter()
                .filter(|&&u| g.units[u.0].owner == Some(atk_seat) && g.units[u.0].kind == UnitType::Soldier)
                .count() as i64;
            let adj: i64 = g
                .neighbour_tiles(hq)
                .iter()
                .map(|&n| {
                    g.tile_units(n)
                        .iter()
                        .filter(|&&u| g.units[u.0].owner == Some(atk_seat) && g.units[u.0].kind == UnitType::Soldier)
                        .count() as i64
                })
                .sum();
            on + adj
        } else {
            0
        };
        if !p.hq_fell {
            if let Some(hq) = fort_hq {
                if g.tiles[hq.0].owner != Some(fort_seat) {
                    p.hq_fell = true;
                }
            }
        }
        let did_end = g.end_turn();
        // Detect conquest AFTER end_turn (resolution happens inside it).
        if !p.hq_conquered {
            if let Some(hq) = fort_hq {
                if g.tiles[hq.0].owner == Some(atk_seat) {
                    p.hq_conquered = true;
                    p.attacker_soldiers_at_fall = assault_force.max(1);
                }
            }
        }
        match did_end {
            EndTurnOutcome::Win(w) => {
                if w == fort_seat {
                    p.fortress_won = true;
                } else if w == atk_seat {
                    p.fortress_lost = true;
                    p.loss_cause = g.last_win_cause();
                }
                break;
            }
            EndTurnOutcome::Tie => break,
            _ => {}
        }
    }
    p
}

/// DEVICE probe: device bot (seat0) vs a real attacker (seat1).
struct DeviceProbe {
    built: bool,
    built_round: i64,
    device_won: bool,        // device completed countdown + won
    device_cracked: bool,    // attacker destroyed the standing device (it vanished while atk gained)
    crack_round: i64,        // round the device was cracked, -1
    attacker_won: bool,
    self_bankrupt: bool,
    mean_ring_fill: f64,
    ring_owned_rounds: i64,
}

fn probe_device(attacker: Bot, seed: u32, width: i32, height: i32, cap: i64) -> DeviceProbe {
    let mut g = Game::new(width, height, &["P0", "P1"]);
    g.generate_map(width, height, seed);
    let dev_seat = PlayerId(0);
    let atk_seat = PlayerId(1);
    let mut dev = HardAi::device_rush();
    let mut atk = attacker.make();
    let placer = HardAi::hard();
    for _ in 0..2 {
        let cur = g.current_player();
        placer.place_headquarters(&mut g, cur);
        g.change_turn();
    }
    let mut p = DeviceProbe {
        built: false,
        built_round: -1,
        device_won: false,
        device_cracked: false,
        crack_round: -1,
        attacker_won: false,
        self_bankrupt: false,
        mean_ring_fill: 0.0,
        ring_owned_rounds: 0,
    };
    let mut prev_owned_device = false;
    let mut ring_sum = 0i64;
    while g.live_players().len() > 1 && g.get_rounds_played() < cap {
        let cur = g.current_player();
        if cur.0 == 0 {
            dev.plan_turn(&mut g, cur);
        } else {
            atk.plan_turn(&mut g, cur);
        }
        let round = g.get_rounds_played();
        let owns_device = g.player_owns_strange_device(dev_seat);
        if owns_device {
            p.built = true;
            if p.built_round < 0 {
                p.built_round = round;
            }
            // ring fill
            if let Some(dt) = g.find_strange_device_tile() {
                let mut filled = 0i64;
                for n in g.neighbour_tiles(dt) {
                    if g.tiles[n.0].owner != Some(dev_seat) {
                        continue;
                    }
                    if g.tiles[n.0].building.as_ref().map(|b| b.kind) == Some(BuildingType::Outpost) {
                        continue;
                    }
                    if g
                        .tile_units(n)
                        .iter()
                        .any(|&u| g.units[u.0].owner == Some(dev_seat) && g.units[u.0].kind == UnitType::Soldier)
                    {
                        filled += 1;
                    }
                }
                ring_sum += filled;
                p.ring_owned_rounds += 1;
            }
        }
        // crack detection: device was owned last step, now no device on the board at all
        // AND the device seat did not win → attacker cracked it.
        if prev_owned_device && !g.has_strange_device() && p.crack_round < 0 {
            p.device_cracked = true;
            p.crack_round = round;
        }
        prev_owned_device = owns_device;

        if RES.iter().any(|&r| res(&g, 0, r) < 0) {
            p.self_bankrupt = true;
        }
        match g.end_turn() {
            EndTurnOutcome::Win(w) => {
                if w == dev_seat {
                    if g.last_win_cause() == Some(WinCause::Device) {
                        p.device_won = true;
                    }
                } else if w == atk_seat {
                    p.attacker_won = true;
                }
                break;
            }
            EndTurnOutcome::Tie => break,
            _ => {}
        }
    }
    if p.ring_owned_rounds > 0 {
        p.mean_ring_fill = ring_sum as f64 / p.ring_owned_rounds as f64;
    }
    p
}

/// STRONG_ARMY commit probe vs a real opponent: does it ever mass 8 soldiers and open
/// a front, or does it stall defensively?
struct CommitProbe {
    max_soldiers: i64,
    reached_gate: bool,    // hit >=8 soldiers at some point
    opened_front: bool,    // staged a conqueror on an enemy tile
    won: bool,
    rounds: i64,
    max_net: f64,          // peak net_money/round proxy via soldier+farm; approximate
}

fn probe_strong_commit(opp: Bot, seed: u32, width: i32, height: i32, cap: i64) -> CommitProbe {
    let mut g = Game::new(width, height, &["P0", "P1"]);
    g.generate_map(width, height, seed);
    let sa_seat = PlayerId(0);
    let mut sa = HardAi::strong_army();
    let mut other = opp.make();
    let placer = HardAi::hard();
    for _ in 0..2 {
        let cur = g.current_player();
        placer.place_headquarters(&mut g, cur);
        g.change_turn();
    }
    let mut p = CommitProbe {
        max_soldiers: 0,
        reached_gate: false,
        opened_front: false,
        won: false,
        rounds: 0,
        max_net: 0.0,
    };
    while g.live_players().len() > 1 && g.get_rounds_played() < cap {
        let cur = g.current_player();
        if cur.0 == 0 {
            sa.plan_turn(&mut g, cur);
        } else {
            other.plan_turn(&mut g, cur);
        }
        let s = g.current_soldier_amount(sa_seat);
        p.max_soldiers = p.max_soldiers.max(s);
        if s >= 8 {
            p.reached_gate = true;
        }
        // opened a front: a conquering soldier on an enemy-owned tile
        for t in g.get_tiles() {
            let o = t.owner;
            if o.is_none() || o == Some(sa_seat) {
                continue;
            }
            if g
                .tile_conquering_units(t.id)
                .iter()
                .any(|&u| g.units[u.0].owner == Some(sa_seat) && g.units[u.0].kind == UnitType::Soldier)
            {
                p.opened_front = true;
                break;
            }
        }
        match g.end_turn() {
            EndTurnOutcome::Win(w) => {
                if w == sa_seat {
                    p.won = true;
                }
                break;
            }
            EndTurnOutcome::Tie => break,
            _ => {}
        }
    }
    p.rounds = g.get_rounds_played();
    p
}

// ============================================================================

struct Args {
    seeds: u32,
    cap: i64,
    width: i32,
    height: i32,
    threads: usize,
}

fn parse() -> Args {
    let a: Vec<String> = std::env::args().collect();
    let args = if let Some(p) = a.iter().position(|x| x == "--") {
        a[p + 1..].to_vec()
    } else {
        a[1..].to_vec()
    };
    let (mut seeds, mut cap, mut width, mut height, mut threads) = (120u32, 200i64, 14, 12, 12usize);
    let mut i = 0;
    while i < args.len() {
        let k = args[i].clone();
        macro_rules! v {
            () => {{
                i += 1;
                args.get(i).cloned().unwrap_or_default()
            }};
        }
        match k.as_str() {
            "--seeds" => seeds = v!().parse().unwrap_or(120),
            "--cap" => cap = v!().parse().unwrap_or(200),
            "--width" => width = v!().parse().unwrap_or(14),
            "--height" => height = v!().parse().unwrap_or(12),
            "--threads" => threads = v!().parse().unwrap_or(12),
            _ => {}
        }
        i += 1;
    }
    Args { seeds, cap, width, height, threads }
}

fn main() {
    let a = parse();
    rayon::ThreadPoolBuilder::new()
        .num_threads(a.threads.max(1))
        .build_global()
        .ok();

    println!(
        "=== LEAGUE HEAD-TO-HEAD REVIEW — {} seeds/order (×2 orders = {} games/pairing), {}x{}, cap {} ===",
        a.seeds,
        a.seeds * 2,
        a.width,
        a.height,
        a.cap
    );

    // --- WIN MATRIX ---------------------------------------------------------
    let bots = [Bot::Rusher, Bot::Fortress, Bot::Device, Bot::StrongArmy, Bot::Hard];
    println!("\n--- WIN MATRIX (row win-rate vs column, both seat orders) ---");
    print!("{:<12}", "row\\col");
    for b in &bots {
        print!("{:>12}", b.name());
    }
    println!("{:>10}", "avg");

    // Precompute all pairings (i<j), store both directions.
    let n = bots.len();
    let mut wr = vec![vec![f64::NAN; n]; n];
    let mut tie_rate = vec![vec![0.0f64; n]; n];
    let mut timeout_rate = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            if !wr[i][j].is_nan() {
                continue;
            }
            let st = pairing(bots[i], bots[j], a.seeds, a.width, a.height, a.cap);
            let rate = st.a_wins / st.games as f64;
            wr[i][j] = rate;
            wr[j][i] = 1.0 - rate;
            let tr = st.ties as f64 / st.games as f64;
            tie_rate[i][j] = tr;
            tie_rate[j][i] = tr;
            let tor = st.timeouts as f64 / st.games as f64;
            timeout_rate[i][j] = tor;
            timeout_rate[j][i] = tor;
        }
    }
    for i in 0..n {
        print!("{:<12}", bots[i].name());
        let mut sum = 0.0;
        let mut cnt = 0;
        for j in 0..n {
            if i == j {
                print!("{:>12}", "—");
            } else {
                print!("{:>11.0}%", wr[i][j] * 100.0);
                sum += wr[i][j];
                cnt += 1;
            }
        }
        println!("{:>9.0}%", 100.0 * sum / cnt as f64);
    }

    println!("\n--- DRAW (true tie) RATE per pairing (%) ---");
    print!("{:<12}", "row\\col");
    for b in &bots {
        print!("{:>12}", b.name());
    }
    println!();
    for i in 0..n {
        print!("{:<12}", bots[i].name());
        for j in 0..n {
            if i == j {
                print!("{:>12}", "—");
            } else {
                print!("{:>11.1}%", tie_rate[i][j] * 100.0);
            }
        }
        println!();
    }

    println!("\n--- TIMEOUT (cap tiebreak, no natural terminal) RATE per pairing (%) ---");
    print!("{:<12}", "row\\col");
    for b in &bots {
        print!("{:>12}", b.name());
    }
    println!();
    for i in 0..n {
        print!("{:<12}", bots[i].name());
        for j in 0..n {
            if i == j {
                print!("{:>12}", "—");
            } else {
                print!("{:>11.1}%", timeout_rate[i][j] * 100.0);
            }
        }
        println!();
    }

    // --- ARCHETYPE 1: RUSHER (homing missile) -------------------------------
    println!("\n=== ASSERTION 1 — RUSHER (homing missile) vs passive (econ) ===");
    let rprobes: Vec<RusherProbe> = (0..a.seeds)
        .into_par_iter()
        .map(|s| probe_rusher(s, a.width, a.height, a.cap))
        .collect();
    let staged = rprobes.iter().filter(|p| p.staged_on_hq).count();
    let fielded = rprobes.iter().filter(|p| p.fielded_soldier).count();
    let won = rprobes.iter().filter(|p| p.won).count();
    let won_conq = rprobes.iter().filter(|p| p.won_by_conquest).count();
    // "reached the enemy": staged on HQ OR won by conquest (the HQ collapse implies reach)
    let reached = rprobes.iter().filter(|p| p.staged_on_hq || p.won_by_conquest).count();
    let reached_pct = 100.0 * reached as f64 / rprobes.len() as f64;
    let staged_pct = 100.0 * staged as f64 / rprobes.len() as f64;
    println!(
        "  WON vs passive: {}/{} ({:.0}%) | by conquest {}/{} ({:.0}%)",
        won,
        rprobes.len(),
        100.0 * won as f64 / rprobes.len() as f64,
        won_conq,
        rprobes.len(),
        100.0 * won_conq as f64 / rprobes.len() as f64
    );
    println!(
        "  reached enemy (staged-on-HQ OR won-by-conquest): {}/{} ({:.0}%)  [bar >= 80%]  {}",
        reached,
        rprobes.len(),
        reached_pct,
        if reached_pct >= 80.0 { "PASS" } else { "FAIL" }
    );
    // River subset: games where the rusher's spawn cluster owns a river OR map has rivers
    // AND it grew tiles. We can't force river-locked seeds, so report: among seeds where
    // the rusher built >=1 bridge, what fraction grew tiles past initial.
    let bridge_seeds: Vec<&RusherProbe> = rprobes.iter().filter(|p| p.bridges_built >= 1).collect();
    let crossed = bridge_seeds.iter().filter(|p| p.max_tiles > p.initial_tiles + 2).count();
    println!(
        "  fielded a soldier:           {:>3}/{} ({:.0}%)",
        fielded,
        rprobes.len(),
        100.0 * fielded as f64 / rprobes.len() as f64
    );
    println!(
        "  staged on/adjacent enemy HQ: {:>3}/{} ({:.0}%)   [bar >= 80%]  {}",
        staged,
        rprobes.len(),
        staged_pct,
        if staged_pct >= 80.0 { "PASS" } else { "FAIL" }
    );
    println!(
        "  built a bridge:              {:>3}/{} games",
        bridge_seeds.len(),
        rprobes.len()
    );
    if !bridge_seeds.is_empty() {
        let cross_pct = 100.0 * crossed as f64 / bridge_seeds.len() as f64;
        println!(
            "  of bridge-builders, crossed (tiles +>2): {}/{} ({:.0}%)  [bar >= 70%]  {}",
            crossed,
            bridge_seeds.len(),
            cross_pct,
            if cross_pct >= 70.0 { "PASS" } else { "FAIL" }
        );
    } else {
        println!("  (no bridge-build observed in these seeds — see river-geography note below)");
    }

    // --- ARCHETYPE 2: FORTRESS (turtle wall) --------------------------------
    println!("\n=== ASSERTION 2 — FORTRESS (turtle wall) vs attackers ===");
    for atk in [Bot::Rusher, Bot::StrongArmy, Bot::Hard] {
        let fprobes: Vec<FortressProbe> = (0..a.seeds)
            .into_par_iter()
            .map(|s| probe_fortress(atk, s, a.width, a.height, a.cap))
            .collect();
        let conq = fprobes.iter().filter(|p| p.hq_conquered).count();
        let conq_pct = 100.0 * conq as f64 / fprobes.len() as f64;
        let min_attacker_soldiers = fprobes
            .iter()
            .filter(|p| p.hq_conquered && p.attacker_soldiers_at_fall >= 0)
            .map(|p| p.attacker_soldiers_at_fall)
            .min();
        let weak_cracks = fprobes
            .iter()
            .filter(|p| p.hq_conquered && p.attacker_soldiers_at_fall >= 0 && p.attacker_soldiers_at_fall <= 2)
            .count();
        let mean_ops = fprobes.iter().map(|p| p.fortress_outposts).sum::<i64>() as f64
            / fprobes.len() as f64;
        let fort_wins = fprobes.iter().filter(|p| p.fortress_won).count();
        // loss cause breakdown
        let (mut l_conq, mut l_dom, mut l_bank, mut l_other) = (0, 0, 0, 0);
        for pr in &fprobes {
            if !pr.fortress_lost {
                continue;
            }
            match pr.loss_cause {
                Some(WinCause::Conquest) => l_conq += 1,
                Some(WinCause::Domination) => l_dom += 1,
                Some(WinCause::Bankruptcy) => l_bank += 1,
                _ => l_other += 1,
            }
        }
        println!(
            "  vs {:<11}: HQ-CONQUERED {:>3}/{} ({:.0}%) | min attacker soldiers at conquest {} | <=2-sol cracks {} | fort wins {}/{} | mean ops {:.1}",
            atk.name(),
            conq,
            fprobes.len(),
            conq_pct,
            min_attacker_soldiers
                .map(|m| m.to_string())
                .unwrap_or_else(|| "n/a".into()),
            weak_cracks,
            fort_wins,
            fprobes.len(),
            mean_ops,
        );
        println!(
            "                fortress losses by cause: conquest {} | domination {} | bankruptcy {} | other {}",
            l_conq, l_dom, l_bank, l_other
        );
    }
    println!("  [bar: a <=2-soldier attacker must NOT crack the HQ (weak-cracks should be 0; min-at-conquest >= 3)]");

    // --- ARCHETYPE 3: DEVICE-strategist -------------------------------------
    println!("\n=== ASSERTION 3 — DEVICE-strategist vs real attackers ===");
    for atk in [Bot::Rusher, Bot::StrongArmy] {
        let dprobes: Vec<DeviceProbe> = (0..a.seeds)
            .into_par_iter()
            .map(|s| probe_device(atk, s, a.width, a.height, a.cap))
            .collect();
        let games = dprobes.len();
        let built = dprobes.iter().filter(|p| p.built).count();
        let dev_won = dprobes.iter().filter(|p| p.device_won).count();
        let cracked = dprobes.iter().filter(|p| p.device_cracked).count();
        let atk_won = dprobes.iter().filter(|p| p.attacker_won).count();
        let self_bk = dprobes.iter().filter(|p| p.self_bankrupt).count();
        let mut crack_rounds: Vec<i64> = dprobes
            .iter()
            .filter(|p| p.device_cracked)
            .map(|p| p.crack_round)
            .collect();
        crack_rounds.sort_unstable();
        let median_crack = if crack_rounds.is_empty() {
            -1
        } else {
            crack_rounds[crack_rounds.len() / 2]
        };
        let mut built_rounds: Vec<i64> = dprobes
            .iter()
            .filter(|p| p.built_round >= 0)
            .map(|p| p.built_round)
            .collect();
        built_rounds.sort_unstable();
        let median_built = if built_rounds.is_empty() {
            -1
        } else {
            built_rounds[built_rounds.len() / 2]
        };
        let ring_mean = {
            let owners: Vec<&DeviceProbe> = dprobes.iter().filter(|p| p.ring_owned_rounds > 0).collect();
            if owners.is_empty() {
                0.0
            } else {
                owners.iter().map(|p| p.mean_ring_fill).sum::<f64>() / owners.len() as f64
            }
        };
        let pc = |x: usize| 100.0 * x as f64 / games as f64;
        println!("  device vs {}:", atk.name());
        println!(
            "    built {:.0}% (median r{}) | device-WON {:.0}% | attacker cracked/won {:.0}%/{:.0}% | median crack r{} | self-bankrupt {:.0}% | mean ring fill {:.2}",
            pc(built),
            median_built,
            pc(dev_won),
            pc(cracked),
            pc(atk_won),
            median_crack,
            pc(self_bk),
            ring_mean
        );
    }
    println!("  [teacher bar: NOT uncrackable (some attacker wins) AND NOT trivially cracked round ~1 of standing]");

    // --- ARCHETYPE 4: STRONG_ARMY (yardstick) -------------------------------
    println!("\n=== ASSERTION 4 — STRONG_ARMY (yardstick) — must dominate lesser bots ===");
    let sa_idx = bots.iter().position(|&b| b == Bot::StrongArmy).unwrap();
    let pairs = [
        (Bot::Rusher, 0.60),
        (Bot::Fortress, 0.75),
        (Bot::Device, 0.70),
        (Bot::Hard, 0.50),
    ];
    for (opp, bar) in pairs {
        let j = bots.iter().position(|&b| b == opp).unwrap();
        let rate = wr[sa_idx][j];
        let pass = rate >= bar - 1e-9;
        println!(
            "  strong_army vs {:<11}: {:.0}%  [bar >= {:.0}%]  {}",
            opp.name(),
            rate * 100.0,
            bar * 100.0,
            if pass { "PASS" } else { "FAIL" }
        );
    }
    // strong_army self-bankruptcy under real pressure (vs all opponents).
    let mut sa_games = 0u32;
    let mut sa_bank = 0u32;
    for opp in [Bot::Rusher, Bot::Fortress, Bot::Device, Bot::Hard] {
        let st = pairing(Bot::StrongArmy, opp, a.seeds / 2, a.width, a.height, a.cap);
        sa_games += st.games;
        sa_bank += st.a_bankrupt;
    }
    let sa_bk_pct = 100.0 * sa_bank as f64 / sa_games.max(1) as f64;
    println!(
        "  strong_army self-bankrupt under pressure: {:.1}% ({}/{})  [bar <= 5%]  {}",
        sa_bk_pct,
        sa_bank,
        sa_games,
        if sa_bk_pct <= 5.0 { "PASS" } else { "FAIL" }
    );
    // COMMIT diagnostic: does strong_army ever mass + attack, or stall?
    println!("  --- strong_army COMMIT diagnostic (does it reach the 8-soldier gate + open a front?) ---");
    for opp in [Bot::Rusher, Bot::Hard, Bot::Fortress] {
        let cps: Vec<CommitProbe> = (0..a.seeds)
            .into_par_iter()
            .map(|s| probe_strong_commit(opp, s, a.width, a.height, a.cap))
            .collect();
        let gate = cps.iter().filter(|p| p.reached_gate).count();
        let front = cps.iter().filter(|p| p.opened_front).count();
        let won = cps.iter().filter(|p| p.won).count();
        let mean_max = cps.iter().map(|p| p.max_soldiers).sum::<i64>() as f64 / cps.len() as f64;
        let pc = |x: usize| 100.0 * x as f64 / cps.len() as f64;
        println!(
            "    vs {:<11}: reached 8 soldiers {:.0}% | opened a front {:.0}% | won {:.0}% | mean peak soldiers {:.1}",
            opp.name(),
            pc(gate),
            pc(front),
            pc(won),
            mean_max
        );
    }

    // River geography note (descriptive).
    {
        let mut riverlocked = 0;
        let mut total = 0;
        for s in 0..a.seeds.min(200) {
            let mut g = Game::new(a.width, a.height, &["P0", "P1"]);
            g.generate_map(a.width, a.height, s);
            let placer = HardAi::hard();
            for _ in 0..2 {
                let cur = g.current_player();
                placer.place_headquarters(&mut g, cur);
                g.change_turn();
            }
            total += 1;
            if owned_rivers(&g, PlayerId(0)) >= 1 || map_rivers(&g) >= 8 {
                riverlocked += 1;
            }
        }
        println!(
            "\n  river geography: {}/{} sampled seeds have a river adjacent to spawn or a river-heavy map.",
            riverlocked, total
        );
    }

    println!("\n=== END REVIEW ===");
}
