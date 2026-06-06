//! Port of the TS `src/managers/*` DAL layer (`ObjectManager`,
//! `PlayerManager`, `GameSettingsManager`, `GameEventHandler`) collapsed into a
//! single owning [`Game`] arena.
//!
//! The TS spreads this state across four manager objects that all hold pointers
//! to each other and to the model graph. Because the model is index-based here
//! (see [`crate::model`]), the managers' methods need mutable access to the
//! shared arenas, so they live as methods on one `Game` struct. The original
//! method names are preserved (snake_cased) and grouped by their source
//! manager. Rendering / menu / mouse-follow calls become no-ops — the sim has
//! no renderer (the dependency-inversion seam in the TS exists precisely so this
//! is possible).

use crate::coordinate::Coordinate;
use crate::model::*;
use crate::resources::{self, BasicResource, ResourceMap};

// ---------------------------------------------------------------------------
// GameSettings (DAL/gamesettingsmanager + the grid-size maths)
// ---------------------------------------------------------------------------

/// Port of `GameSettingsManager`. For the headless sim the only values that
/// matter are the grid width/height (tiles), which equal the map width/height —
/// the pixel/menu sizing is irrelevant. We store the grid dimensions directly.
#[derive(Debug, Clone, Copy)]
pub struct GameSettings {
    pub grid_width: i32,
    pub grid_height: i32,
}

impl GameSettings {
    pub fn new(grid_width: i32, grid_height: i32) -> Self {
        GameSettings {
            grid_width,
            grid_height,
        }
    }
    pub fn map_grid_width(&self) -> i32 {
        self.grid_width
    }
    pub fn map_grid_height(&self) -> i32 {
        self.grid_height
    }
}

// ---------------------------------------------------------------------------
// SeatEvents — observation-only tactical counters (parity-safe telemetry)
// ---------------------------------------------------------------------------

/// Per-player cumulative tactical event counters. These are pure observation:
/// they are incremented at the exact points where an owner actually changes
/// inside `end_turn`, but they NEVER influence any branch, state transition,
/// owner assignment, or RNG call. They are not part of any state fingerprint,
/// so adding them cannot affect the parity gate. Attribution is always to the
/// player that gains the tile (the conquering/confiscating player).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SeatEvents {
    /// +1 when this player conquers a tile that is an enemy Headquarters.
    pub enemy_hqs_captured: i64,
    /// +1 when this player conquers an enemy tile that holds a real building
    /// (HQ is counted here too).
    pub enemy_buildings_captured: i64,
    /// +1 per enemy-owned tile this player takes by soldier conquest.
    pub enemy_tiles_conquered: i64,
    /// +1 per disconnected enemy tile CONFISCATED to this player during the
    /// HQ-connectivity cut step.
    pub tiles_gained_via_cut: i64,
    /// +N enemy soldiers this player DESTROYED in conquest resolution:
    /// on a successful assault, the defender's soldiers that get removed;
    /// on a successful DEFENCE (the attacker's failed assault), the attacker's
    /// soldiers that get removed. Attributed to the player who destroyed them.
    pub enemy_soldiers_killed: i64,
    /// +1 each time this player BUILDS a Strange Device (counts rebuilds after a
    /// destroy). Lets the harness measure build rate + builder win-rate.
    pub strange_devices_built: i64,
}

// ---------------------------------------------------------------------------
// Game — the unified arena + manager logic
// ---------------------------------------------------------------------------

/// The whole game world: arenas for tiles/units/players plus the turn-flow
/// state from `PlayerManager`/`GameEventHandler`.
///
/// `#[derive(Clone)]` makes the forward model branchable for test-time search
/// (`cp-ai::search`). The arena holds only `Clone` value types (no `Rc`/`RefCell`),
/// so a clone is an independent deep copy — parity-neutral (normal play never
/// clones).
#[derive(Clone)]
pub struct Game {
    // --- arenas ---
    /// `ObjectManager.tiles_` — also the canonical `getTiles()` order used by
    /// fingerprints. Worldgen pushes in column-major order.
    pub tiles: Vec<Tile>,
    /// All units ever created, indexed by `UnitId`. Dead units are left in place
    /// (their ids never dangle); liveness is tracked by tile/player membership.
    pub units: Vec<Unit>,
    /// All players in original (1-based) order; never reordered.
    pub players: Vec<Player>,

    pub settings: GameSettings,

    // --- PlayerManager state ---
    /// Index into the *live* player list `player_order` (the TS `playerIndex_`).
    player_index: usize,
    /// Live players, in turn order (the TS `players_`). Holds `PlayerId`s into
    /// `self.players`; shrinks as players are eliminated.
    player_order: Vec<PlayerId>,
    /// The TS `lostPlayers_` (kept for parity; order of elimination).
    lost_players: Vec<PlayerId>,
    /// `roundsPlayed_`, starts at -1, increments inside `change_turn` on wrap.
    rounds_played: i64,

    /// Observation-only per-player tactical event counters (see [`SeatEvents`]).
    /// Indexed by `PlayerId.0`, parallel to `self.players`. NOT part of any
    /// state fingerprint and never read by game logic — pure telemetry.
    seat_events: Vec<SeatEvents>,

    /// The cause of the most recent terminal elimination event (Device / Domination
    /// / Conquest / Bankruptcy). Telemetry only — set inside `end_turn`, read by
    /// benchmarks after a `Win` to tally the outcome breakdown (STRANGE-DEVICE-DESIGN
    /// §10). Never influences game state, RNG, or fingerprints.
    last_win_cause: Option<WinCause>,
}

impl Game {
    /// Construct an empty game (no tiles yet — call [`Game::generate_map`]).
    /// `player_specs` are `(name, ...)`; player numbers are assigned 1..=n in
    /// order, matching `PlayerManager`'s constructor.
    pub fn new(grid_width: i32, grid_height: i32, player_names: &[&str]) -> Self {
        let settings = GameSettings::new(grid_width, grid_height);
        let mut players = Vec::with_capacity(player_names.len());
        let mut player_order = Vec::with_capacity(player_names.len());
        for (i, name) in player_names.iter().enumerate() {
            let id = PlayerId(i);
            players.push(Player::new(id, (i + 1) as i64, (*name).to_string()));
            player_order.push(id);
        }
        let n = players.len();
        Game {
            tiles: Vec::new(),
            units: Vec::new(),
            players,
            settings,
            player_index: 0,
            player_order,
            lost_players: Vec::new(),
            rounds_played: -1,
            seat_events: vec![SeatEvents::default(); n],
            last_win_cause: None,
        }
    }

    /// The cause of the most recent terminal elimination (telemetry; see
    /// [`WinCause`]). Read by benchmarks after a [`EndTurnOutcome::Win`].
    pub fn last_win_cause(&self) -> Option<WinCause> {
        self.last_win_cause
    }

    /// Read-only accessor for a player's cumulative tactical event counters.
    pub fn seat_events(&self, player: PlayerId) -> SeatEvents {
        self.seat_events[player.0]
    }

    // =======================================================================
    // PlayerManager
    // =======================================================================

    /// `getCurrentPlayer()`.
    pub fn current_player(&self) -> PlayerId {
        self.player_order[self.player_index]
    }

    /// `getPlayers()` — the live players, in turn order.
    pub fn live_players(&self) -> &[PlayerId] {
        &self.player_order
    }

    pub fn get_rounds_played(&self) -> i64 {
        self.rounds_played
    }

    /// `changeTurn()`.
    pub fn change_turn(&mut self) {
        self.player_index += 1;
        if self.player_index >= self.player_order.len() {
            self.player_index = 0;
        }
        if self.player_index == 0 {
            self.rounds_played += 1;
        }
    }

    /// `setPlayerAsLost(lostPlayer, currentPlayer?)`.
    fn set_player_as_lost(&mut self, lost: PlayerId, current: Option<PlayerId>) {
        self.lost_players.push(lost);
        self.players[lost.0].alive = false;

        if let Some(cur) = current {
            let lost_num = self.players[lost.0].player_num;
            let cur_num = self.players[cur.0].player_num;
            if lost_num < cur_num {
                // playerIndex_-- (can wrap negative in JS; saturate at 0 here,
                // but the wrap is guarded by the splice below keeping indices
                // valid — see note). Use wrapping via isize.
                if self.player_index == 0 {
                    // JS would make it -1; the next changeTurn ++ brings it to 0.
                    // We replicate by leaving it at 0 only if the removed element
                    // was before index 0 which is impossible, so this branch is
                    // unreachable in practice.
                    self.player_index = 0;
                } else {
                    self.player_index -= 1;
                }
            }
        }

        if let Some(i) = self.player_order.iter().position(|&p| p == lost) {
            self.player_order.remove(i);
        }
    }

    // =======================================================================
    // ObjectManager
    // =======================================================================

    /// `getTiles()`.
    pub fn get_tiles(&self) -> &[Tile] {
        &self.tiles
    }

    /// `getTile(coord)` — linear scan, matching the TS (and preserving its O(n)
    /// "first match wins" semantics; coordinates are unique anyway).
    pub fn get_tile_at(&self, coord: Coordinate) -> Option<TileId> {
        self.tiles
            .iter()
            .find(|t| t.x == coord.x && t.y == coord.y)
            .map(|t| t.id)
    }

    pub fn get_tile_count(&self) -> i64 {
        self.tiles.len() as i64
    }

    pub fn get_tile_count_for_player(&self, player: PlayerId) -> i64 {
        self.tiles.iter().filter(|t| t.owner == Some(player)).count() as i64
    }

    pub fn get_neutral_tiles(&self) -> i64 {
        self.tiles.iter().filter(|t| t.owner.is_none()).count() as i64
    }

    /// `ObjectManager.findStrangeDeviceTile()` — the tile carrying the (at most one)
    /// standing Strange Device, scanned in canonical tile order.
    pub fn find_strange_device_tile(&self) -> Option<TileId> {
        self.tiles
            .iter()
            .find(|t| matches!(&t.building, Some(b) if b.kind == BuildingType::StrangeDevice))
            .map(|t| t.id)
    }

    /// `ObjectManager.hasStrangeDevice()`.
    pub fn has_strange_device(&self) -> bool {
        self.find_strange_device_tile().is_some()
    }

    /// `PlayerBase.ownsStrangeDevice()` — true while `player` owns the tile a standing
    /// Device sits on. (A captured device-tile no longer counts for the original
    /// builder, matching the TS, where the building's owner never changes but the
    /// tile's does — see the destroy-detection in end_turn.)
    pub fn player_owns_strange_device(&self, player: PlayerId) -> bool {
        match self.find_strange_device_tile() {
            Some(tid) => self.tiles[tid.0].owner == Some(player),
            None => false,
        }
    }

    /// `getHqTile(player)` — the player's un-conquered Headquarters tile.
    pub fn get_hq_tile(&self, player: PlayerId) -> Option<TileId> {
        for obj in &self.players[player.0].objects {
            if let ObjId::Tile(tid) = obj {
                let tile = &self.tiles[tid.0];
                if let Some(b) = &tile.building {
                    if b.kind == BuildingType::Headquarters && !b.conquered {
                        return Some(*tid);
                    }
                }
            }
        }
        None
    }

    /// `getHqConnectedTiles(player)` — BFS over orthogonally-adjacent owned tiles
    /// starting from the HQ. Order of the returned list matches the TS (it grows
    /// the `tiles` vec and re-scans, with `neighbouringFour` order).
    pub fn get_hq_connected_tiles(&self, player: PlayerId) -> Vec<TileId> {
        let mut tiles: Vec<TileId> = Vec::new();
        let hq = match self.get_hq_tile(player) {
            Some(h) => h,
            None => return tiles,
        };
        tiles.push(hq);
        let mut i = 0;
        while i < tiles.len() {
            let neighbours = self.neighbour_four_tiles(tiles[i]);
            for n in neighbours {
                if tiles.contains(&n) {
                    continue;
                }
                if self.tiles[n.0].owner == Some(player) {
                    tiles.push(n);
                }
            }
            i += 1;
        }
        tiles
    }

    /// `getNeighbourFourTiles()` for a tile (orthogonal, in-grid, S/W/N/E order).
    pub fn neighbour_four_tiles(&self, tid: TileId) -> Vec<TileId> {
        let t = &self.tiles[tid.0];
        let coord = Coordinate::new(t.x, t.y);
        let mut out = Vec::new();
        for nc in coord.neighbouring_four(self.settings.grid_width, self.settings.grid_height) {
            if let Some(n) = self.get_tile_at(nc) {
                out.push(n);
            }
        }
        out
    }

    /// `getNeighbourTiles()` — the 8-neighbourhood (Chebyshev radius 1).
    pub fn neighbour_tiles(&self, tid: TileId) -> Vec<TileId> {
        let t = &self.tiles[tid.0];
        let coord = Coordinate::new(t.x, t.y);
        let mut out = Vec::new();
        for nc in coord.neighbours(1, self.settings.grid_width, self.settings.grid_height) {
            if let Some(n) = self.get_tile_at(nc) {
                out.push(n);
            }
        }
        out
    }

    /// `hasOpponentHeadquarters(player)` (the Grassland override; base is always
    /// true). False only for the player's *own* un-conquered HQ tile.
    /// Made `pub` (Plan-B) so the candidate generator's `build_bridge`
    /// `bridge_unblock_count` feature can use the same gate as
    /// `get_available_tiles_for` without duplicating logic.
    pub fn has_opponent_headquarters(&self, tid: TileId, player: PlayerId) -> bool {
        let tile = &self.tiles[tid.0];
        if tile.tile_type == TileType::Grassland {
            if let Some(b) = &tile.building {
                if b.kind == BuildingType::Headquarters
                    && tile.owner == Some(player)
                    && !b.conquered
                {
                    return false;
                }
            }
        }
        true
    }

    /// `getAvailableTiles()` for the current player — owned tiles + their
    /// orthogonal neighbours that aren't the current player's own HQ. River
    /// tiles without a building don't expand (the TS `continue`). Order matches
    /// the TS exactly.
    pub fn get_available_tiles(&self) -> Vec<TileId> {
        self.get_available_tiles_for(self.current_player())
    }

    /// `getAvailableTiles()` computed for an ARBITRARY `player` (not just the
    /// current one). Identical logic to [`get_available_tiles`] — owned tiles +
    /// their orthogonal-4 neighbours passing `has_opponent_headquarters`, with the
    /// unbridged-river expansion block and the own-un-conquered-HQ exclusion. This
    /// is a READ-ONLY query helper used by the AZ planes extractor to compute each
    /// enemy's staging frontier; it changes no game rule, cost, or candidate gate
    /// and is never invoked on the parity path.
    pub fn get_available_tiles_for(&self, player: PlayerId) -> Vec<TileId> {
        let mut available: Vec<TileId> = Vec::new();
        let objs: Vec<ObjId> = self.players[player.0].objects.clone();
        for obj in objs {
            if let ObjId::Tile(tid) = obj {
                let tile = &self.tiles[tid.0];
                if tile.owner == Some(player) && self.has_opponent_headquarters(tid, player) {
                    if !available.contains(&tid) {
                        available.push(tid);
                    }
                }
                // River with no building does not expand availability.
                if tile.tile_type == TileType::River && tile.building.is_none() {
                    continue;
                } else {
                    for n in self.neighbour_four_tiles(tid) {
                        if available.contains(&n) {
                            continue;
                        }
                        if self.has_opponent_headquarters(n, player) {
                            available.push(n);
                        }
                    }
                }
            }
        }
        available
    }

    /// `replaceTile(oldTile, newTile)` is only triggered by a forest being built
    /// on (the renderer-driven `updateForest('Grassland', …)` path). In the
    /// headless sim a building on a forest is rejected up front (see
    /// `add_building`), so this is intentionally not ported.

    // =======================================================================
    // setOwner (GameObject.setOwner specialised for tiles)
    // =======================================================================

    /// Port of `GameObject.setOwner` for a *tile*: detaches from the old owner's
    /// `objects_`, attaches to the new owner's. (Buildings' owner is tracked
    /// separately and not registered in `objects_`.)
    pub fn tile_set_owner(&mut self, tid: TileId, owner: Option<PlayerId>) {
        let current = self.tiles[tid.0].owner;
        let obj = ObjId::Tile(tid);
        if let Some(cur) = current {
            if Some(cur) != owner {
                self.players[cur.0].remove_object(obj);
            }
        }
        if let Some(new) = owner {
            if !self.players[new.0].has_object(obj) {
                self.players[new.0].add_object(obj);
            }
        }
        self.tiles[tid.0].owner = owner;
    }

    /// Port of `GameObject.setOwner` for a *unit*.
    fn unit_set_owner(&mut self, uid: UnitId, owner: Option<PlayerId>) {
        let current = self.units[uid.0].owner;
        let obj = ObjId::Unit(uid);
        if let Some(cur) = current {
            if Some(cur) != owner {
                self.players[cur.0].remove_object(obj);
            }
        }
        if let Some(new) = owner {
            if !self.players[new.0].has_object(obj) {
                self.players[new.0].add_object(obj);
            }
        }
        self.units[uid.0].owner = owner;
    }

    // =======================================================================
    // Tile unit add/remove + conquest
    // =======================================================================

    /// `TileBase.addUnit` (+ the River override, which only changes sprites — so
    /// the logic is identical here). Honours the conquering distinction and the
    /// MAX_UNITS room check. Panics on overflow, matching the TS `throw`.
    pub fn tile_add_unit(&mut self, tid: TileId, uid: UnitId) {
        let is_conq = self.units[uid.0].is_conquering;
        let max = self.tiles[tid.0].max_units;
        if !is_conq {
            if self.tiles[tid.0].unit_count() as i64 + 1 > max {
                panic!("Tile has no more room for Units!");
            }
            self.units[uid.0].location = Some(tid);
            self.tiles[tid.0].units.push(uid);
        } else {
            if self.tiles[tid.0].conquering_unit_count() as i64 + 1 > max {
                panic!("Tile has no more room for conquering units!");
            }
            self.units[uid.0].location = Some(tid);
            self.tiles[tid.0].conquering_units.push(uid);
        }
    }

    /// `TileBase.removeUnit` — drop from whichever list holds it.
    pub fn tile_remove_unit(&mut self, tid: TileId, uid: UnitId) {
        let t = &mut self.tiles[tid.0];
        if let Some(i) = t.units.iter().position(|&u| u == uid) {
            t.units.remove(i);
        }
        if let Some(i) = t.conquering_units.iter().position(|&u| u == uid) {
            t.conquering_units.remove(i);
        }
    }

    /// `getSoldierCount()` — soldiers among *conquering* units (the attackers).
    fn tile_soldier_count(&self, tid: TileId) -> i64 {
        self.tiles[tid.0]
            .conquering_units
            .iter()
            .filter(|&&u| self.units[u.0].kind == UnitType::Soldier)
            .count() as i64
    }
    /// `getOpponentSoldierCount()` — soldiers among *owned* units (defenders).
    fn tile_opponent_soldier_count(&self, tid: TileId) -> i64 {
        self.tiles[tid.0]
            .units
            .iter()
            .filter(|&&u| self.units[u.0].kind == UnitType::Soldier)
            .count() as i64
    }

    /// `TileBase.conquerTile(currentPlayer)`. Faithful port including the strict
    /// `>` soldier comparison, the outpost +3 defence (modelled as "outpost can
    /// never be taken by assault"), and HQ conquered-flag flip.
    pub fn conquer_tile(&mut self, tid: TileId, current: PlayerId) {
        // 1. Claim an unowned tile that has this player's conquering units.
        {
            let conq = self.tiles[tid.0].conquering_units.clone();
            let owner = self.tiles[tid.0].owner;
            let mut claimed = false;
            for uid in &conq {
                if self.units[uid.0].owner == Some(current) && owner.is_none() {
                    claimed = true;
                    break;
                }
            }
            if claimed {
                self.tile_set_owner(tid, Some(current));
                // setAsConquering(false) for all, move them into owned units.
                let conq = std::mem::take(&mut self.tiles[tid.0].conquering_units);
                for uid in conq {
                    self.units[uid.0].is_conquering = false;
                    self.tiles[tid.0].units.push(uid);
                }
            }
        }

        // 2. Assault an enemy-owned tile.
        let owner = self.tiles[tid.0].owner;
        if owner != Some(current) && owner.is_some() {
            let own_soldiers = self.tile_soldier_count(tid);
            let opp_soldiers = self.tile_opponent_soldier_count(tid);
            let has_outpost = matches!(
                &self.tiles[tid.0].building,
                Some(b) if b.kind == BuildingType::Outpost
            );

            if own_soldiers > opp_soldiers && !has_outpost {
                // Observation-only: classify the tile being taken BEFORE the
                // owner change, then attribute to `current`. This reads only and
                // does not affect any branch, state, owner, or RNG call.
                {
                    let bld = self.tiles[tid.0].building.as_ref();
                    let is_hq = matches!(bld, Some(b) if b.kind == BuildingType::Headquarters);
                    let has_building = bld.is_some();
                    let ev = &mut self.seat_events[current.0];
                    ev.enemy_tiles_conquered += 1;
                    if is_hq {
                        ev.enemy_hqs_captured += 1;
                    }
                    if has_building {
                        ev.enemy_buildings_captured += 1;
                    }
                    // The defending soldiers (counted before removal) are about
                    // to be destroyed below; credit the attacker. Read-only.
                    ev.enemy_soldiers_killed += opp_soldiers;
                }
                self.tile_set_owner(tid, Some(current));
                // Conquered HQ flips its flag.
                if let Some(b) = &mut self.tiles[tid.0].building {
                    if b.kind == BuildingType::Headquarters {
                        b.conquered = true;
                    }
                }
                // Delete defending (owned) units, then absorb attackers.
                let defenders = self.tiles[tid.0].units.clone();
                for uid in defenders {
                    self.delete_unit_from_tile(uid, tid);
                }
                let attackers = std::mem::take(&mut self.tiles[tid.0].conquering_units);
                for uid in attackers {
                    self.units[uid.0].is_conquering = false;
                    self.tiles[tid.0].units.push(uid);
                }
            } else {
                // Failed assault: attackers are destroyed.
                // Observation-only: the defender (the tile owner) successfully
                // defended; credit it with the attacker soldiers destroyed.
                // `own_soldiers` was counted above; reading it changes nothing.
                if let Some(defender) = owner {
                    self.seat_events[defender.0].enemy_soldiers_killed += own_soldiers;
                }
                let attackers = self.tiles[tid.0].conquering_units.clone();
                for uid in attackers {
                    self.delete_unit_from_tile(uid, tid);
                }
            }
        }
    }

    // =======================================================================
    // Tile max-unit / max-soldier contributions
    // =======================================================================

    /// `getMaxUnitsIncrease()` — HQ(+3, unless conquered)/Village(+3)/Mikontalo(+2).
    fn tile_max_units_increase(&self, tid: TileId) -> i64 {
        if let Some(b) = &self.tiles[tid.0].building {
            match b.kind {
                BuildingType::Headquarters if !b.conquered => return resources::HQ_UNIT_VALUE,
                BuildingType::Village => return resources::VILLAGE_UNIT_VALUE,
                BuildingType::Mikontalo => return resources::MIKONTALO_UNIT_VALUE,
                _ => {}
            }
        }
        0
    }

    /// `getMaxSoldiersIncrease()` — HQ(+1, unless conquered)/Outpost(+3).
    fn tile_max_soldiers_increase(&self, tid: TileId) -> i64 {
        if let Some(b) = &self.tiles[tid.0].building {
            match b.kind {
                BuildingType::Headquarters if !b.conquered => return resources::HQ_SOLDIER_VALUE,
                BuildingType::Outpost => return resources::OUTPOST_SOLDIER_VALUE,
                _ => {}
            }
        }
        0
    }

    // =======================================================================
    // Player unit-cap bookkeeping (PlayerBase)
    // =======================================================================

    /// `updateUnitAmounts()` — recompute max unit/soldier caps from owned tiles.
    pub fn update_unit_amounts(&mut self, player: PlayerId) {
        let mut max_unit = 0i64;
        let mut max_soldier = 0i64;
        let objs: Vec<ObjId> = self.players[player.0].objects.clone();
        for obj in objs {
            if let ObjId::Tile(tid) = obj {
                max_unit += self.tile_max_units_increase(tid);
                max_soldier += self.tile_max_soldiers_increase(tid);
            }
        }
        if max_unit >= resources::UNIT_LIMITS {
            max_unit = resources::UNIT_LIMITS;
        }
        if max_soldier >= resources::UNIT_LIMITS {
            max_soldier = resources::UNIT_LIMITS;
        }
        // Owning a standing Strange Device halves the soldier cap (floored): the cost
        // of racing the Device's countdown is being left defensively exposed. Forced
        // disband of now-excess soldiers happens on build (ai_build_building) and at
        // every end_turn via eliminate_excess_units. Conditional — fires only when a
        // device exists, so device-free games keep the original cap byte-for-byte.
        if self.player_owns_strange_device(player) {
            max_soldier /= 2; // i64 division floors for non-negatives == Math.floor
        }
        self.players[player.0].max_unit_amount = max_unit;
        self.players[player.0].max_soldier_amount = max_soldier;
    }

    fn count_units(&self, player: PlayerId, kind: UnitType) -> i64 {
        self.players[player.0]
            .objects
            .iter()
            .filter(|o| matches!(o, ObjId::Unit(u) if self.units[u.0].kind == kind))
            .count() as i64
    }

    pub fn current_basic_worker_amount(&self, player: PlayerId) -> i64 {
        self.count_units(player, UnitType::BasicWorker)
    }
    pub fn current_expert_amount(&self, player: PlayerId) -> i64 {
        self.count_units(player, UnitType::Expert)
    }
    pub fn current_soldier_amount(&self, player: PlayerId) -> i64 {
        self.count_units(player, UnitType::Soldier)
    }

    /// `getFreeUnitAmount()` (uses the cached cap, like the TS).
    pub fn free_unit_amount(&self, player: PlayerId) -> i64 {
        self.players[player.0].max_unit_amount
            - self.current_basic_worker_amount(player)
            - self.current_expert_amount(player)
    }
    /// `getFreeSoldierAmount()`.
    pub fn free_soldier_amount(&self, player: PlayerId) -> i64 {
        self.players[player.0].max_soldier_amount - self.current_soldier_amount(player)
    }

    /// `getMaxUnitAmount()` — refreshes the cap first (TS does too).
    pub fn max_unit_amount(&mut self, player: PlayerId) -> i64 {
        self.update_unit_amounts(player);
        self.players[player.0].max_unit_amount
    }
    pub fn max_soldier_amount(&mut self, player: PlayerId) -> i64 {
        self.update_unit_amounts(player);
        self.players[player.0].max_soldier_amount
    }

    /// `eliminateExcessUnits()` — cull workers/experts then soldiers until under
    /// cap. Removing a unit destroys it (drops from tile + owner).
    pub fn eliminate_excess_units(&mut self, player: PlayerId) {
        self.update_unit_amounts(player);

        // Returns true if it removed one matching unit.
        let remove_one = |game: &mut Game, want_soldier: bool| -> bool {
            let objs: Vec<ObjId> = game.players[player.0].objects.clone();
            for obj in objs {
                if let ObjId::Unit(uid) = obj {
                    let kind = game.units[uid.0].kind;
                    let matches = if want_soldier {
                        kind == UnitType::Soldier
                    } else {
                        kind == UnitType::BasicWorker || kind == UnitType::Expert
                    };
                    if matches {
                        if let Some(tid) = game.units[uid.0].location {
                            game.tile_remove_unit(tid, uid);
                        }
                        game.players[player.0].remove_object(obj);
                        return true;
                    }
                }
            }
            false
        };

        while self.free_unit_amount(player) < 0 {
            if !remove_one(self, false) {
                break;
            }
            self.update_unit_amounts(player);
        }
        while self.free_soldier_amount(player) < 0 {
            if !remove_one(self, true) {
                break;
            }
            self.update_unit_amounts(player);
        }
    }

    // =======================================================================
    // Resource generation (per-tile generateResources, dispatched by terrain)
    // =======================================================================

    /// Add `delta` to a tile's owner, if any (`owner?.addOrRemoveResources`).
    fn owner_add_resources(&mut self, owner: Option<PlayerId>, delta: &ResourceMap) {
        if let Some(p) = owner {
            self.players[p.0].add_or_remove_resources(delta);
        }
    }

    fn tile_has_unit_type(&self, tid: TileId, kind: UnitType) -> bool {
        self.tiles[tid.0]
            .units
            .iter()
            .any(|&u| self.units[u.0].kind == kind)
    }

    /// `generateResources()` for one tile. Dispatches on terrain + building.
    pub fn generate_resources(&mut self, tid: TileId) {
        let tile_type = self.tiles[tid.0].tile_type;
        let owner = self.tiles[tid.0].owner;
        match tile_type {
            TileType::Grassland => self.gen_grassland(tid, owner),
            TileType::Forest => self.gen_forest(tid, owner),
            TileType::AbundantForest => self.gen_abundant_forest(tid, owner),
            TileType::Mountain => self.gen_mountain(tid, owner),
            TileType::River => self.gen_river(tid, owner),
        }
    }

    fn gen_grassland(&mut self, tid: TileId, owner: Option<PlayerId>) {
        let kind = match &self.tiles[tid.0].building {
            Some(b) => b.kind,
            None => return,
        };
        match kind {
            BuildingType::Farm => {
                let has_worker = self.tile_has_unit_type(tid, UnitType::BasicWorker);
                let growth_phase = self.tiles[tid.0].building.as_ref().unwrap().growth_phase + 1;
                self.tiles[tid.0]
                    .building
                    .as_mut()
                    .unwrap()
                    .set_growth_phase(growth_phase);
                if growth_phase == 5 && has_worker {
                    let prod = self.tiles[tid.0].building.as_ref().unwrap().production();
                    self.owner_add_resources(owner, &prod);
                    self.tiles[tid.0].building.as_mut().unwrap().reset_farm();
                } else if !has_worker {
                    self.tiles[tid.0].building.as_mut().unwrap().reset_farm();
                }
                // else (has_worker, not phase 5): just an animation tick — no-op.
            }
            BuildingType::Nuclear => {
                if !self.tile_has_unit_type(tid, UnitType::Expert) {
                    return;
                }
                let prod = self.tiles[tid.0].building.as_ref().unwrap().production();
                let worker_uids: Vec<UnitId> = self.tiles[tid.0].units.clone();
                for uid in worker_uids {
                    if self.units[uid.0].kind == UnitType::BasicWorker {
                        self.owner_add_resources(owner, &prod);
                    }
                }
            }
            BuildingType::Village | BuildingType::Outpost => {
                let prod = self.tiles[tid.0].building.as_ref().unwrap().production();
                self.owner_add_resources(owner, &prod);
            }
            _ => {}
        }
    }

    fn gen_forest(&mut self, tid: TileId, owner: Option<PlayerId>) {
        let forest_prod = resources::forest_production();
        let wood_per = forest_prod.get(BasicResource::Wood).unwrap();
        let units = self.tiles[tid.0].units.clone();
        for uid in units {
            if self.units[uid.0].kind == UnitType::BasicWorker && self.tiles[tid.0].wood_left > 0 {
                self.owner_add_resources(owner, &forest_prod);
                self.tiles[tid.0].wood_left -= wood_per;
            }
        }
        let stumps = self.tiles[tid.0].rounds_stumps;
        if stumps == resources::FOREST_GROW_TIME {
            self.tiles[tid.0].rounds_stumps = 0;
            self.tiles[tid.0].wood_left =
                resources::forest_capacity().get(BasicResource::Wood).unwrap();
        } else if self.tiles[tid.0].wood_left == 0 && stumps == 0 {
            self.tiles[tid.0].rounds_stumps += 1;
        } else if self.tiles[tid.0].wood_left == 0 {
            self.tiles[tid.0].rounds_stumps += 1;
        }
    }

    fn gen_abundant_forest(&mut self, tid: TileId, owner: Option<PlayerId>) {
        let prod = resources::abundant_forest_production();
        let units = self.tiles[tid.0].units.clone();
        for uid in units {
            if self.units[uid.0].kind == UnitType::BasicWorker {
                self.owner_add_resources(owner, &prod);
                break;
            }
        }
    }

    fn gen_mountain(&mut self, tid: TileId, owner: Option<PlayerId>) {
        let is_mine = matches!(&self.tiles[tid.0].building, Some(b) if b.kind == BuildingType::Mine);
        if !is_mine {
            return;
        }
        let prod = self.tiles[tid.0].building.as_ref().unwrap().production();
        let has_expert = self.tile_has_unit_type(tid, UnitType::Expert);
        let units = self.tiles[tid.0].units.clone();
        for uid in units {
            if self.units[uid.0].kind == UnitType::BasicWorker {
                self.owner_add_resources(owner, &prod);
                if has_expert {
                    self.owner_add_resources(owner, &prod);
                }
            }
        }
    }

    fn gen_river(&mut self, tid: TileId, owner: Option<PlayerId>) {
        let kind = match &self.tiles[tid.0].building {
            Some(b) => b.kind,
            None => return,
        };
        match kind {
            BuildingType::Hydro => {
                if !self.tile_has_unit_type(tid, UnitType::Expert) {
                    return;
                }
                let prod = self.tiles[tid.0].building.as_ref().unwrap().production();
                let units = self.tiles[tid.0].units.clone();
                for uid in units {
                    if self.units[uid.0].kind == UnitType::BasicWorker {
                        self.owner_add_resources(owner, &prod);
                    }
                }
            }
            BuildingType::Bridge => {
                let prod = self.tiles[tid.0].building.as_ref().unwrap().production();
                self.owner_add_resources(owner, &prod);
            }
            _ => {}
        }
    }

    // =======================================================================
    // GameEventHandler
    // =======================================================================

    /// `deleteUnitFromTile(unit, tile)` — destroy a unit: remove from tile and
    /// from its owner's `objects_`.
    pub fn delete_unit_from_tile(&mut self, uid: UnitId, tid: TileId) {
        self.tile_remove_unit(tid, uid);
        let owner = self.units[uid.0].owner;
        if let Some(p) = owner {
            self.players[p.0].remove_object(ObjId::Unit(uid));
        }
    }

    /// `clearTileUnits(tile)` — destroy all owned + conquering units on a tile.
    fn clear_tile_units(&mut self, tid: TileId) {
        let owned = self.tiles[tid.0].units.clone();
        for uid in owned {
            self.delete_unit_from_tile(uid, tid);
        }
        let conq = self.tiles[tid.0].conquering_units.clone();
        for uid in conq {
            self.delete_unit_from_tile(uid, tid);
        }
    }

    /// `makeHeadquarters` + place + claim surroundings, the HQ-placement half of
    /// `firstRoundActions`. The TS guards on `tile.getBuilding() !== null`. This
    /// does NOT call `changeTurn` (the caller decides) — the controller in later
    /// milestones drives turn flow, but for setup we expose claiming here and let
    /// callers advance. Returns true if the HQ was placed.
    pub fn first_round_actions(&mut self, tid: TileId) -> bool {
        if self.tiles[tid.0].building.is_some() {
            return false;
        }
        let current = self.current_player();
        // Build HQ owned by current.
        self.tiles[tid.0].building = Some(Building::new(BuildingType::Headquarters, Some(current)));
        self.tile_set_owner(tid, Some(current));

        // Claim the 8-neighbourhood unowned tiles.
        let coord = {
            let t = &self.tiles[tid.0];
            Coordinate::new(t.x, t.y)
        };
        for nc in coord.neighbours(1, self.settings.grid_width, self.settings.grid_height) {
            if let Some(ntid) = self.get_tile_at(nc) {
                if self.tiles[ntid.0].owner.is_none() {
                    self.tile_set_owner(ntid, Some(current));
                    // A Mikontalo on a claimed neighbour gets its owner set too.
                    if let Some(b) = &mut self.tiles[ntid.0].building {
                        if b.kind == BuildingType::Mikontalo {
                            b.owner = Some(current);
                        }
                    }
                }
            }
        }

        // Disconnect any owned tile not connected to the HQ.
        let hq_connected = self.get_hq_connected_tiles(current);
        let owned: Vec<ObjId> = self.players[current.0].objects.clone();
        for obj in owned {
            if let ObjId::Tile(t) = obj {
                if !hq_connected.contains(&t) {
                    self.tile_set_owner(t, None);
                }
            }
        }
        true
    }

    /// `endTurn()` — the full turn-resolution pipeline. Returns the win/lose
    /// outcome of this turn (see [`EndTurnOutcome`]).
    pub fn end_turn(&mut self) -> EndTurnOutcome {
        let current = self.current_player();

        // 1. Generate resources from current player's tiles (in objects_ order).
        let objs: Vec<ObjId> = self.players[current.0].objects.clone();
        for obj in &objs {
            if let ObjId::Tile(tid) = obj {
                self.generate_resources(*tid);
            }
        }
        // 2. Pay salaries for current player's units (in objects_ order).
        let objs: Vec<ObjId> = self.players[current.0].objects.clone();
        for obj in &objs {
            if let ObjId::Unit(uid) = obj {
                let salary = self.units[uid.0].get_salary();
                let owner = self.units[uid.0].owner;
                self.owner_add_resources(owner, &salary);
            }
        }
        // 3. Conquest resolution over ALL tiles (getTiles order).
        let tids: Vec<TileId> = self.tiles.iter().map(|t| t.id).collect();
        for tid in tids {
            self.conquer_tile(tid, current);
        }

        // 4. HQ-connectivity cut for every opponent.
        let opponents: Vec<PlayerId> = self
            .player_order
            .iter()
            .copied()
            .filter(|&p| p != current)
            .collect();
        for player in opponents {
            let hq_connected = self.get_hq_connected_tiles(player);
            let player_objs: Vec<ObjId> = self.players[player.0].objects.clone();
            for obj in player_objs {
                if let ObjId::Tile(tid) = obj {
                    if !hq_connected.contains(&tid) && hq_connected.is_empty() {
                        // No HQ at all: confiscated by current player.
                        // Observation-only: attribute the confiscated tile to
                        // `current`. Pure read+increment, no effect on state.
                        self.seat_events[current.0].tiles_gained_via_cut += 1;
                        self.clear_tile_units(tid);
                        self.tile_set_owner(tid, Some(current));
                        if let Some(b) = &mut self.tiles[tid.0].building {
                            if b.kind == BuildingType::Farm {
                                b.reset_farm();
                            }
                        }
                    } else if !hq_connected.contains(&tid) {
                        // Has an HQ but this tile is stranded: neutralised.
                        self.clear_tile_units(tid);
                        self.tile_set_owner(tid, None);
                        if let Some(b) = &mut self.tiles[tid.0].building {
                            if b.kind == BuildingType::Farm {
                                b.reset_farm();
                            }
                        }
                    }
                }
            }
            self.eliminate_excess_units(player);
            self.players[player.0].limit_resources();
        }

        let mut lost_this_round: Vec<PlayerId> = Vec::new();

        // 5a. Lost: no tiles (objects_.length === 0).
        let snapshot: Vec<PlayerId> = self.player_order.clone();
        for player in snapshot {
            if self.players[player.0].objects.is_empty() {
                self.set_player_as_lost(player, Some(current));
                lost_this_round.push(player);
                self.last_win_cause = Some(WinCause::Conquest);
            }
        }

        // 5b. Lost: any negative resource (and still has objects) -> neutralise.
        let snapshot: Vec<PlayerId> = self.player_order.clone();
        for player in snapshot {
            let has_negative = self.players[player.0].resources.iter().any(|(_, v)| v < 0);
            if has_negative && !self.players[player.0].objects.is_empty() {
                self.set_player_as_lost(player, Some(current));
                lost_this_round.push(player);
                self.neutralize_player(player);
                self.last_win_cause = Some(WinCause::Bankruptcy);
            }
        }

        // 5c. Win: a player owns >= 70% of tiles -> everyone else loses.
        let snapshot: Vec<PlayerId> = self.player_order.clone();
        for player in snapshot {
            let owned = self.get_tile_count_for_player(player);
            let total = self.get_tile_count();
            if total > 0 && (owned * 100) / total >= 70 {
                let others: Vec<PlayerId> = self
                    .player_order
                    .iter()
                    .copied()
                    .filter(|&p| p != player)
                    .collect();
                for p in others {
                    self.set_player_as_lost(p, None);
                    lost_this_round.push(p);
                    self.neutralize_player(p);
                }
                self.last_win_cause = Some(WinCause::Domination);
            }
        }

        // 6. Strange Device — destroy-detection THEN countdown tick + device-win,
        //    matching the TS order (gameeventhandler.ts 404-435): both run after the
        //    domination check and before change_turn.
        // 6a. A Device whose tile is no longer owned by its builder (captured this
        //     turn, or cut off / its builder neutralised) is DESTROYED: the building's
        //     owner never changes, but the tile's does. Destroying it reopens the
        //     one-per-game slot and clears the countdown.
        if let Some(dt) = self.find_strange_device_tile() {
            let builder = self.tiles[dt.0].building.as_ref().unwrap().owner;
            if self.tiles[dt.0].owner != builder {
                self.tiles[dt.0].building = None;
            }
        }
        // 6b. The surviving Device's clock ticks on its owner's end-of-turn; if it hits
        //     zero while still standing, the owner wins immediately — everyone else
        //     loses (same resolution as the 70% domination win).
        if let Some(dt) = self.find_strange_device_tile() {
            let owner = self.tiles[dt.0].owner;
            if owner == Some(current) {
                self.tiles[dt.0].building.as_mut().unwrap().decrement_countdown();
            }
            let countdown = self.tiles[dt.0].building.as_ref().unwrap().countdown;
            if let Some(owner) = owner {
                if countdown <= 0 {
                    let others: Vec<PlayerId> = self
                        .player_order
                        .iter()
                        .copied()
                        .filter(|&p| p != owner)
                        .collect();
                    for p in others {
                        self.set_player_as_lost(p, None);
                        lost_this_round.push(p);
                        self.neutralize_player(p);
                    }
                    self.last_win_cause = Some(WinCause::Device);
                }
            }
        }

        self.change_turn();

        // Determine outcome (mirrors the menu-view branch in the TS).
        let outcome = if self.player_order.is_empty() {
            EndTurnOutcome::Tie
        } else if self.player_order.len() == 1 {
            EndTurnOutcome::Win(self.player_order[0])
        } else if lost_this_round.is_empty() {
            EndTurnOutcome::Continue
        } else {
            EndTurnOutcome::PlayersLost(lost_this_round)
        };
        outcome
    }

    /// `neutralizePlayer(player)` — strip every owned tile (units cleared, owner
    /// nulled, farms reset, HQ marked conquered).
    pub fn neutralize_player(&mut self, player: PlayerId) {
        let objs: Vec<ObjId> = self.players[player.0].objects.clone();
        for obj in objs {
            if let ObjId::Tile(tid) = obj {
                self.clear_tile_units(tid);
                self.tile_set_owner(tid, None);
                if let Some(b) = &mut self.tiles[tid.0].building {
                    match b.kind {
                        BuildingType::Farm => b.reset_farm(),
                        BuildingType::Headquarters => b.conquered = true,
                        _ => {}
                    }
                }
            }
        }
    }

    // =======================================================================
    // Buying helpers
    // =======================================================================

    /// `canBuyUnitOrBuilding(cost)` for the current player.
    fn can_buy(&self, cost: &ResourceMap) -> bool {
        let p = self.current_player();
        self.players[p.0].has_enough_resources(cost)
    }
    /// `buyUnitOrBuilding(cost)` — charge if affordable.
    fn buy(&mut self, cost: &ResourceMap) {
        let p = self.current_player();
        if !self.players[p.0].has_enough_resources(cost) {
            return;
        }
        self.players[p.0].add_or_remove_resources(cost);
    }

    /// Allocate a new unit in the arena (does NOT place it).
    fn make_unit(&mut self, kind: UnitType, owner: Option<PlayerId>) -> UnitId {
        let id = UnitId(self.units.len());
        self.units.push(Unit {
            id,
            kind,
            owner,
            location: None,
            is_conquering: false,
        });
        id
    }

    // =======================================================================
    // Scenario construction (used by tests, the parity harness, and the AI
    // controller's HQ-placement setup). These bypass affordability/legality on
    // purpose — they are the headless analogue of the firstRoundActions /
    // save-restore "direct" placement paths.
    // =======================================================================

    /// Look up a player by 1-based player number.
    pub fn player_id_by_num(&self, num: i64) -> Option<PlayerId> {
        self.players
            .iter()
            .find(|p| p.player_num == num)
            .map(|p| p.id)
    }

    /// Give a player ownership of a tile (registers it in `objects_`), without
    /// any connectivity check — the direct analogue of `tile.setOwner(player)`.
    pub fn set_tile_owner(&mut self, tid: TileId, owner: Option<PlayerId>) {
        self.tile_set_owner(tid, owner);
    }

    /// Place a building on a tile for free, owned by `owner`. Mirrors the
    /// save-restore `placeBuildingDirect`.
    pub fn place_building(&mut self, tid: TileId, kind: BuildingType, owner: Option<PlayerId>) {
        self.tiles[tid.0].building = Some(Building::new(kind, owner));
    }

    /// Spawn a unit of `kind` owned by `owner` and add it to `tile`. `conquering`
    /// selects the owned vs conquering list. Skips the cost/availability checks
    /// (test/setup helper). Returns the new unit id.
    pub fn spawn_unit_on_tile(
        &mut self,
        kind: UnitType,
        owner: PlayerId,
        tid: TileId,
        conquering: bool,
    ) -> UnitId {
        let uid = self.make_unit(kind, None);
        self.units[uid.0].is_conquering = conquering;
        self.unit_set_owner(uid, Some(owner));
        self.tile_add_unit(tid, uid);
        uid
    }

    /// Directly set a player's treasury (test/setup helper; analogue of
    /// `PlayerBase.setResources`).
    pub fn set_player_resources(&mut self, player: PlayerId, money: i64, wood: i64, stone: i64, metal: i64) {
        let mut m = ResourceMap::new();
        m.set(BasicResource::Money, money);
        m.set(BasicResource::Wood, wood);
        m.set(BasicResource::Stone, stone);
        m.set(BasicResource::Metal, metal);
        self.players[player.0].resources = m;
    }

    // =======================================================================
    // AI action surface (IAiActions) + queries the candidate layer needs
    // =======================================================================

    /// `aiBuildBuilding(buildingString, tile)` — build on an owned tile.
    pub fn ai_build_building(&mut self, building_string: &str, tid: TileId) -> bool {
        let current = self.current_player();
        let kind = match BuildingType::from_str(building_string) {
            Some(k) => k,
            None => return false,
        };
        // One Strange Device per game — refuse a second BEFORE any cost is charged
        // (TS `makeBuilding` returns null before `canBuy`).
        if kind == BuildingType::StrangeDevice {
            if self.has_strange_device() {
                return false;
            }
            // The Device tile can never hold defenders (Tile::has_space_for_units), so
            // it must be built on an EMPTY tile — else you could pre-stack soldiers then
            // build on top and make it impossible to conquer.
            if !self.tiles[tid.0].units.is_empty() {
                return false;
            }
        }
        let cost = kind.build_cost();
        if !self.can_buy(&cost) {
            return false;
        }
        self.buy(&cost);
        // addBuilding: a building on a Forest is rejected in the headless sim
        // (the TS turns the forest into grassland via the renderer path, which
        // the AI never targets). Otherwise place it.
        if self.tiles[tid.0].tile_type == TileType::Forest {
            return false;
        }
        let mut building = Building::new(kind, Some(current));
        if kind == BuildingType::StrangeDevice {
            // Countdown scales with map size (bigger map = longer to mass an army +
            // cross it). Computed before the building is moved into the tile.
            building.countdown = resources::strange_device_countdown(self.get_tile_count());
        }
        self.tiles[tid.0].building = Some(building);
        if kind == BuildingType::StrangeDevice {
            // Observation-only: count the build for harness telemetry (parity-safe).
            self.seat_events[current.0].strange_devices_built += 1;
            // The cap-halving bites immediately: disband any soldiers now over the new
            // halved cap (else the degenerate line is "field a full army, THEN build").
            self.eliminate_excess_units(current);
        }
        true
    }

    /// `aiBuyAndPlaceUnit(type, tile)` — buy a unit and place it on `tile`.
    pub fn ai_buy_and_place_unit(&mut self, unit_string: &str, tid: TileId) -> bool {
        let current = self.current_player();
        let kind = match UnitType::from_str(unit_string) {
            Some(k) => k,
            None => return false,
        };
        if kind == UnitType::Soldier && self.free_soldier_amount(current) <= 0 {
            return false;
        }
        if (kind == UnitType::BasicWorker || kind == UnitType::Expert)
            && self.free_unit_amount(current) <= 0
        {
            return false;
        }
        let cost = kind.cost();
        if !self.can_buy(&cost) {
            return false;
        }
        // Faithful port of TS `aiBuyAndPlaceUnit` + `UnitBase.canBePlacedOnTile`,
        // including an original engine quirk that produces an *orphan* worker.
        //
        // The TS order is: makeUnit (conquering flag defaults to FALSE — the unit
        // is not yet parented) → canBuy → `unit.canBePlacedOnTile(tile)` → then a
        // try-block that calls addParentTile (which NOW sets the real conquering
        // flag from the tile owner), setOwner (registers the unit on the player, so
        // it immediately counts toward free-unit bookkeeping) and `tile.addUnit`
        // (which re-checks room for the REAL conquering status and THROWS if full,
        // caught → return false). Crucially:
        //   * `canBePlacedOnTile` checks room against the *pre-parent* flag, i.e.
        //     ALWAYS the regular `hasSpaceForUnits`, plus tile availability.
        //   * `tile.addUnit` checks room against the *real* flag.
        // So a neutral tile that is full of conquering units (regular count 0) PASSES
        // `canBePlacedOnTile` (regular room) but FAILS in `addUnit` (conquering full).
        // setOwner has already run, so the unit stays REGISTERED (counted, shrinking
        // free-unit room) yet unplaced and UNPAID. The candidate layer depends on
        // this (golden trace-2, round 40: a failed Expand-hire onto a full neutral
        // tile leaves a phantom worker that suppresses a later staffing hire).
        let avail = self.get_available_tiles();
        // canBePlacedOnTile: regular-room (pre-parent flag is false) AND available.
        if !self.tiles[tid.0].has_space_for_units() || !avail.contains(&tid) {
            return false;
        }

        // makeUnit + (addParentTile sets the real conquering flag) + setOwner: the
        // unit is now registered and counted.
        let is_conquering = self.tiles[tid.0].owner != Some(current);
        let uid = self.make_unit(kind, Some(current));
        self.units[uid.0].is_conquering = is_conquering;
        self.unit_set_owner(uid, Some(current));

        // tile.addUnit room check against the REAL conquering status.
        let has_room = if is_conquering {
            self.tiles[tid.0].has_space_for_conquering_units()
        } else {
            self.tiles[tid.0].has_space_for_units()
        };
        if !has_room {
            // `addUnit` throws → caught → return false; the unit is left an orphan
            // (registered + counted, not on any tile) and is NOT charged.
            return false;
        }

        self.tile_add_unit(tid, uid);
        self.buy(&cost);
        true
    }

    /// `aiMoveUnit(unit, fromTile, toTile)` — relocate an owned unit.
    ///
    /// Faithful to TS `aiMoveUnit` + `UnitBase.canBePlacedOnTile`. Two distinct
    /// room checks with DIFFERENT conquering flags:
    ///   1. `canBePlacedOnTile(toTile)` uses the unit's CURRENT (pre-move)
    ///      `is_conquering` flag plus tile availability — this is the gate that may
    ///      bail early with NO state change.
    ///   2. `tile.addUnit` (after `setAsConquering` flips the flag to match the
    ///      destination owner) re-checks room for the NEW flag and THROWS if full.
    /// A unit currently standing as a regular unit on its own tile (flag=false)
    /// being moved onto a neutral/enemy tile is gated on the destination's REGULAR
    /// room (step 1, old flag), even though it will be placed as a conquering unit.
    /// So a target full of regular units but with conquering room is REJECTED at
    /// step 1 (golden trace-2 round 47: an assault soldier can't move onto an enemy
    /// tile occupied by 3 workers). Mirrors the original engine exactly.
    pub fn ai_move_unit(&mut self, uid: UnitId, from: TileId, to: TileId) -> bool {
        let current = self.current_player();
        if self.units[uid.0].owner != Some(current) {
            return false;
        }
        // canBePlacedOnTile: room check uses the unit's CURRENT conquering flag.
        let cur_conquering = self.units[uid.0].is_conquering;
        let pre_room = if cur_conquering {
            self.tiles[to.0].has_space_for_conquering_units()
        } else {
            self.tiles[to.0].has_space_for_units()
        };
        if !pre_room || !self.get_available_tiles().contains(&to) {
            return false;
        }
        // setOwner + setAsConquering(toTile.owner != current) + addUnit. The new
        // flag selects the room check inside addUnit; if full the TS `addUnit`
        // throws (caught → false). We guard the equivalent here.
        let new_conquering = self.tiles[to.0].owner != Some(current);
        let add_room = if new_conquering {
            self.tiles[to.0].has_space_for_conquering_units()
        } else {
            self.tiles[to.0].has_space_for_units()
        };
        if !add_room {
            // addUnit would throw before the unit is detached from `from`; TS
            // catches and returns false with no net change.
            return false;
        }
        self.unit_set_owner(uid, Some(current));
        self.units[uid.0].is_conquering = new_conquering;
        self.tile_add_unit(to, uid);
        self.tile_remove_unit(from, uid);
        true
    }

    // --- queries the candidate/feature layer (cp-ai) will need ---

    /// Owned tiles of a player (`M.ownedTiles`), in `objects_` order.
    pub fn owned_tiles(&self, player: PlayerId) -> Vec<TileId> {
        self.players[player.0]
            .objects
            .iter()
            .filter_map(|o| match o {
                ObjId::Tile(t) => Some(*t),
                _ => None,
            })
            .collect()
    }

    /// Units (owned) currently on a tile, in tile order.
    pub fn tile_units(&self, tid: TileId) -> &[UnitId] {
        &self.tiles[tid.0].units
    }
    /// Conquering units on a tile, in tile order.
    pub fn tile_conquering_units(&self, tid: TileId) -> &[UnitId] {
        &self.tiles[tid.0].conquering_units
    }

    /// The buildable-building list for a tile, for the current player's owner
    /// context (`getBuildableBuildings`). Used by the candidate generator.
    pub fn buildable_buildings(&self, tid: TileId) -> Vec<&'static str> {
        let tile = &self.tiles[tid.0];
        match tile.tile_type {
            TileType::Grassland => self.buildable_grassland(tid),
            TileType::Forest => self.buildable_forest(tid),
            TileType::AbundantForest => vec![],
            TileType::Mountain => vec!["Mine"],
            TileType::River => {
                if tile.river_orientation == 1 || tile.river_orientation == 0 {
                    vec!["Bridge", "Hydroelectric Power Plant"]
                } else {
                    vec![]
                }
            }
        }
    }

    fn buildable_grassland(&self, tid: TileId) -> Vec<&'static str> {
        let owner = self.tiles[tid.0].owner;
        let mut list = vec!["Farm", "Village", "Outpost", "Nuclear Power Plant"];
        for ntid in self.neighbour_tiles(tid) {
            let n = &self.tiles[ntid.0];
            if let Some(b) = &n.building {
                if n.owner == owner
                    && (b.kind == BuildingType::Headquarters || b.kind == BuildingType::Outpost)
                {
                    // Outposts may not sit next to an HQ/another outpost.
                    list = vec!["Farm", "Village", "Nuclear Power Plant"];
                    break;
                }
            }
        }
        // The Strange Device is offered on any owned, UNOCCUPIED grassland while none
        // exists yet (at most one in the whole game) — mirrors tiles.ts. It must be
        // empty because the Device tile can never hold units.
        if !self.has_strange_device() && self.tiles[tid.0].units.is_empty() {
            list.push("Strange Device");
        }
        list
    }

    fn buildable_forest(&self, tid: TileId) -> Vec<&'static str> {
        if self.tiles[tid.0].wood_left == 0 {
            let mut list = vec!["Farm", "Village", "Outpost", "Nuclear Power Plant"];
            for ntid in self.neighbour_tiles(tid) {
                if let Some(b) = &self.tiles[ntid.0].building {
                    if b.kind == BuildingType::Headquarters || b.kind == BuildingType::Outpost {
                        list = vec!["Farm", "Village", "Nuclear Power Plant"];
                        break;
                    }
                }
            }
            if !self.has_strange_device() && self.tiles[tid.0].units.is_empty() {
                list.push("Strange Device");
            }
            return list;
        }
        vec![]
    }
}

/// Why the field collapsed to a single survivor this turn. Telemetry only (set on
/// `Game.last_win_cause`); the tile-majority tiebreak + true-timeout are resolved
/// harness-side where the round cap lives, so they are NOT causes here. See
/// STRANGE-DEVICE-DESIGN.md §10.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WinCause {
    /// A standing Strange Device's countdown reached 0.
    Device,
    /// A player reached >= 70% tile domination.
    Domination,
    /// An opponent was reduced to 0 tiles (conquest / HQ-cut confiscation).
    Conquest,
    /// An opponent went bankrupt (negative resources).
    Bankruptcy,
}

/// What an `end_turn` resolved to (the TS branches into win/tie/lost menus).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EndTurnOutcome {
    /// Game continues, no eliminations this turn.
    Continue,
    /// One or more players were eliminated this turn (game continues).
    PlayersLost(Vec<PlayerId>),
    /// Exactly one player remains — they win (`reason` = last-standing/domination).
    Win(PlayerId),
    /// No players remain — a tie.
    Tie,
}
