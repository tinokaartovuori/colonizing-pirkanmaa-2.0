//! Port of the TS `src/model/*` object hierarchy
//! (`BaseObject → GameObject → PlaceableGameObject`, tiles, buildings, units,
//! player), flattened into an arena/index representation.
//!
//! ## Why indices instead of pointers
//! The TS model is a web of shared mutable references: a tile holds its units, a
//! unit knows its owner and parent tile, a player holds a heterogeneous
//! `objects_` list of tiles *and* units, and ownership comparisons rely on
//! reference identity (`tile.getOwner() === player`). Rust cannot express that
//! cyclic graph with plain `&` borrows. Rather than reach for `Rc<RefCell<…>>`
//! (which would scatter runtime borrow checks across the hot path), every
//! entity lives in a `Vec` owned by [`crate::managers::Game`] and is referenced
//! by a small copyable id. Identity comparisons become id equality, which is
//! exactly the semantics the TS relied on.
//!
//! The actual storage `Vec`s live in `managers.rs`; this module defines the
//! per-entity data and the id newtypes.

use crate::resources::{self, BasicResource, ResourceMap};

// ---------------------------------------------------------------------------
// Ids
// ---------------------------------------------------------------------------

/// Index into `Game::tiles`. Mirrors a `TileBase` pointer's identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TileId(pub usize);

/// Index into `Game::units`. Mirrors a `UnitBase` pointer's identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct UnitId(pub usize);

/// Index into `Game::players` (stable for the player's lifetime — players are
/// never moved within the `Vec`, only flagged dead). NOT the 1-based player
/// number; see [`Player::player_num`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PlayerId(pub usize);

/// An entry in a player's `objects_` list — the TS list is heterogeneous
/// (`GameObject[]` holding both `TileBase` and `UnitBase`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjId {
    Tile(TileId),
    Unit(UnitId),
}

// ---------------------------------------------------------------------------
// Enums for the string-typed `getType()` discriminants
// ---------------------------------------------------------------------------

/// Tile terrain class — the TS `getType()` string for tiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileType {
    Grassland,
    Forest,
    AbundantForest,
    Mountain,
    River,
}

impl TileType {
    /// The exact `getType()` string the TS returns (used for fingerprints).
    pub fn as_str(self) -> &'static str {
        match self {
            TileType::Grassland => "Grassland",
            TileType::Forest => "Forest",
            TileType::AbundantForest => "Abundant Forest",
            TileType::Mountain => "Mountain",
            TileType::River => "River",
        }
    }
}

/// Building class — the TS `getType()` string for buildings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildingType {
    Farm,
    Mine,
    Village,
    Outpost,
    Hydro,   // "Hydroelectric Power Plant"
    Nuclear, // "Nuclear Power Plant"
    Headquarters,
    Bridge,
    Mikontalo,
    StrangeDevice,
}

impl BuildingType {
    pub fn as_str(self) -> &'static str {
        match self {
            BuildingType::Farm => "Farm",
            BuildingType::Mine => "Mine",
            BuildingType::Village => "Village",
            BuildingType::Outpost => "Outpost",
            BuildingType::Hydro => "Hydroelectric Power Plant",
            BuildingType::Nuclear => "Nuclear Power Plant",
            BuildingType::Headquarters => "Headquarters",
            BuildingType::Bridge => "Bridge",
            BuildingType::Mikontalo => "Mikontalo",
            BuildingType::StrangeDevice => "Strange Device",
        }
    }

    /// Parse the action-surface string the AI passes (e.g. `"Farm"`,
    /// `"Hydroelectric Power Plant"`). Returns `None` for unknown strings,
    /// matching the TS `makeBuilding` which leaves `building` null.
    pub fn from_str(s: &str) -> Option<BuildingType> {
        Some(match s {
            "Farm" => BuildingType::Farm,
            "Mine" => BuildingType::Mine,
            "Village" => BuildingType::Village,
            "Outpost" => BuildingType::Outpost,
            "Hydroelectric Power Plant" => BuildingType::Hydro,
            "Nuclear Power Plant" => BuildingType::Nuclear,
            "Headquarters" => BuildingType::Headquarters,
            "Bridge" => BuildingType::Bridge,
            "Mikontalo" => BuildingType::Mikontalo,
            "Strange Device" => BuildingType::StrangeDevice,
            _ => return None,
        })
    }

    pub fn build_cost(self) -> ResourceMap {
        match self {
            BuildingType::Farm => resources::farm_build_cost(),
            BuildingType::Mine => resources::mine_build_cost(),
            BuildingType::Village => resources::village_build_cost(),
            BuildingType::Outpost => resources::outpost_build_cost(),
            BuildingType::Hydro => resources::hepp_build_cost(),
            BuildingType::Nuclear => resources::nuclearpp_build_cost(),
            BuildingType::Bridge => resources::bridge_build_cost(),
            BuildingType::StrangeDevice => resources::strange_device_build_cost(),
            // HQ and Mikontalo are placed for free / by terrain gen.
            BuildingType::Headquarters | BuildingType::Mikontalo => resources::no_resources(),
        }
    }

    pub fn production(self) -> ResourceMap {
        match self {
            BuildingType::Farm => resources::farm_production(),
            BuildingType::Mine => resources::mine_production(),
            BuildingType::Village => resources::village_production(),
            BuildingType::Outpost => resources::outpost_production(),
            BuildingType::Hydro => resources::hepp_production(),
            BuildingType::Nuclear => resources::nuclearpp_production(),
            BuildingType::Bridge => resources::bridge_production(),
            // The Device has no per-turn production — the soldier-cap halving is its
            // only cost (TS passes NO_RESOURCES as its production).
            BuildingType::StrangeDevice
            | BuildingType::Headquarters
            | BuildingType::Mikontalo => resources::no_resources(),
        }
    }
}

/// Unit class — the TS `getType()` string for units.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnitType {
    BasicWorker,
    Expert,
    Soldier,
}

impl UnitType {
    pub fn as_str(self) -> &'static str {
        match self {
            UnitType::BasicWorker => "BasicWorker",
            UnitType::Expert => "Expert",
            UnitType::Soldier => "Soldier",
        }
    }

    /// Single-char fingerprint code (`W`/`E`/`S`), per golden SCHEMA.
    pub fn code(self) -> char {
        match self {
            UnitType::BasicWorker => 'W',
            UnitType::Expert => 'E',
            UnitType::Soldier => 'S',
        }
    }

    pub fn from_str(s: &str) -> Option<UnitType> {
        Some(match s {
            "BasicWorker" => UnitType::BasicWorker,
            "Expert" => UnitType::Expert,
            "Soldier" => UnitType::Soldier,
            _ => return None,
        })
    }

    pub fn cost(self) -> ResourceMap {
        match self {
            UnitType::BasicWorker => resources::basic_worker_cost(),
            UnitType::Expert => resources::expert_cost(),
            UnitType::Soldier => resources::soldier_cost(),
        }
    }

    pub fn salary(self) -> ResourceMap {
        match self {
            UnitType::BasicWorker => resources::basic_worker_salary(),
            UnitType::Expert => resources::expert_salary(),
            UnitType::Soldier => resources::soldier_salary(),
        }
    }
}

// ---------------------------------------------------------------------------
// Building
// ---------------------------------------------------------------------------

/// Port of `BuildingBase` + concrete buildings. Concrete subclasses differed
/// only by type + (Farm) growth phase + (HQ) conquered flag, so a single struct
/// with a `kind` tag suffices.
#[derive(Debug, Clone)]
pub struct Building {
    pub kind: BuildingType,
    pub owner: Option<PlayerId>,
    /// Farm growth phase (`growthPhase_`, starts at 1). Unused by other kinds.
    pub growth_phase: i64,
    /// HQ conquered flag (`conqured_`). Unused by other kinds.
    pub conquered: bool,
    /// Strange Device countdown (`countdown_`, rounds left until a standing Device
    /// wins). Set on build; 0 for all other kinds. Unused by other kinds.
    pub countdown: i64,
}

impl Building {
    pub fn new(kind: BuildingType, owner: Option<PlayerId>) -> Self {
        Building {
            kind,
            owner,
            growth_phase: 1,
            conquered: false,
            countdown: 0,
        }
    }

    pub fn get_type(&self) -> BuildingType {
        self.kind
    }

    pub fn production(&self) -> ResourceMap {
        self.kind.production()
    }

    /// `Farm::setGrowthPhase` — wraps back to 1 once it reaches 5.
    pub fn set_growth_phase(&mut self, phase: i64) {
        self.growth_phase = phase;
        if self.growth_phase >= 5 {
            self.growth_phase = 1;
        }
    }

    /// `Farm::resetFarm`. (The TS also pokes the renderer; no-op here.)
    pub fn reset_farm(&mut self) {
        self.set_growth_phase(1);
    }

    /// `StrangeDevice::decrementCountdown` — never goes below 0.
    pub fn decrement_countdown(&mut self) {
        if self.countdown > 0 {
            self.countdown -= 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Unit
// ---------------------------------------------------------------------------

/// Port of `UnitBase` + BasicWorker/Expert/Soldier.
#[derive(Debug, Clone)]
pub struct Unit {
    pub id: UnitId,
    pub kind: UnitType,
    pub owner: Option<PlayerId>,
    /// `m_location` / current tile (parent tile). The unit also lives in that
    /// tile's `units_` or `conquering_units_` list.
    pub location: Option<TileId>,
    /// `isConqueringUnit_`.
    pub is_conquering: bool,
}

impl Unit {
    pub fn get_type(&self) -> UnitType {
        self.kind
    }
    pub fn get_salary(&self) -> ResourceMap {
        self.kind.salary()
    }
    pub fn get_cost(&self) -> ResourceMap {
        self.kind.cost()
    }
}

// ---------------------------------------------------------------------------
// Tile
// ---------------------------------------------------------------------------

/// Port of `TileBase` + the five concrete terrain tiles. The terrain-specific
/// behaviour (production, buildable rules, conquest) is dispatched on
/// [`Tile::tile_type`]; everything else is shared, exactly as in the TS.
#[derive(Debug, Clone)]
pub struct Tile {
    pub id: TileId,
    pub tile_type: TileType,
    pub x: i32,
    pub y: i32,
    pub owner: Option<PlayerId>,
    pub building: Option<Building>,
    /// `units_` (owned units), in tile order.
    pub units: Vec<UnitId>,
    /// `conqueringUnits_` (staged assault units), in tile order.
    pub conquering_units: Vec<UnitId>,
    pub max_units: i64,
    // Forest harvest state (`woodLeft_`, `roundsStumpsHaveBeen_`).
    pub wood_left: i64,
    pub rounds_stumps: i64,
    /// River orientation (`riverOrientation_`, default 3). 0=EW, 1=NS, 3=other.
    pub river_orientation: i64,
}

impl Tile {
    pub fn get_type(&self) -> TileType {
        self.tile_type
    }

    /// Number of owned units (`getUnitCount`).
    pub fn unit_count(&self) -> usize {
        self.units.len()
    }
    /// Number of conquering units (`getConqueringUnitCount`).
    pub fn conquering_unit_count(&self) -> usize {
        self.conquering_units.len()
    }

    /// `hasSpaceForUnits`.
    pub fn has_space_for_units(&self) -> bool {
        // A Strange Device tile holds at most ONE defending soldier (arc sd5 rebalance):
        // a single defender raises the crack requirement from 1 to 2 attackers so a lone
        // raider can no longer one-shot it, but the cap stays well below the normal 3 so
        // the device never becomes impossible to crack. Conquering units may still stage
        // (has_space_for_conquering_units is unchanged), so it remains crackable.
        if matches!(&self.building, Some(b) if b.kind == BuildingType::StrangeDevice) {
            return 1 + self.unit_count() as i64 <= 1;
        }
        1 + self.unit_count() as i64 <= self.max_units
    }
    /// `hasSpaceForConqueringUnits`.
    pub fn has_space_for_conquering_units(&self) -> bool {
        1 + self.conquering_unit_count() as i64 <= self.max_units
    }
}

// ---------------------------------------------------------------------------
// Player
// ---------------------------------------------------------------------------

/// Port of `PlayerBase`.
#[derive(Debug, Clone)]
pub struct Player {
    pub id: PlayerId,
    /// 1-based `playerNum_`.
    pub player_num: i64,
    pub name: String,
    /// `objects_` — heterogeneous list of owned tiles and units, in insertion
    /// order (order matters: `eliminateExcessUnits` and resource generation walk
    /// it in order).
    pub objects: Vec<ObjId>,
    pub resources: ResourceMap,
    pub max_unit_amount: i64,
    pub max_soldier_amount: i64,
    /// False once the player has been removed from the live player list.
    pub alive: bool,
}

impl Player {
    pub fn new(id: PlayerId, player_num: i64, name: String) -> Self {
        let mut p = Player {
            id,
            player_num,
            name,
            objects: Vec::new(),
            resources: ResourceMap::new(),
            max_unit_amount: 0,
            max_soldier_amount: 0,
            alive: true,
        };
        // PlayerBase ctor: addOrRemoveResources(STARTING_RESOURCES).
        p.add_or_remove_resources(&resources::starting_resources());
        p
    }

    /// `addOrRemoveResources` — merge (additively) into the treasury.
    pub fn add_or_remove_resources(&mut self, delta: &ResourceMap) {
        self.resources = resources::merge_resource_maps(&self.resources, delta);
    }

    /// `getResources`.
    pub fn get_resources(&self) -> &ResourceMap {
        &self.resources
    }

    /// `hasEnoughResources(cost)` — true iff applying `cost` leaves no negative.
    pub fn has_enough_resources(&self, cost: &ResourceMap) -> bool {
        let r = resources::merge_resource_maps(&self.resources, cost);
        let ok = r.iter().all(|(_, v)| v >= 0);
        ok
    }

    /// `getResources().get(MONEY) ?? 0` style accessor.
    pub fn money(&self) -> i64 {
        self.resources.get(BasicResource::Money).unwrap_or(0)
    }
    pub fn wood(&self) -> i64 {
        self.resources.get(BasicResource::Wood).unwrap_or(0)
    }
    pub fn stone(&self) -> i64 {
        self.resources.get(BasicResource::Stone).unwrap_or(0)
    }
    pub fn metal(&self) -> i64 {
        self.resources.get(BasicResource::Metal).unwrap_or(0)
    }

    pub fn has_object(&self, obj: ObjId) -> bool {
        self.objects.contains(&obj)
    }

    pub fn add_object(&mut self, obj: ObjId) {
        self.objects.push(obj);
    }

    /// `removeObject` — silently ignores a missing object (the TS throws and the
    /// only callers that matter wrap it in try/catch).
    pub fn remove_object(&mut self, obj: ObjId) {
        if let Some(i) = self.objects.iter().position(|&o| o == obj) {
            self.objects.remove(i);
        }
    }

    /// `limitResources` — clamp each present resource to its limit.
    pub fn limit_resources(&mut self) {
        let limits = resources::resource_limits();
        let keys: Vec<(BasicResource, i64)> = self.resources.iter().collect();
        for (k, v) in keys {
            if let Some(limit) = limits.get(k) {
                if v >= limit {
                    self.resources.set(k, limit);
                }
            }
        }
    }
}
