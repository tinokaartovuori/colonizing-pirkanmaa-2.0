//! Port of `src/world/worldgenerator.ts` (`Core/worldgenerator`).
//! FIDELITY-CRITICAL: the order of `rand()` calls must match the TS exactly so a
//! seed reproduces the identical map. Tile storage order is **column-major**:
//! the build loop is `for x in 0..sizeX { for y in 0..sizeY }`, so a tile's
//! index is `x * sizeY + y`.
//!
//! The TS `generateTerrain` uses a `matrix[y][x]` (row-major) `number[][]` and
//! relies on `std::vector::at()` throwing out of range to short-circuit a cell's
//! forest spread (the C++ try/catch). We reproduce that with checked helpers
//! that signal "out of range", aborting the remaining spreads for that cell —
//! preserving the exact `rand()` consumption.

use crate::managers::Game;
use crate::model::{Building, BuildingType, Tile, TileId, TileType};
use crate::resources::{self, BasicResource};
use crate::rng::Rng;

/// Terrain codes used inside the generator matrix (match the TS literals):
/// 0 grassland, 1 forest, 2 mountain, 3 grassland+Mikontalo, 4 river,
/// 5 mountain (was forest), 6 abundant forest.
fn matrix_get(matrix: &[Vec<i32>], y: i32, x: i32) -> Option<i32> {
    if y < 0 || y as usize >= matrix.len() {
        return None;
    }
    let row = &matrix[y as usize];
    if x < 0 || x as usize >= row.len() {
        return None;
    }
    Some(row[x as usize])
}

/// Set with bounds-check; returns `false` (the "throw") when out of range.
fn matrix_set(matrix: &mut [Vec<i32>], y: i32, x: i32, value: i32) -> bool {
    if y < 0 || y as usize >= matrix.len() {
        return false;
    }
    let row = &mut matrix[y as usize];
    if x < 0 || x as usize >= row.len() {
        return false;
    }
    row[x as usize] = value;
    true
}

impl Game {
    /// `WorldGenerator.generateMap` — seed the RNG, build terrain, then create
    /// the tiles in column-major order and push them into the object manager.
    pub fn generate_map(&mut self, size_x: i32, size_y: i32, seed: u32) {
        let mut rng = Rng::new(seed);

        let matrix = generate_terrain(&mut rng, size_x, size_y);

        let mut tiles: Vec<Tile> = Vec::with_capacity((size_x * size_y) as usize);
        for x in 0..size_x {
            for y in 0..size_y {
                let val = matrix[y as usize][x as usize];
                let id = TileId(tiles.len());
                let (tile_type, spawn_mikontalo) = match val {
                    0 => (TileType::Grassland, false),
                    1 => {
                        // Forest consumes one rand() (sprite choice) — order matters.
                        let _ = rng.rand();
                        (TileType::Forest, false)
                    }
                    2 => (TileType::Mountain, false),
                    4 => (TileType::River, false),
                    5 => (TileType::Mountain, false),
                    3 => (TileType::Grassland, true),
                    6 => (TileType::AbundantForest, false),
                    _ => (TileType::Grassland, false),
                };

                let wood_left = if tile_type == TileType::Forest {
                    resources::forest_capacity()
                        .get(BasicResource::Wood)
                        .unwrap()
                } else {
                    0
                };

                let river_orientation = if tile_type == TileType::River {
                    get_river_orientation(x, y, &matrix, 4).1
                } else {
                    3
                };

                let building = if spawn_mikontalo {
                    Some(Building::new(BuildingType::Mikontalo, None))
                } else {
                    None
                };

                tiles.push(Tile {
                    id,
                    tile_type,
                    x,
                    y,
                    owner: None,
                    building,
                    units: Vec::new(),
                    conquering_units: Vec::new(),
                    max_units: 3,
                    wood_left,
                    rounds_stumps: 0,
                    river_orientation,
                });
            }
        }

        self.tiles = tiles;
    }
}

/// `generateTerrain` — the six-phase procedural map. Returns a `matrix[y][x]`.
fn generate_terrain(rng: &mut Rng, size_x: i32, size_y: i32) -> Vec<Vec<i32>> {
    let sx = size_x;
    let sy = size_y;
    let mut matrix: Vec<Vec<i32>> = Vec::with_capacity(sy as usize);

    // Phase 1: 15% forest seed.
    for _y in 0..sy {
        let mut row = Vec::with_capacity(sx as usize);
        for _x in 0..sx {
            let rnd = (rng.rand() as i32 % 100) + 1;
            row.push(if rnd < 15 { 1 } else { 0 });
        }
        matrix.push(row);
    }

    // Phase 2: cluster forests (>40 spreads to 8 neighbours; abort cell on the
    // first out-of-range `setAt`, matching the C++ try/catch).
    let temp: Vec<Vec<i32>> = matrix.clone();
    for y in 0..sy {
        for x in 0..sx {
            if temp[y as usize][x as usize] == 1 {
                // Each guarded set: if a rand() roll passes AND the set is out of
                // range, the cell aborts (the rest of its spreads are skipped).
                let offsets: [(i32, i32); 8] = [
                    (y - 1, x - 1),
                    (y - 1, x),
                    (y - 1, x + 1),
                    (y, x - 1),
                    (y, x + 1),
                    (y + 1, x - 1),
                    (y + 1, x),
                    (y + 1, x + 1),
                ];
                for (ny, nx) in offsets {
                    if (rng.rand() as i32 % 100) + 1 > 40 {
                        if !matrix_set(&mut matrix, ny, nx, 1) {
                            break; // out-of-range throw aborts remaining spreads
                        }
                    }
                }
            }
        }
    }

    // Phase 3: a single winding river.
    let dir = (rng.rand() as i32 % 2) - 1; // 0 or -1
    let starting_tile_x = (rng.rand() as i32 % (sx - 4)) + 2;
    let starting_tile_y = (rng.rand() as i32 % (sy - 4)) + 2;
    let mut current_x: i32 = 0;
    let mut current_y: i32 = 0;
    let mut last_dirs: Vec<i32> = vec![0, 0];
    if dir == 0 {
        current_x = starting_tile_x;
    } else {
        current_y = starting_tile_y;
    }

    loop {
        if current_x >= sx || current_x < 0 {
            break;
        }
        if current_y >= sy || current_y < 0 {
            break;
        }
        matrix[current_y as usize][current_x as usize] = 4;

        let back = last_dirs[last_dirs.len() - 1];
        let prev = last_dirs[last_dirs.len() - 2];
        let next_dir: i32 = if back == 0 && prev == 0 {
            rng.rand() as i32 % 3
        } else if back == 0 && prev == 1 {
            (rng.rand() as i32 % 2) - 1
        } else if back == 0 && prev == 2 {
            ((rng.rand() as i32 % 2) - 1) * 2
        } else if back == 1 {
            (rng.rand() as i32 % 2) - 1
        } else if back == 2 {
            ((rng.rand() as i32 % 2) - 1) * 2
        } else {
            0
        };

        if dir == 0 {
            if next_dir == 0 {
                current_y += 1;
            }
            if next_dir == 1 {
                current_x += 1;
            }
            if next_dir == 2 {
                current_x -= 1;
            }
        } else {
            if next_dir == 0 {
                current_x += 1;
            }
            if next_dir == 1 {
                current_y += 1;
            }
            if next_dir == 2 {
                current_y -= 1;
            }
        }
        last_dirs.push(next_dir);
    }

    // Phase 4: mountains. The loop bound is re-evaluated every iteration (the TS
    // calls rand() in the condition), so we replicate that exactly.
    // Math.round(sizeX*sizeY*0.3) — round half away from zero (positive here).
    let mountain_base = ((sx * sy) as f64 * 0.3).round() as i32;
    let mut i = 0i32;
    loop {
        let bound = (rng.rand() as i32 % mountain_base) + 4;
        if i >= bound {
            break;
        }
        let rnd_x = rng.rand() as i32 % sx;
        let rnd_y = rng.rand() as i32 % sy;
        let cell = matrix[rnd_y as usize][rnd_x as usize];
        if cell == 1 {
            matrix[rnd_y as usize][rnd_x as usize] = 5;
        } else if cell == 0 {
            matrix[rnd_y as usize][rnd_x as usize] = 2;
        }
        // else: continue (no change)
        i += 1;
    }

    // Phase 5: abundant forests. Math.trunc((sizeX*sizeY)/30).
    let abundant_count = (sx * sy) / 30;
    for _ in 0..abundant_count {
        let rnd_x = rng.rand() as i32 % sx;
        let rnd_y = rng.rand() as i32 % sy;
        if matrix[rnd_y as usize][rnd_x as usize] == 4 {
            continue;
        }
        matrix[rnd_y as usize][rnd_x as usize] = 6;
    }

    // Phase 6: a single Mikontalo.
    loop {
        let rnd_x = rng.rand() as i32 % sx;
        let rnd_y = rng.rand() as i32 % sy;
        if matrix[rnd_y as usize][rnd_x as usize] == 4 {
            continue;
        }
        matrix[rnd_y as usize][rnd_x as usize] = 3;
        break;
    }

    matrix
}

/// River shape tag returned by [`get_river_orientation`] — encodes which sprite
/// the TS picked (we only need it for the orientation value, but the shape is
/// kept for completeness / future swim-sprite parity).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RiverShape {
    Nw,
    Ne,
    Sw,
    Se,
    Ns,
    Ew,
}

/// `getRiverOrientation(x,y,matrix,num)` — returns `(shape, orientation)`.
/// Orientation: 3 = bend/default, 1 = N-S, 0 = E-W (matches the TS second tuple
/// element; only orientation 0/1 makes a river buildable). The control flow
/// (and therefore the result) mirrors the TS try/catch nesting exactly.
fn get_river_orientation(x: i32, y: i32, matrix: &[Vec<i32>], num: i32) -> (RiverShape, i64) {
    let size_x = matrix[0].len() as i32;
    let size_y = matrix.len() as i32;

    // Inner closure: returns Some(result) if every probed cell is in-range and a
    // branch matched; None if any probe was out of range (the C++ `.at()` throw)
    // OR no branch matched (falls through).
    // We mirror the TS: the outer try probes the four "both neighbour" cases;
    // any out-of-range probe jumps to the edge-handling catch.
    let g = |yy: i32, xx: i32| matrix_get(matrix, yy, xx);

    // Outer try block. In the TS, `at()` is evaluated left-to-right inside each
    // `if`; an out-of-range access throws immediately and control passes to the
    // catch. We emulate by treating any None probe as a throw -> go to edge code.
    let outer = (|| -> Option<(RiverShape, i64)> {
        let up = g(y - 1, x)?;
        let left = g(y, x - 1)?;
        if up == num && left == num {
            return Some((RiverShape::Nw, 3));
        }
        let right = g(y, x + 1)?;
        if up == num && right == num {
            return Some((RiverShape::Ne, 3));
        }
        let down = g(y + 1, x)?;
        if down == num && left == num {
            return Some((RiverShape::Sw, 3));
        }
        if down == num && right == num {
            return Some((RiverShape::Se, 3));
        }
        if up == num && down == num {
            return Some((RiverShape::Ns, 1));
        }
        if left == num && right == num {
            return Some((RiverShape::Ew, 0));
        }
        None
    })();

    if let Some(res) = outer {
        return res;
    }

    // Edge-handling catch. Each branch probes only in-range cells for that edge;
    // out-of-range probes inside fall to the inner catch (which the TS leaves as
    // a debug no-op) and the function returns the default N-S piece.
    let edge = (|| -> Option<(RiverShape, i64)> {
        if x == 0 && y == 0 {
            if g(y + 1, x)? == num {
                return Some((RiverShape::Ns, 1));
            }
            if g(y, x + 1)? == num {
                return Some((RiverShape::Ew, 0));
            }
        } else if x == 0 && y == size_y - 1 {
            if g(y - 1, x)? == num {
                return Some((RiverShape::Ns, 1));
            }
            if g(y, x + 1)? == num {
                return Some((RiverShape::Ew, 0));
            }
        } else if x == size_x - 1 && y == 0 {
            if g(y + 1, x)? == num {
                return Some((RiverShape::Ns, 1));
            }
            if g(y, x - 1)? == num {
                return Some((RiverShape::Ew, 0));
            }
        } else if x == size_x - 1 && y == size_y - 1 {
            if g(y - 1, x)? == num {
                return Some((RiverShape::Ns, 1));
            }
            if g(y, x - 1)? == num {
                return Some((RiverShape::Ew, 0));
            }
        } else if y == 0 {
            let left = g(y, x - 1)?;
            let right = g(y, x + 1)?;
            if left == num && right == num {
                return Some((RiverShape::Ew, 0));
            }
            let down = g(y + 1, x)?;
            if down == num && left == num {
                return Some((RiverShape::Sw, 3));
            } else if down == num && right == num {
                return Some((RiverShape::Se, 3));
            } else if left == num {
                return Some((RiverShape::Nw, 3));
            } else if right == num {
                return Some((RiverShape::Ne, 3));
            } else {
                return Some((RiverShape::Ns, 1));
            }
        } else if y == size_y - 1 {
            let left = g(y, x - 1)?;
            let right = g(y, x + 1)?;
            if left == num && right == num {
                return Some((RiverShape::Ew, 0));
            }
            let up = g(y - 1, x)?;
            if up == num && left == num {
                return Some((RiverShape::Nw, 3));
            } else if up == num && right == num {
                return Some((RiverShape::Ne, 3));
            } else if left == num {
                return Some((RiverShape::Sw, 3));
            } else if right == num {
                return Some((RiverShape::Se, 3));
            } else {
                return Some((RiverShape::Ns, 1));
            }
        } else if x == 0 {
            let up = g(y - 1, x)?;
            let down = g(y + 1, x)?;
            if up == num && down == num {
                return Some((RiverShape::Ns, 1));
            }
            let right = g(y, x + 1)?;
            if right == num && up == num {
                return Some((RiverShape::Ne, 3));
            } else if right == num && down == num {
                return Some((RiverShape::Se, 3));
            } else if up == num {
                return Some((RiverShape::Nw, 3));
            } else if down == num {
                return Some((RiverShape::Sw, 3));
            } else {
                return Some((RiverShape::Ew, 0));
            }
        } else if x == size_x - 1 {
            let up = g(y - 1, x)?;
            let down = g(y + 1, x)?;
            if up == num && down == num {
                return Some((RiverShape::Ns, 1));
            }
            let left = g(y, x - 1)?;
            if left == num && up == num {
                return Some((RiverShape::Nw, 3));
            } else if left == num && down == num {
                return Some((RiverShape::Sw, 3));
            } else if up == num {
                return Some((RiverShape::Ne, 3));
            } else if down == num {
                return Some((RiverShape::Se, 3));
            } else {
                return Some((RiverShape::Ew, 0));
            }
        }
        None
    })();

    edge.unwrap_or((RiverShape::Ns, 1))
}
