//! Port of `src/core/coordinate.ts` (`Core/coordinate.*`).
//!
//! Coordinates are `i32` (the TS uses JS numbers but only ever stores small
//! integer grid positions; intermediate values can go negative before
//! clamping, so a signed type is required).
//!
//! Field-name mapping: the TS private fields are `m_x` / `m_y`. In Rust they
//! become `x` / `y`. The TS accessor methods `x()` / `y()` map to the public
//! fields; mutators `set_x` / `set_y` map to direct assignment.

/// `Core::Coordinate::Direction`. Discriminants match the TS enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Direction {
    N = 0,
    Ne = 1,
    E = 2,
    Se = 3,
    S = 4,
    Sw = 5,
    W = 6,
    Nw = 7,
    End = 8,
}

/// A grid coordinate. Mirrors `Core::Coordinate`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Coordinate {
    pub x: i32,
    pub y: i32,
}

impl Coordinate {
    pub fn new(x: i32, y: i32) -> Self {
        Coordinate { x, y }
    }

    /// (dx, dy) unit step for a direction, scaled by `steps`.
    fn translate_direction(direction: Direction, steps: i32) -> (i32, i32) {
        let (dx, dy) = match direction {
            Direction::N => (0, -1),
            Direction::Ne => (1, -1),
            Direction::E => (1, 0),
            Direction::Se => (1, 1),
            Direction::S => (0, 1),
            Direction::Sw => (-1, 1),
            Direction::W => (-1, 0),
            Direction::Nw => (-1, -1),
            Direction::End => (0, 0),
        };
        (dx * steps, dy * steps)
    }

    /// `neighbour_at(direction, steps = 1)`.
    pub fn neighbour_at(&self, direction: Direction, steps: i32) -> Coordinate {
        let (dx, dy) = Self::translate_direction(direction, steps);
        Coordinate::new(self.x + dx, self.y + dy)
    }

    /// All tiles within Chebyshev `radius`, clamped to
    /// `[0,width-1] x [0,height-1]`, excluding self.
    ///
    /// Iteration order matches the TS: outer loop x, inner loop y.
    pub fn neighbours(&self, radius: i32, width: i32, height: i32) -> Vec<Coordinate> {
        let mut x_lower = self.x - radius;
        let mut x_upper = self.x + radius;
        let mut y_lower = self.y - radius;
        let mut y_upper = self.y + radius;

        if x_lower < 0 {
            x_lower = 0;
        }
        if x_upper > width - 1 {
            x_upper = width - 1;
        }
        if y_lower < 0 {
            y_lower = 0;
        }
        if y_upper > height - 1 {
            y_upper = height - 1;
        }

        let mut result = Vec::new();
        for x in x_lower..=x_upper {
            for y in y_lower..=y_upper {
                if x == self.x && y == self.y {
                    continue;
                }
                result.push(Coordinate::new(x, y));
            }
        }
        result
    }

    /// The orthogonal neighbours inside the grid, in south, west, north, east
    /// order (matching the TS `neighbouringFour`).
    pub fn neighbouring_four(&self, width: i32, height: i32) -> Vec<Coordinate> {
        let mut x_lower = self.x - 1;
        let mut x_upper = self.x + 1;
        let mut y_lower = self.y - 1;
        let mut y_upper = self.y + 1;

        if x_lower < 0 {
            x_lower = 0;
        }
        if x_upper > width - 1 {
            x_upper = width - 1;
        }
        if y_lower < 0 {
            y_lower = 0;
        }
        if y_upper > height - 1 {
            y_upper = height - 1;
        }

        let mut result = Vec::new();
        if y_lower != self.y {
            result.push(Coordinate::new(self.x, y_lower));
        }
        if x_lower != self.x {
            result.push(Coordinate::new(x_lower, self.y));
        }
        if y_upper != self.y {
            result.push(Coordinate::new(self.x, y_upper));
        }
        if x_upper != self.x {
            result.push(Coordinate::new(x_upper, self.y));
        }
        result
    }

    pub fn add(&self, other: &Coordinate) -> Coordinate {
        Coordinate::new(self.x + other.x, self.y + other.y)
    }

    pub fn sub(&self, other: &Coordinate) -> Coordinate {
        Coordinate::new(self.x - other.x, self.y - other.y)
    }

    /// Stable string key for use in map/set lookups (`"x,y"`), matching the TS
    /// `key()`. Prefer using `Coordinate` directly as a `HashMap` key in Rust
    /// (it derives `Hash`/`Eq`); this exists for parity / debugging.
    pub fn key(&self) -> String {
        format!("{},{}", self.x, self.y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neighbour_at_directions() {
        let c = Coordinate::new(5, 5);
        assert_eq!(c.neighbour_at(Direction::N, 1), Coordinate::new(5, 4));
        assert_eq!(c.neighbour_at(Direction::Se, 2), Coordinate::new(7, 7));
        assert_eq!(c.neighbour_at(Direction::End, 3), Coordinate::new(5, 5));
    }

    #[test]
    fn neighbours_clamped_and_excludes_self() {
        let c = Coordinate::new(0, 0);
        let n = c.neighbours(1, 10, 10);
        // top-left corner: only (0,1),(1,0),(1,1) are in-bounds, self excluded.
        assert_eq!(n.len(), 3);
        assert!(!n.contains(&Coordinate::new(0, 0)));
        // outer x then inner y ordering: (0,1),(1,0),(1,1)
        assert_eq!(
            n,
            vec![
                Coordinate::new(0, 1),
                Coordinate::new(1, 0),
                Coordinate::new(1, 1)
            ]
        );
    }

    #[test]
    fn neighbouring_four_order_and_clamp() {
        let c = Coordinate::new(5, 5);
        let f = c.neighbouring_four(10, 10);
        // south, west, north, east order
        assert_eq!(
            f,
            vec![
                Coordinate::new(5, 4), // y_lower -> north visually but listed first
                Coordinate::new(4, 5),
                Coordinate::new(5, 6),
                Coordinate::new(6, 5),
            ]
        );
        // corner clamps out the off-grid sides
        let corner = Coordinate::new(0, 0).neighbouring_four(10, 10);
        assert_eq!(corner, vec![Coordinate::new(0, 1), Coordinate::new(1, 0)]);
    }

    #[test]
    fn add_sub_key() {
        let a = Coordinate::new(3, 4);
        let b = Coordinate::new(1, 2);
        assert_eq!(a.add(&b), Coordinate::new(4, 6));
        assert_eq!(a.sub(&b), Coordinate::new(2, 2));
        assert_eq!(a.key(), "3,4");
    }
}
