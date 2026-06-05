// Port of Core/coordinate.{h,cpp}.

export enum Direction {
  N = 0,
  NE = 1,
  E = 2,
  SE = 3,
  S = 4,
  SW = 5,
  W = 6,
  NW = 7,
  END = 8,
}

export class Coordinate {
  private m_x: number;
  private m_y: number;

  constructor(x: number, y: number) {
    this.m_x = x;
    this.m_y = y;
  }

  static copy(other: Coordinate): Coordinate {
    return new Coordinate(other.x(), other.y());
  }

  x(): number {
    return this.m_x;
  }
  y(): number {
    return this.m_y;
  }
  set_x(x: number): void {
    this.m_x = x;
  }
  set_y(y: number): void {
    this.m_y = y;
  }

  private translate_direction(direction: Direction, steps: number): [number, number] {
    let dx = 0;
    let dy = 0;
    switch (direction) {
      case Direction.N: dx = 0; dy = -1; break;
      case Direction.NE: dx = 1; dy = -1; break;
      case Direction.E: dx = 1; dy = 0; break;
      case Direction.SE: dx = 1; dy = 1; break;
      case Direction.S: dx = 0; dy = 1; break;
      case Direction.SW: dx = -1; dy = 1; break;
      case Direction.W: dx = -1; dy = 0; break;
      case Direction.NW: dx = -1; dy = -1; break;
      case Direction.END: dx = 0; dy = 0; break;
    }
    return [dx * steps, dy * steps];
  }

  neighbour_at(direction: Direction, steps = 1): Coordinate {
    const [dx, dy] = this.translate_direction(direction, steps);
    return new Coordinate(this.m_x + dx, this.m_y + dy);
  }

  /** All tiles within Chebyshev `radius`, clamped to [0,width-1]x[0,height-1], excluding self. */
  neighbours(radius: number, width: number, height: number): Coordinate[] {
    let x_lower = this.m_x - radius;
    let x_upper = this.m_x + radius;
    let y_lower = this.m_y - radius;
    let y_upper = this.m_y + radius;

    if (x_lower < 0) x_lower = 0;
    if (x_upper > width - 1) x_upper = width - 1;
    if (y_lower < 0) y_lower = 0;
    if (y_upper > height - 1) y_upper = height - 1;

    const result: Coordinate[] = [];
    for (let x = x_lower; x <= x_upper; x++) {
      for (let y = y_lower; y <= y_upper; y++) {
        if (x === this.m_x && y === this.m_y) continue;
        result.push(new Coordinate(x, y));
      }
    }
    return result;
  }

  /** The four orthogonal neighbours that lie inside the grid (south, west, north, east order). */
  neighbouringFour(width: number, height: number): Coordinate[] {
    let x_lower = this.m_x - 1;
    let x_upper = this.m_x + 1;
    let y_lower = this.m_y - 1;
    let y_upper = this.m_y + 1;

    if (x_lower < 0) x_lower = 0;
    if (x_upper > width - 1) x_upper = width - 1;
    if (y_lower < 0) y_lower = 0;
    if (y_upper > height - 1) y_upper = height - 1;

    const result: Coordinate[] = [];
    if (y_lower !== this.m_y) result.push(new Coordinate(this.m_x, y_lower));
    if (x_lower !== this.m_x) result.push(new Coordinate(x_lower, this.m_y));
    if (y_upper !== this.m_y) result.push(new Coordinate(this.m_x, y_upper));
    if (x_upper !== this.m_x) result.push(new Coordinate(x_upper, this.m_y));
    return result;
  }

  add(other: Coordinate): Coordinate {
    return new Coordinate(this.m_x + other.x(), this.m_y + other.y());
  }
  sub(other: Coordinate): Coordinate {
    return new Coordinate(this.m_x - other.x(), this.m_y - other.y());
  }
  equals(other: Coordinate): boolean {
    return this.m_x === other.x() && this.m_y === other.y();
  }

  /** Stable string key for use in Map/Set lookups. */
  key(): string {
    return `${this.m_x},${this.m_y}`;
  }
}
