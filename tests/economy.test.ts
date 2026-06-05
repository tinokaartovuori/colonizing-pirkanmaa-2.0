import { describe, it, expect } from 'vitest';
import {
  BasicResource,
  mergeResourceMaps,
  reverseResourceMap,
  getNegativesMap,
  getPositivesMap,
  rmap,
  STARTING_RESOURCES,
} from '../src/core/resources';
import { Coordinate, Direction } from '../src/core/coordinate';
import { srand, rand } from '../src/core/rng';

describe('ResourceMap helpers (Course::basicresources)', () => {
  it('mergeResourceMaps sums shared keys and keeps unique keys', () => {
    const a = rmap({ [BasicResource.MONEY]: 100, [BasicResource.WOOD]: 50 });
    const b = rmap({ [BasicResource.MONEY]: -30, [BasicResource.STONE]: 10 });
    const m = mergeResourceMaps(a, b);
    expect(m.get(BasicResource.MONEY)).toBe(70);
    expect(m.get(BasicResource.WOOD)).toBe(50);
    expect(m.get(BasicResource.STONE)).toBe(10);
  });

  it('reverse / negatives / positives', () => {
    const a = rmap({ [BasicResource.MONEY]: 100, [BasicResource.WOOD]: -50 });
    expect(reverseResourceMap(a).get(BasicResource.MONEY)).toBe(-100);
    expect(getNegativesMap(a).get(BasicResource.MONEY)).toBe(0);
    expect(getNegativesMap(a).get(BasicResource.WOOD)).toBe(-50);
    expect(getPositivesMap(a).get(BasicResource.WOOD)).toBe(0);
    expect(getPositivesMap(a).get(BasicResource.MONEY)).toBe(100);
  });

  it('starting resources are 400/200/100/25', () => {
    expect(STARTING_RESOURCES.get(BasicResource.MONEY)).toBe(400);
    expect(STARTING_RESOURCES.get(BasicResource.WOOD)).toBe(200);
    expect(STARTING_RESOURCES.get(BasicResource.STONE)).toBe(100);
    expect(STARTING_RESOURCES.get(BasicResource.METAL)).toBe(25);
  });
});

describe('Coordinate (Course::Coordinate)', () => {
  it('neighbouringFour clamps at the border', () => {
    const c = new Coordinate(0, 0);
    const n = c.neighbouringFour(10, 10);
    // top-left corner: only south + east are inside
    expect(n.length).toBe(2);
  });

  it('neighbours within radius excludes self and clamps', () => {
    const c = new Coordinate(5, 5);
    expect(c.neighbours(1, 10, 10).length).toBe(8);
    const corner = new Coordinate(0, 0);
    expect(corner.neighbours(1, 10, 10).length).toBe(3);
  });

  it('neighbour_at directions', () => {
    const c = new Coordinate(5, 5);
    expect(c.neighbour_at(Direction.N).equals(new Coordinate(5, 4))).toBe(true);
    expect(c.neighbour_at(Direction.E).equals(new Coordinate(6, 5))).toBe(true);
  });
});

describe('MSVCRT rand() replica', () => {
  it('is deterministic for a given seed', () => {
    srand(1);
    const a = [rand(), rand(), rand()];
    srand(1);
    const b = [rand(), rand(), rand()];
    expect(a).toEqual(b);
  });

  it('matches known MSVCRT sequence for seed 1', () => {
    // First three rand() outputs of the Windows CRT LCG with seed 1.
    srand(1);
    expect([rand(), rand(), rand()]).toEqual([41, 18467, 6334]);
  });
});
