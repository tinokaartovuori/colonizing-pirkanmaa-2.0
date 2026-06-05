// Port of Core/worldgenerator.{h,cpp}.
// The RNG call order matches the original exactly so a seed reproduces the same
// map (paired with the MSVCRT rand() in core/rng.ts). std::vector::at() throwing
// on out-of-range is reproduced with at()/setAt() helpers so the C++ try/catch
// control flow (and therefore the rand() consumption) is preserved.

import { rand, srand } from '../core/rng';
import { Coordinate } from '../core/coordinate';
import { ImageVectors, AnimationOptions, AnimationOption, ImageVector } from '../core/images';
import { TileBase } from '../model/tile';
import { Grassland, Forest, AbundantForest, Mountain, River } from '../model/tiles';
import { Mikontalo } from '../model/building';
import { IGameEventHandler, IObjectManager, IGameScene, IGameSettingsManager } from '../model/base';

class RangeErr extends Error {}

function at(matrix: number[][], y: number, x: number): number {
  if (y < 0 || y >= matrix.length) throw new RangeErr('y');
  const row = matrix[y];
  if (x < 0 || x >= row.length) throw new RangeErr('x');
  return row[x];
}
function setAt(matrix: number[][], y: number, x: number, value: number): void {
  if (y < 0 || y >= matrix.length) throw new RangeErr('y');
  const row = matrix[y];
  if (x < 0 || x >= row.length) throw new RangeErr('x');
  row[x] = value;
}

export interface WorldGenDeps {
  objectManager: IObjectManager & { addTiles(tiles: TileBase[]): void };
  eventHandler: IGameEventHandler;
  gameSettings: IGameSettingsManager;
  scene: IGameScene;
}

export class WorldGenerator {
  generateMap(sizeX: number, sizeY: number, seed: number, deps: WorldGenDeps): void {
    srand(seed);

    const tiles: TileBase[] = [];
    const matrix = this.generateTerrain(sizeX, sizeY);

    for (let x = 0; x < sizeX; x++) {
      for (let y = 0; y < sizeY; y++) {
        const val = matrix[y][x];
        let imageVector: ImageVector = ImageVectors.GRASSLAND;
        let animationOption: AnimationOption = AnimationOptions.GRASSLAND;
        let spawnMikontalo = false;
        let tile: TileBase;

        const coord = new Coordinate(x, y);
        const eh = deps.eventHandler;
        const om = deps.objectManager;

        if (val === 0) {
          tile = new Grassland(coord, 1, 1, eh, om);
          imageVector = ImageVectors.GRASSLAND;
          animationOption = AnimationOptions.GRASSLAND;
        } else if (val === 1) {
          tile = new Forest(coord, 1, 1, eh, om);
          const rnd = (rand() % 2) - 1;
          imageVector = rnd === 0 ? ImageVectors.FOREST_1 : ImageVectors.FOREST_2;
          animationOption = AnimationOptions.FOREST;
        } else if (val === 2) {
          tile = new Mountain(coord, 1, 1, eh, om);
          imageVector = ImageVectors.MOUNTAIN;
          animationOption = AnimationOptions.MOUNTAIN;
        } else if (val === 4) {
          tile = new River(coord, 1, 1, eh, om);
          imageVector = this.getRiverOrientation(x, y, matrix, 4)[0];
          animationOption = AnimationOptions.RIVER;
        } else if (val === 5) {
          tile = new Mountain(coord, 1, 1, eh, om);
          imageVector = ImageVectors.MOUNTAIN_FOREST;
          animationOption = AnimationOptions.MOUNTAIN_FOREST;
        } else if (val === 3) {
          tile = new Grassland(coord, 1, 1, eh, om);
          imageVector = ImageVectors.GRASSLAND;
          animationOption = AnimationOptions.GRASSLAND;
          spawnMikontalo = true;
        } else if (val === 6) {
          tile = new AbundantForest(coord, 1, 1, eh, om);
          imageVector = ImageVectors.ABUNDANT_FOREST;
          animationOption = AnimationOptions.FOREST;
        } else {
          tile = new Grassland(coord, 1, 1, eh, om);
          imageVector = ImageVectors.GRASSLAND;
          animationOption = AnimationOptions.GRASSLAND;
        }

        tile.setGameSettings(deps.gameSettings);

        if (spawnMikontalo) {
          const mikontalo = new Mikontalo(eh, om, null);
          mikontalo.setImageFiles(ImageVectors.MIKONTALO);
          tile.addBuilding(mikontalo);
          eh.updateTile(tile);
        }

        if (tile instanceof River) {
          tile.setRiverOrientation(this.getRiverOrientation(x, y, matrix, 4)[1]);
          if (imageVector === ImageVectors.RIVER_EW) tile.setRiverShape('EW');
          if (imageVector === ImageVectors.RIVER_NS) tile.setRiverShape('NS');
          if (imageVector === ImageVectors.RIVER_NE) tile.setRiverShape('NE');
          if (imageVector === ImageVectors.RIVER_NW) tile.setRiverShape('NW');
          if (imageVector === ImageVectors.RIVER_SW) tile.setRiverShape('SW');
          if (imageVector === ImageVectors.RIVER_SE) tile.setRiverShape('SE');
        }

        tile.setImageFiles(imageVector);
        tile.setAnimationOption(animationOption);
        deps.scene.drawItem(tile);
        tiles.push(tile);
      }
    }

    deps.objectManager.addTiles(tiles);
  }

  generateTerrain(sizeX: number, sizeY: number): number[][] {
    const matrix: number[][] = [];

    // Phase 1: 15% forest seed.
    for (let y = 0; y < sizeY; y++) {
      const row: number[] = [];
      for (let x = 0; x < sizeX; x++) {
        const rnd = (rand() % 100) + 1;
        row.push(rnd < 15 ? 1 : 0);
      }
      matrix.push(row);
    }

    // Phase 2: cluster forests (60% spread to 8 neighbours; faithful to the
    // try/catch that aborts a cell's remaining spreads once an .at() throws).
    const tempMatrix = matrix.map((r) => [...r]);
    for (let y = 0; y < sizeY; y++) {
      for (let x = 0; x < sizeX; x++) {
        if (tempMatrix[y][x] === 1) {
          try {
            if ((rand() % 100) + 1 > 40) setAt(matrix, y - 1, x - 1, 1);
            if ((rand() % 100) + 1 > 40) setAt(matrix, y - 1, x, 1);
            if ((rand() % 100) + 1 > 40) setAt(matrix, y - 1, x + 1, 1);
            if ((rand() % 100) + 1 > 40) setAt(matrix, y, x - 1, 1);
            if ((rand() % 100) + 1 > 40) setAt(matrix, y, x + 1, 1);
            if ((rand() % 100) + 1 > 40) setAt(matrix, y + 1, x - 1, 1);
            if ((rand() % 100) + 1 > 40) setAt(matrix, y + 1, x, 1);
            if ((rand() % 100) + 1 > 40) setAt(matrix, y + 1, x + 1, 1);
          } catch {
            continue;
          }
        }
      }
    }

    // Phase 3: a single winding river.
    const dir = (rand() % 2) - 1; // 0 or -1
    const startingTileX = (rand() % (sizeX - 4)) + 2;
    const startingTileY = (rand() % (sizeY - 4)) + 2;
    let currentX = 0;
    let currentY = 0;
    const lastDirs = [0, 0];
    if (dir === 0) currentX = startingTileX;
    else currentY = startingTileY;

    for (;;) {
      if (currentX >= sizeX || currentX < 0) break;
      if (currentY >= sizeY || currentY < 0) break;
      matrix[currentY][currentX] = 4;

      let nextDir: number;
      const back = lastDirs[lastDirs.length - 1];
      const prev = lastDirs[lastDirs.length - 2];
      if (back === 0 && prev === 0) nextDir = rand() % 3;
      else if (back === 0 && prev === 1) nextDir = (rand() % 2) - 1;
      else if (back === 0 && prev === 2) nextDir = ((rand() % 2) - 1) * 2;
      else if (back === 1) nextDir = (rand() % 2) - 1;
      else if (back === 2) nextDir = ((rand() % 2) - 1) * 2;
      else nextDir = 0;

      if (dir === 0) {
        if (nextDir === 0) currentY += 1;
        if (nextDir === 1) currentX += 1;
        if (nextDir === 2) currentX -= 1;
      } else {
        if (nextDir === 0) currentX += 1;
        if (nextDir === 1) currentY += 1;
        if (nextDir === 2) currentY -= 1;
      }
      lastDirs.push(nextDir);
    }

    // Phase 4: mountains (loop bound re-evaluated every iteration, as in C++).
    for (let i = 0; i < (rand() % Math.round(sizeX * sizeY * 0.3)) + 4; i++) {
      const rndX = rand() % sizeX;
      const rndY = rand() % sizeY;
      if (matrix[rndY][rndX] === 1) matrix[rndY][rndX] = 5;
      else if (matrix[rndY][rndX] === 0) matrix[rndY][rndX] = 2;
      else continue;
    }

    // Phase 5: abundant forests.
    for (let x = 0; x < Math.trunc((sizeX * sizeY) / 30); x++) {
      const rndX = rand() % sizeX;
      const rndY = rand() % sizeY;
      if (matrix[rndY][rndX] === 4) continue;
      matrix[rndY][rndX] = 6;
    }

    // Phase 6: a single Mikontalo.
    for (;;) {
      const rndX = rand() % sizeX;
      const rndY = rand() % sizeY;
      if (matrix[rndY][rndX] === 4) continue;
      matrix[rndY][rndX] = 3;
      break;
    }

    return matrix;
  }

  getRiverOrientation(x: number, y: number, matrix: number[][], num: number): [ImageVector, number] {
    const sizeX = matrix[0].length;
    const sizeY = matrix.length;
    try {
      if (at(matrix, y - 1, x) === num && at(matrix, y, x - 1) === num) return [ImageVectors.RIVER_NW, 3];
      if (at(matrix, y - 1, x) === num && at(matrix, y, x + 1) === num) return [ImageVectors.RIVER_NE, 3];
      if (at(matrix, y + 1, x) === num && at(matrix, y, x - 1) === num) return [ImageVectors.RIVER_SW, 3];
      if (at(matrix, y + 1, x) === num && at(matrix, y, x + 1) === num) return [ImageVectors.RIVER_SE, 3];
      if (at(matrix, y - 1, x) === num && at(matrix, y + 1, x) === num) return [ImageVectors.RIVER_NS, 1];
      if (at(matrix, y, x - 1) === num && at(matrix, y, x + 1) === num) return [ImageVectors.RIVER_EW, 0];
    } catch {
      try {
        if (x === 0 && y === 0) {
          if (at(matrix, y + 1, x) === num) return [ImageVectors.RIVER_NS, 1];
          if (at(matrix, y, x + 1) === num) return [ImageVectors.RIVER_EW, 0];
        } else if (x === 0 && y === sizeY - 1) {
          if (at(matrix, y - 1, x) === num) return [ImageVectors.RIVER_NS, 1];
          if (at(matrix, y, x + 1) === num) return [ImageVectors.RIVER_EW, 0];
        } else if (x === sizeX - 1 && y === 0) {
          if (at(matrix, y + 1, x) === num) return [ImageVectors.RIVER_NS, 1];
          if (at(matrix, y, x - 1) === num) return [ImageVectors.RIVER_EW, 0];
        } else if (x === sizeX - 1 && y === sizeY - 1) {
          if (at(matrix, y - 1, x) === num) return [ImageVectors.RIVER_NS, 1];
          if (at(matrix, y, x - 1) === num) return [ImageVectors.RIVER_EW, 0];
        } else if (y === 0) {
          if (at(matrix, y, x - 1) === num && at(matrix, y, x + 1) === num) return [ImageVectors.RIVER_EW, 0];
          else if (at(matrix, y + 1, x) === num && at(matrix, y, x - 1) === num) return [ImageVectors.RIVER_SW, 3];
          else if (at(matrix, y + 1, x) === num && at(matrix, y, x + 1) === num) return [ImageVectors.RIVER_SE, 3];
          else if (at(matrix, y, x - 1) === num) return [ImageVectors.RIVER_NW, 3];
          else if (at(matrix, y, x + 1) === num) return [ImageVectors.RIVER_NE, 3];
          else return [ImageVectors.RIVER_NS, 1];
        } else if (y === sizeY - 1) {
          if (at(matrix, y, x - 1) === num && at(matrix, y, x + 1) === num) return [ImageVectors.RIVER_EW, 0];
          else if (at(matrix, y - 1, x) === num && at(matrix, y, x - 1) === num) return [ImageVectors.RIVER_NW, 3];
          else if (at(matrix, y - 1, x) === num && at(matrix, y, x + 1) === num) return [ImageVectors.RIVER_NE, 3];
          else if (at(matrix, y, x - 1) === num) return [ImageVectors.RIVER_SW, 3];
          else if (at(matrix, y, x + 1) === num) return [ImageVectors.RIVER_SE, 3];
          else return [ImageVectors.RIVER_NS, 1];
        } else if (x === 0) {
          if (at(matrix, y - 1, x) === num && at(matrix, y + 1, x) === num) return [ImageVectors.RIVER_NS, 1];
          else if (at(matrix, y, x + 1) === num && at(matrix, y - 1, x) === num) return [ImageVectors.RIVER_NE, 3];
          else if (at(matrix, y, x + 1) === num && at(matrix, y + 1, x) === num) return [ImageVectors.RIVER_SE, 3];
          else if (at(matrix, y - 1, x) === num) return [ImageVectors.RIVER_NW, 3];
          else if (at(matrix, y + 1, x) === num) return [ImageVectors.RIVER_SW, 3];
          else return [ImageVectors.RIVER_EW, 0];
        } else if (x === sizeX - 1) {
          if (at(matrix, y - 1, x) === num && at(matrix, y + 1, x) === num) return [ImageVectors.RIVER_NS, 1];
          else if (at(matrix, y, x - 1) === num && at(matrix, y - 1, x) === num) return [ImageVectors.RIVER_NW, 3];
          else if (at(matrix, y, x - 1) === num && at(matrix, y + 1, x) === num) return [ImageVectors.RIVER_SW, 3];
          else if (at(matrix, y - 1, x) === num) return [ImageVectors.RIVER_NE, 3];
          else if (at(matrix, y + 1, x) === num) return [ImageVectors.RIVER_SE, 3];
          else return [ImageVectors.RIVER_EW, 0];
        }
      } catch {
        // matches the qDebug("Error with river orientation") branch
      }
    }
    // Original returns an empty tuple here; default to a straight N-S piece.
    return [ImageVectors.RIVER_NS, 1];
  }
}
