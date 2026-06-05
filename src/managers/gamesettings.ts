// Port of DAL/gamesettingsmanager.{h,cpp} + the grid-size computation that lives
// in mainwindow.cpp::initializeGame (kept together so callers get one factory).

import { IGameSettingsManager } from '../model/base';

const idiv = (a: number, b: number): number => Math.trunc(a / b);

export class GameSettingsManager implements IGameSettingsManager {
  constructor(
    private mapGridSize_: number,
    private menuGridSize_: number,
    private mapWidth_: number,
    private mapHeight_: number,
    private menuWidth_: number,
    private menuHeight_: number,
  ) {}

  getMapGridSize(): number {
    return this.mapGridSize_;
  }
  getMenuGridSize(): number {
    return this.menuGridSize_;
  }
  getMapWidth(): number {
    return this.mapWidth_;
  }
  getMapHeight(): number {
    return this.mapHeight_;
  }
  getMapGridWidth(): number {
    return idiv(this.mapWidth_, this.mapGridSize_);
  }
  getMapGridHeight(): number {
    return idiv(this.mapHeight_, this.mapGridSize_);
  }
  getMenuWidth(): number {
    return this.menuWidth_;
  }
  getMenuHeight(): number {
    return this.menuHeight_;
  }

  /** Reproduces mainwindow.cpp::initializeGame sizing maths (integer arithmetic). */
  static fromMapDimensions(width: number, height: number): GameSettingsManager {
    const mapGridSize = idiv(idiv(640, height), 2) * 2 + (idiv(height * 3, 2) - 10);
    const menuGridSize = 16;
    const menuGridWidth = 22;
    let menuGridHeight = idiv(mapGridSize * height, menuGridSize);
    const menuCap = 640 % menuGridHeight;
    if (menuCap > 0) menuGridHeight += 1;

    const menuWidth = menuGridSize * menuGridWidth;
    const menuHeight = menuGridSize * menuGridHeight;
    const mapWidth = mapGridSize * width;
    const mapHeight = mapGridSize * height;

    return new GameSettingsManager(mapGridSize, menuGridSize, mapWidth, mapHeight, menuWidth, menuHeight);
  }
}
