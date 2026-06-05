// Port of Overlays/{clickedtileborder,mousehoverborder,blockedtile}.
// These are plain coordinate-anchored GameObjects the scene draws like tiles.

import { Coordinate } from '../core/coordinate';
import { GameObject, IGameEventHandler, IObjectManager } from './base';

export class ClickedTileBorder extends GameObject {
  constructor(
    location: Coordinate,
    size_x: number,
    size_y: number,
    eventhandler: IGameEventHandler,
    objectmanager: IObjectManager,
  ) {
    super(eventhandler, objectmanager, { coordinate: location, width: size_x, height: size_y });
  }
  getType(): string {
    return 'ClickedTileBorder';
  }
}

export class BlockedTile extends GameObject {
  constructor(
    location: Coordinate,
    size_x: number,
    size_y: number,
    eventhandler: IGameEventHandler,
    objectmanager: IObjectManager,
  ) {
    super(eventhandler, objectmanager, { coordinate: location, width: size_x, height: size_y });
  }
  getType(): string {
    return 'BlockedTile';
  }
}

export class MouseHoverBorder extends GameObject {
  private drawn_ = false;
  constructor(
    location: Coordinate,
    size_x: number,
    size_y: number,
    eventhandler: IGameEventHandler,
    objectmanager: IObjectManager,
  ) {
    super(eventhandler, objectmanager, { coordinate: location, width: size_x, height: size_y });
  }
  getType(): string {
    return 'MouseHoverBorder';
  }
  drawn(): boolean {
    return this.drawn_;
  }
  setDrawn(d: boolean): void {
    this.drawn_ = d;
  }
}
