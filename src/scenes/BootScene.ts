// Preloads every texture the game uses, once, then signals readiness.

import Phaser from 'phaser';
import { allTextureKeys } from '../core/images';

export class BootScene extends Phaser.Scene {
  constructor() {
    super({ key: 'BootScene' });
  }

  preload(): void {
    for (const key of allTextureKeys()) {
      this.load.image(key, `assets/images/${key}.png`);
    }
  }

  create(): void {
    this.game.events.emit('boot-complete');
  }
}
