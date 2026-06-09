import Phaser from 'phaser';
import { injectStyles } from './ui/styles';
import { showStartDialog, showResumeDialog, StartSettings } from './ui/startdialog';
import { buildSnapshot, saveSnapshot, loadSnapshot, clearSnapshot, GameSnapshot } from './managers/persistence';
import { GameRecorder } from './managers/gamerecorder';
import { showHelpWindow } from './ui/help';
import { MenuController } from './ui/menu';
import { showTurnBanner, clearBanner } from './ui/banner';
import { GameSettingsManager } from './managers/gamesettings';
import { ObjectManager } from './managers/objectmanager';
import { PlayerManager } from './managers/playermanager';
import { GameEventHandler } from './managers/gameeventhandler';
import { AiController } from './managers/ai';
import { createNeuralController, createModelController } from './ai/nn';
import { ICpuController } from './ai/controller-types';
import { WorldGenerator } from './world/worldgenerator';
import { MouseHoverBorder } from './model/overlays';
import { TileBase } from './model/tile';
import { UnitBase } from './model/unit';
import { PlayerBase, isNeuralModelDifficulty } from './model/player';
import { Coordinate } from './core/coordinate';
import { ImageVectors, AnimationOptions } from './core/images';
import { BootScene } from './scenes/BootScene';
import { GameScene } from './scenes/GameScene';

injectStyles();

// #game is a full-viewport flex container (see index.html). The actual game —
// Phaser canvas + DOM menu — lives in an inner "stage" sized to the game's native
// pixels, which we scale as a whole to fit the viewport so tall maps never spill
// off the bottom of the window. Keeping canvas and menu in one scaled element
// preserves their alignment, and Phaser's ScaleManager corrects pointer input for
// the CSS scale (we call game.scale.refresh() whenever the scale changes).
const parent = document.getElementById('game') as HTMLElement;
const stage = document.createElement('div');
stage.id = 'cp-stage';
stage.style.position = 'relative';
stage.style.transformOrigin = 'center center';
parent.appendChild(stage);

let activeMenu: MenuController | null = null;
// Identity token for the running match; replaced on quit so stray CPU timers
// scheduled by a previous match become no-ops.
let matchToken: object = {};
/** Pause before a CPU starts its turn, and between its individual visible actions. */
const CPU_START_MS = 550;
const CPU_ACTION_MS = 320;
/** Upper bound on the viewport-fit zoom so the pixel font stays crisp & readable. */
const MAX_STAGE_SCALE = 2.5;
/** Current match's viewport-fit handler, re-bound to window resize each match. */
let fitStage: () => void = () => {};
window.addEventListener('resize', () => fitStage());

/** Current match's "write a save snapshot" handler, re-bound each match. Saving on
 *  beforeunload is what makes a browser refresh resume exactly where you left off. */
let saveCurrent: () => void = () => {};
window.addEventListener('beforeunload', () => saveCurrent());

/** Current match's Esc handler, re-bound each match. */
let onEscape: () => void = () => {};
window.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') onEscape();
});

const game = new Phaser.Game({
  type: Phaser.AUTO,
  parent: stage,
  width: 320,
  height: 320,
  backgroundColor: '#35772c',
  render: { pixelArt: true, roundPixels: true, antialias: false },
  scale: { mode: Phaser.Scale.NONE, autoCenter: Phaser.Scale.NO_CENTER },
  scene: [BootScene, GameScene],
});

game.events.once('boot-complete', () => {
  // Offer to resume a saved game if one survived a refresh; otherwise the start dialog.
  const saved = loadSnapshot();
  if (saved) {
    showResumeDialog(
      () => startMatch(saved.settings, saved),
      () => {
        clearSnapshot();
        showStartDialog(startMatch);
      },
    );
  } else {
    showStartDialog(startMatch);
  }
});

function startMatch(settings: StartSettings, restore?: GameSnapshot): void {
  if (activeMenu) {
    activeMenu.destroy();
    activeMenu = null;
  }
  clearBanner();

  const token = {};
  matchToken = token;
  const isCurrent = () => matchToken === token;

  const gsm = GameSettingsManager.fromMapDimensions(settings.width, settings.height);
  const totalWidth = gsm.getMapWidth() + gsm.getMenuWidth();
  const totalHeight = gsm.getMapHeight();

  stage.style.width = `${totalWidth}px`;
  stage.style.height = `${totalHeight}px`;
  game.scale.resize(totalWidth, totalHeight);

  // Scale the whole stage (board + menu) to fill the window as much as possible
  // while keeping its aspect ratio — scaling UP on big screens and DOWN on small
  // ones — so the game makes good use of any viewport. #game's flex centring keeps
  // it in the middle. A small margin stops it touching the edges, and we cap the
  // zoom so the pixel-art font never blows up to an unreadable size. Re-runs on
  // window resize.
  fitStage = () => {
    const margin = 24;
    const avail = (px: number) => Math.max(px - margin, 1);
    const scale = Math.min(MAX_STAGE_SCALE, avail(window.innerWidth) / totalWidth, avail(window.innerHeight) / totalHeight);
    stage.style.transform = `scale(${scale})`;
    game.scale.refresh(); // re-derive pointer-input mapping for the new CSS scale
  };
  fitStage();

  const objectManager = new ObjectManager();
  const menu = new MenuController(stage, gsm);
  const playerManager = new PlayerManager(settings.players, objectManager);
  const eventHandler = new GameEventHandler(objectManager, playerManager, menu, gsm);
  const ai = new AiController(eventHandler, objectManager, playerManager);
  // Per-player controller: heuristic AiController for easy/medium/hard, the
  // trained NeuralAiController for the nn-* opponents. Neural controllers are
  // cached per player (they carry no per-turn state, but this avoids re-allocs).
  const neuralCache = new Map<PlayerBase, ICpuController>();
  const aiFor = (player: PlayerBase): ICpuController => {
    const d = player.getDifficulty();
    const mapInfo = { width: settings.width, height: settings.height, seed: settings.seed };
    if (d === 'nn-easy' || d === 'nn-medium' || d === 'nn-hard') {
      let c = neuralCache.get(player);
      if (!c) {
        c = createNeuralController(eventHandler, objectManager, playerManager, d, Math.random, mapInfo);
        neuralCache.set(player, c);
      }
      return c;
    }
    if (isNeuralModelDifficulty(d)) {
      let c = neuralCache.get(player);
      if (!c) {
        c = createModelController(eventHandler, objectManager, playerManager, d.slice('model:'.length), Math.random, mapInfo);
        neuralCache.set(player, c);
      }
      return c;
    }
    return ai;
  };

  menu.setEventHandler(eventHandler);
  menu.onHelp = showHelpWindow;
  menu.onQuit = quitToMenu;
  eventHandler.onRestart = quitToMenu;
  eventHandler.onTurnChanged = driveTurn;

  // Passive game recorder: append per-turn history at each turn boundary, and on
  // game-over upload the completed human-vs-AI game to the analysis backend. Fresh
  // per match (a restored save starts a new history from the restored state).
  const recorder = new GameRecorder(objectManager, playerManager, {
    width: settings.width,
    height: settings.height,
    seed: settings.seed,
  });
  eventHandler.onTurnEnded = (endedBy) => {
    if (!isCurrent()) return;
    recorder.recordTurn(endedBy);
  };
  eventHandler.onGameOver = (gameInfo) => {
    if (!isCurrent()) return;
    recorder.finish(gameInfo);
  };

  objectManager.addDALS(eventHandler, menu, gsm);
  activeMenu = menu;

  // Persist this match's state (only while it's the live match).
  const saveSnapshotNow = (): void => {
    if (!isCurrent()) return;
    saveSnapshot(buildSnapshot(objectManager, playerManager, { width: settings.width, height: settings.height, seed: settings.seed }));
  };
  saveCurrent = saveSnapshotNow;

  // Esc: cancel an in-progress unit drag; if nothing is being dragged, close whatever
  // submenu is open and return to the default ("main") view. Ignored while a modal
  // dialog is open, during a CPU turn, or once the game is over.
  onEscape = () => {
    if (!isCurrent()) return;
    if (document.querySelector('.cp-overlay')) return; // a modal handles its own keys
    if (playerManager.getPlayers().length <= 1) return; // game over
    if (playerManager.getCurrentPlayer().isCpu()) return; // not the human's turn
    if (!eventHandler.cancelUnitAction()) eventHandler.openDefaultMenuView();
  };

  // Dev-only debug handle (enables headless verification).
  if (import.meta.env.DEV) {
    (window as unknown as { __cp: unknown }).__cp = { game, objectManager, playerManager, eventHandler, gsm, ai };
  }

  // --- turn driving (CPU players + "your turn" banner) ----------------------

  function driveTurn(): void {
    if (!isCurrent()) return;
    if (playerManager.getPlayers().length <= 1) {
      clearSnapshot(); // game over — don't resume a finished match
      return;
    }
    saveSnapshotNow(); // snapshot at every turn boundary

    const cur = playerManager.getCurrentPlayer();
    if (cur.isCpu()) {
      menu.setCpuTurnMenuView(cur);
      // Persistent banner — stays for the whole CPU turn (replaced when it ends).
      showTurnBanner(stage, `${cur.getName()} is playing…`, cur.getPlayerNum(), 0);
      window.setTimeout(() => runCpuTurn(cur), CPU_START_MS);
    } else {
      const msg =
        cur.getObjects().length === 0
          ? `${cur.getName()}, choose your starting tile`
          : `It's your turn, ${cur.getName()}!`;
      showTurnBanner(stage, msg, cur.getPlayerNum(), 2400);
    }
  }

  function runCpuTurn(player: PlayerBase): void {
    if (!isCurrent() || playerManager.getCurrentPlayer() !== player) return;

    eventHandler.setAiActive(true);

    if (player.getObjects().length === 0) {
      // First round: place the HQ (this advances the turn via firstRoundActions).
      try {
        aiFor(player).placeHeadquarters(player);
      } catch {
        /* ignore */
      }
      eventHandler.setAiActive(false);
      return;
    }

    // Play the turn one visible action at a time so it looks like real play.
    const steps = aiFor(player).planTurn(player);
    const stepOnce = (): void => {
      if (!isCurrent() || playerManager.getCurrentPlayer() !== player) {
        eventHandler.setAiActive(false);
        return;
      }
      let done = false;
      try {
        done = steps.next().done === true;
      } catch {
        done = true;
      }
      if (done) {
        eventHandler.setAiActive(false);
        eventHandler.endTurn(); // -> onTurnChanged -> driveTurn
        return;
      }
      menu.setCpuTurnMenuView(player); // refresh so resources tick down as it spends
      window.setTimeout(stepOnce, CPU_ACTION_MS);
    };
    stepOnce();
  }

  // --- input that is ignored while a CPU is playing -------------------------

  const onTileClick = (tile: TileBase): void => {
    if (playerManager.getCurrentPlayer().isCpu()) return;
    eventHandler.tileClicked(tile);
    saveSnapshotNow();
  };
  const onUnitClick = (unit: UnitBase, tile: TileBase): boolean => {
    if (playerManager.getCurrentPlayer().isCpu()) return false;
    return eventHandler.selectUnitForMove(unit, tile);
  };

  // Hover border, created up-front like mainwindow.cpp.
  const hover = new MouseHoverBorder(new Coordinate(0, 0), 1, 1, eventHandler, objectManager);
  hover.setImageFiles(ImageVectors.MOUSEHOVERBORDER);
  hover.setAnimationOption(AnimationOptions.MOUSEHOVERBORDER);
  objectManager.setHoverBorder(hover);

  game.scene.start('GameScene', {
    objectManager,
    settings: gsm,
    onTileClick,
    onUnitClick,
    onReady: (scene: GameScene) => {
      eventHandler.setGameScene(scene);
      objectManager.setGameScene(scene);
      new WorldGenerator().generateMap(settings.width, settings.height, settings.seed, {
        objectManager,
        eventHandler,
        gameSettings: gsm,
        scene,
      });
      if (restore) {
        // Re-apply the saved state on top of the freshly generated terrain, then
        // resume from the saved player's turn.
        eventHandler.restoreSnapshot(restore);
        const cur = playerManager.getCurrentPlayer();
        if (cur.getObjects().length === 0) menu.selectFirstTileMenuView(cur);
        else menu.setDefaultMenuView();
      } else {
        menu.selectFirstTileMenuView(playerManager.getCurrentPlayer());
      }
      // Kick off the turn: place the banner for a human, or start the CPU.
      driveTurn();
    },
  });
}

function quitToMenu(): void {
  matchToken = {}; // invalidate any pending CPU timers
  saveCurrent = () => {}; // stop the torn-down match from being re-saved on unload
  onEscape = () => {};
  clearSnapshot(); // the player chose to abandon this game
  clearBanner();
  game.scene.stop('GameScene');
  if (activeMenu) {
    activeMenu.destroy();
    activeMenu = null;
  }
  showStartDialog(startMatch);
}
