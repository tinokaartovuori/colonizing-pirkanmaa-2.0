// Replay dashboard — browse recorded human-vs-AI games and step through them.
//
// Reached from the start menu ("Watch Games"). Left: a scrollable list of games
// fetched from the analysis backend (VITE_CP_SERVER, same as gamerecorder.ts).
// Right: a replay viewer — the live GameScene board, a turn slider + step/play
// controls, and a per-turn metrics panel.
//
// How a turn is rendered: each recorded turn is a full GameSnapshot. restoreSnapshot
// is ADDITIVE and sparse (it only touches tiles present in the snapshot and stacks
// units), so it can't be replayed incrementally — instead, exactly like the proven
// save/resume path, every rendered turn gets a FRESH board: new managers + a fresh
// GameScene + generateMap(seed) + restoreSnapshot(turn). Scrubbing backward is
// therefore always correct. Board renders are cheap (small maps) and only happen on
// slider release / step / autoplay; live drag only updates the (Phaser-free) info.

import type Phaser from 'phaser';
import { GameSnapshot } from '../managers/persistence';
import { GameSettingsManager } from '../managers/gamesettings';
import { ObjectManager } from '../managers/objectmanager';
import { PlayerManager } from '../managers/playermanager';
import { GameEventHandler } from '../managers/gameeventhandler';
import { WorldGenerator } from '../world/worldgenerator';
import { MenuController } from './menu';
import { GameScene } from '../scenes/GameScene';
import { PlayerConfig } from '../model/player';

/** Backend base URL; override with VITE_CP_SERVER at build time (mirrors gamerecorder.ts). */
const SERVER_URL =
  (import.meta.env.VITE_CP_SERVER as string | undefined) ?? 'http://127.0.0.1:8790';

/** Per-player seat colour, indexed by 0-based seat (mirrors menu.ts COLOR_BAR / banner.ts). */
const COLOR_BALL = ['red', 'blue', 'purple', 'yellow'];

/** Autoplay cadence (ms per turn). Each step re-renders the board on a fresh scene. */
const AUTOPLAY_MS = 450;

interface SeatMetrics {
  seat: number;
  money: number; wood: number; stone: number; metal: number;
  tiles: number; soldiers: number; buildings: number;
  hasDevice: boolean; deviceCountdown: number | null;
}
interface HistoryEntry { round: number; seat: number; snapshot: GameSnapshot; metrics: SeatMetrics[]; }
interface GameSummary {
  id: string; createdAt: string; matchup: string;
  winnerSeat: number | null; winCause: string | null; rounds: number | null;
  map: { width: number; height: number }; humanCount: number;
}
interface FullGame {
  id: string; matchup: string;
  players: Array<{ seat: number; type: string; name: string }>;
  outcome: { winnerSeat: number | null; winCause: string | null; rounds: number | null };
  gameData: { seed: number; history: HistoryEntry[]; finalSnapshot: GameSnapshot; winnerSeat: number | null; winCause: string | null };
}

export interface ReplayDeps {
  game: Phaser.Game;
  /** The #cp-stage element (Phaser canvas host); reparented into the viewer while open. */
  stage: HTMLElement;
  /** Where the stage normally lives (#game); the stage is returned here on close. */
  parent: HTMLElement;
  /** Called once the dashboard has fully closed (e.g. to re-open the start dialog). */
  onExit: () => void;
}

const el = (tag: string, cls?: string, html?: string): HTMLElement => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html != null) e.innerHTML = html;
  return e;
};

const fmtCause: Record<string, string> = {
  conquest: 'Conquest', domination: 'Domination', device: 'Device',
  bankruptcy: 'Bankruptcy', tie: 'Tie', resign: 'Resign', other: 'Other',
};

function fmtDate(iso: string): string {
  // ISO → "9.6.2026 18:16" without pulling in a date lib.
  const m = iso.match(/^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/);
  if (!m) return iso;
  const [, y, mo, d, h, mi] = m;
  return `${Number(d)}.${Number(mo)}.${y} ${h}:${mi}`;
}

export function showReplayDashboard(deps: ReplayDeps): void {
  const { game, stage, parent } = deps;

  // --- DOM scaffold (themed: raised dialog frame, sunken containers) ---------
  const overlay = el('div', 'cp-root cp-overlay cp-replay-overlay');
  const panel = el('div', 'cp-dialog cp-replay');
  panel.innerHTML = `
    <div class="cp-replay-head">
      <h2>Recorded Games</h2>
      <button class="cp-mini" id="cp-rp-close">Back</button>
    </div>
    <div class="cp-replay-body">
      <div class="cp-replay-list cp-container"><div class="cp-replay-list-inner" id="cp-rp-list"></div></div>
      <div class="cp-replay-view">
        <div class="cp-replay-board" id="cp-rp-board"></div>
        <div class="cp-replay-controls cp-container"><div class="cp-replay-controls-inner">
          <div class="cp-replay-title" id="cp-rp-title">Select a game from the list</div>
          <div class="cp-replay-transport">
            <button class="cp-mini" id="cp-rp-prev" disabled>&lt;</button>
            <button class="cp-mini" id="cp-rp-play" disabled>Play</button>
            <button class="cp-mini" id="cp-rp-next" disabled>&gt;</button>
            <input type="range" id="cp-rp-slider" min="0" max="0" value="0" step="1" disabled>
            <span class="cp-replay-turn" id="cp-rp-turn">– / –</span>
          </div>
        </div></div>
        <div class="cp-replay-metrics cp-container"><div class="cp-replay-metrics-inner" id="cp-rp-metrics"></div></div>
      </div>
    </div>
  `;
  overlay.appendChild(panel);
  document.body.appendChild(overlay);

  const $ = <T extends HTMLElement>(id: string) => panel.querySelector(`#${id}`) as T;
  const listEl = $<HTMLDivElement>('cp-rp-list');
  const boardHost = $<HTMLDivElement>('cp-rp-board');
  const titleEl = $<HTMLDivElement>('cp-rp-title');
  const metricsEl = $<HTMLDivElement>('cp-rp-metrics');
  const slider = $<HTMLInputElement>('cp-rp-slider');
  const turnEl = $<HTMLSpanElement>('cp-rp-turn');
  const prevBtn = $<HTMLButtonElement>('cp-rp-prev');
  const nextBtn = $<HTMLButtonElement>('cp-rp-next');
  const playBtn = $<HTMLButtonElement>('cp-rp-play');

  // The board (Phaser canvas) lives in #cp-stage — move it into the viewer.
  boardHost.appendChild(stage);
  stage.style.transform = '';

  // --- replay state ----------------------------------------------------------
  let current: FullGame | null = null;
  let index = 0;
  let replayMenu: MenuController | null = null;
  // Latest-render-wins token: a late GameScene onReady from a superseded render
  // must not restore onto the wrong board.
  let renderToken: object = {};
  let autoplay: number | null = null;

  const stopAutoplay = () => {
    if (autoplay != null) { window.clearInterval(autoplay); autoplay = null; }
    playBtn.textContent = 'Play';
  };

  const fitBoard = (mapW: number, mapH: number) => {
    const availW = Math.max(boardHost.clientWidth - 4, 1);
    const availH = Math.max(boardHost.clientHeight - 4, 1);
    const scale = Math.min(availW / mapW, availH / mapH);
    stage.style.width = `${mapW}px`;
    stage.style.height = `${mapH}px`;
    stage.style.transform = `scale(${scale})`;
    game.scale.refresh();
  };
  let lastDims: { w: number; h: number } | null = null;
  const onResize = () => { if (lastDims) fitBoard(lastDims.w, lastDims.h); };
  window.addEventListener('resize', onResize);

  /** Render the board for `current.gameData.history[i]` on a fresh set of managers. */
  const renderBoard = (i: number) => {
    if (!current) return;
    const snap = current.gameData.history[i].snapshot;
    const s = snap.settings;
    const gsm = GameSettingsManager.fromMapDimensions(s.width, s.height);
    const mapW = gsm.getMapWidth();
    const mapH = gsm.getMapHeight();
    lastDims = { w: mapW, h: mapH };

    const om = new ObjectManager();
    // MenuController must exist for the GameEventHandler, but the replay shows no
    // HUD — give it a detached host so its panel never renders.
    const menu = new MenuController(el('div'), gsm);
    const players: PlayerConfig[] = s.players.map((p) => ({ name: p.name, difficulty: p.difficulty }));
    const pm = new PlayerManager(players, om);
    const eh = new GameEventHandler(om, pm, menu, gsm);
    menu.setEventHandler(eh);
    om.addDALS(eh, menu, gsm);

    if (replayMenu) replayMenu.destroy();
    replayMenu = menu;

    const token = {};
    renderToken = token;

    game.scale.resize(mapW, mapH);
    game.scene.start('GameScene', {
      objectManager: om,
      settings: gsm,
      onTileClick: () => {},
      onUnitClick: () => false,
      onReady: (scene: GameScene) => {
        if (renderToken !== token) return; // superseded by a newer render
        eh.setGameScene(scene);
        om.setGameScene(scene);
        new WorldGenerator().generateMap(s.width, s.height, s.seed, {
          objectManager: om, eventHandler: eh, gameSettings: gsm, scene,
        });
        eh.restoreSnapshot(snap);
        requestAnimationFrame(() => fitBoard(mapW, mapH));
      },
    });
  };

  /** Cheap (Phaser-free) update of the title + slider + per-seat metrics for turn `i`. */
  const updateInfo = (i: number) => {
    if (!current) return;
    const entry = current.gameData.history[i];
    const total = current.gameData.history.length;
    turnEl.textContent = `${i + 1} / ${total}`;
    slider.value = String(i);

    const seatName = (seat: number) =>
      current!.players.find((p) => p.seat === seat)?.name ?? `Player ${seat + 1}`;

    titleEl.innerHTML =
      `<span class="cp-rp-matchup">${escapeHtml(current.matchup)}</span>` +
      ` — round ${entry.round}, <b>${escapeHtml(seatName(entry.seat))}</b> to move`;

    const res = (key: string, v: number) =>
      `<span class="cp-rp-res"><img class="cp-img" src="assets/images/${key}.png" alt="${key}">${v}</span>`;
    metricsEl.innerHTML = entry.metrics
      .slice()
      .sort((a, b) => a.seat - b.seat)
      .map((m) => {
        const dot = `<img class="cp-rp-dot" src="assets/images/${COLOR_BALL[m.seat] ?? 'neutral'}.png" alt="">`;
        const device = m.hasDevice
          ? `<span class="cp-rp-device">Device ${m.deviceCountdown ?? '?'}</span>` : '';
        return `<div class="cp-rp-seat">
          <div class="cp-rp-seat-head">${dot}<span class="cp-rp-seat-name">${escapeHtml(seatName(m.seat))}</span>${device}</div>
          <div class="cp-rp-seat-stats">
            ${res('money', m.money)}${res('wood', m.wood)}${res('stone', m.stone)}${res('metal', m.metal)}
            <span class="cp-rp-res"><img class="cp-img" src="assets/images/soldier_1.png" alt="soldier">${m.soldiers}</span>
            <span class="cp-rp-stat">Tiles ${m.tiles}</span>
            <span class="cp-rp-stat">Bld. ${m.buildings}</span>
          </div>
        </div>`;
      })
      .join('');
  };

  const goTo = (i: number, withBoard: boolean) => {
    if (!current) return;
    index = Math.max(0, Math.min(current.gameData.history.length - 1, i));
    updateInfo(index);
    if (withBoard) renderBoard(index);
    prevBtn.disabled = index <= 0;
    nextBtn.disabled = index >= current.gameData.history.length - 1;
  };

  prevBtn.addEventListener('click', () => { stopAutoplay(); goTo(index - 1, true); });
  nextBtn.addEventListener('click', () => { stopAutoplay(); goTo(index + 1, true); });
  // Live drag: update the (cheap) info only; render the board on release.
  slider.addEventListener('input', () => updateInfo(Number(slider.value)));
  slider.addEventListener('change', () => { stopAutoplay(); goTo(Number(slider.value), true); });
  playBtn.addEventListener('click', () => {
    if (!current) return;
    if (autoplay != null) { stopAutoplay(); return; }
    if (index >= current.gameData.history.length - 1) goTo(0, true);
    playBtn.textContent = 'Pause';
    autoplay = window.setInterval(() => {
      if (!current || index >= current.gameData.history.length - 1) { stopAutoplay(); return; }
      goTo(index + 1, true);
    }, AUTOPLAY_MS);
  });

  // --- load a game into the viewer ------------------------------------------
  const loadGame = async (id: string, card: HTMLElement) => {
    stopAutoplay();
    listEl.querySelectorAll('.cp-rp-card.cp-rp-sel').forEach((c) => c.classList.remove('cp-rp-sel'));
    card.classList.add('cp-rp-sel');
    titleEl.textContent = 'Loading…';
    metricsEl.innerHTML = '';
    try {
      const res = await fetch(`${SERVER_URL}/api/games/${encodeURIComponent(id)}`);
      const data = await res.json();
      const game_: FullGame = data.game;
      if (!game_?.gameData?.history?.length) {
        titleEl.textContent = 'This game has no turn history.';
        slider.disabled = prevBtn.disabled = nextBtn.disabled = playBtn.disabled = true;
        return;
      }
      current = game_;
      slider.min = '0';
      slider.max = String(game_.gameData.history.length - 1);
      slider.disabled = playBtn.disabled = false;
      goTo(0, true);
    } catch {
      titleEl.textContent = 'Failed to load game.';
    }
  };

  // --- fetch + render the game list -----------------------------------------
  const loadList = async () => {
    listEl.innerHTML = `<div class="cp-rp-empty">Loading games…</div>`;
    try {
      const res = await fetch(`${SERVER_URL}/api/games?limit=100`);
      const data = await res.json();
      const games: GameSummary[] = data.games ?? [];
      if (!games.length) {
        listEl.innerHTML = `<div class="cp-rp-empty">No recorded games yet.<br><br>Play a game as a human against the AI — it will be saved here.</div>`;
        return;
      }
      listEl.innerHTML = '';
      for (const g of games) {
        const cause = g.winCause ? (fmtCause[g.winCause] ?? g.winCause) : '–';
        const card = el('div', 'cp-rp-card');
        card.innerHTML =
          `<div class="cp-rp-card-matchup">${escapeHtml(g.matchup)}</div>` +
          `<div class="cp-rp-card-meta">${cause} · ${g.rounds ?? '?'} rounds · ${g.map.width}×${g.map.height}</div>` +
          `<div class="cp-rp-card-date">${fmtDate(g.createdAt)}</div>`;
        card.addEventListener('click', () => { void loadGame(g.id, card); });
        listEl.appendChild(card);
      }
    } catch {
      listEl.innerHTML =
        `<div class="cp-rp-empty">Could not reach the server.<br><br>` +
        `<span style="color:#9a9a9a">${escapeHtml(SERVER_URL)}</span></div>`;
    }
  };

  // --- teardown --------------------------------------------------------------
  const close = () => {
    stopAutoplay();
    renderToken = {}; // invalidate any in-flight onReady
    window.removeEventListener('resize', onResize);
    if (replayMenu) { replayMenu.destroy(); replayMenu = null; }
    game.scene.stop('GameScene');
    // Return the board to its normal home.
    stage.style.transform = '';
    parent.appendChild(stage);
    overlay.remove();
    deps.onExit();
  };
  $<HTMLButtonElement>('cp-rp-close').addEventListener('click', close);

  void loadList();
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]!));
}
