// DOM reimplementation of MenuObjectManager + MenuView. Renders the right-hand
// menu panel: same content, fonts, colours, images and button actions as the
// original QGraphics menu, on a 16px cell grid.

import { IMenuObjectManager } from '../managers/menu-interface';
import { GameEventHandler } from '../managers/gameeventhandler';
import { IGameSettingsManager } from '../model/base';
import { TileBase } from '../model/tile';
import { PlayerBase } from '../model/player';
import { BasicResource, ResourceMap } from '../core/resources';
import * as R from '../core/resources';
import * as Desc from '../core/descriptions';

const CELL = 16;
// The 9-slice frame is drawn by a ::before pseudo-element that overlays the box, so
// content is positioned from the element's outer corner (no layout border to offset).
const CONTAINER_BORDER = 0;
const COLOR_BALL = ['red', 'blue', 'purple', 'yellow'];
const COLOR_BAR = ['color_bar_red', 'color_bar_blue', 'color_bar_purple', 'color_bar_yellow'];
const RES_ICON: Record<number, string> = {
  [BasicResource.MONEY]: 'money',
  [BasicResource.WOOD]: 'wood',
  [BasicResource.STONE]: 'stone',
  [BasicResource.METAL]: 'metal',
};
const RES_ORDER = [BasicResource.MONEY, BasicResource.WOOD, BasicResource.STONE, BasicResource.METAL];
const RES_NAME: Record<number, string> = {
  [BasicResource.MONEY]: 'Money',
  [BasicResource.WOOD]: 'Wood',
  [BasicResource.STONE]: 'Stone',
  [BasicResource.METAL]: 'Metal',
};

const BUILD_INFO: Record<string, { cost: ResourceMap; prod: ResourceMap; shop: string }> = {
  Farm: { cost: R.FARM_BUILD_COST, prod: R.FARM_PRODUCTION, shop: Desc.FARM_SHOP_DESCRIPTION },
  Village: { cost: R.VILLAGE_BUILD_COST, prod: R.VILLAGE_PRODUCTION, shop: Desc.VILLAGE_SHOP_DESCRIPTION },
  Outpost: { cost: R.OUTPOST_BUILD_COST, prod: R.OUTPOST_PRODUCTION, shop: Desc.OUTPOST_SHOP_DESCRIPTION },
  'Nuclear Power Plant': { cost: R.NUCLEARPP_BUILD_COST, prod: R.NUCLEARPP_PRODUCTION, shop: Desc.NUCLEAR_SHOP_DESCRIPTION },
  Mine: { cost: R.MINE_BUILD_COST, prod: R.MINE_PRODUCTION, shop: Desc.MINE_SHOP_DESCRIPTION },
  'Hydroelectric Power Plant': { cost: R.HEPP_BUILD_COST, prod: R.HEPP_PRODUCTION, shop: Desc.HEPP_SHOP_DESCRIPTION },
  Bridge: { cost: R.BRIDGE_BUILD_COST, prod: R.BRIDGE_PRODUCTION, shop: Desc.BRIDGE_SHOP_DESCRIPTION },
  'Strange Device': { cost: R.STRANGE_DEVICE_BUILD_COST, prod: R.NO_RESOURCES, shop: Desc.STRANGE_DEVICE_SHOP_DESCRIPTION },
};

const BUILD_PREVIEW: Record<string, string> = {
  Farm: 'farm1',
  Village: 'village',
  Outpost: 'outpost_1',
  'Nuclear Power Plant': 'nuclearPlant1',
  Mine: 'mine',
  'Hydroelectric Power Plant': 'hydropower1NS',
  Bridge: 'bridgeNS',
  'Strange Device': 'strange_device',
};

function abbreviate(name: string): string {
  if (name === 'Nuclear Power Plant') return 'Nuclear PP.';
  if (name === 'Hydroelectric Power Plant') return 'Hydroelectric PP.';
  return name;
}
function amount(map: ResourceMap, res: BasicResource): number {
  return map.get(res) ?? 0;
}

/** Height (in 16px cells) of the persistent control bar pinned to the panel bottom. */
const FOOTER_CELLS = 3;

export class MenuController implements IMenuObjectManager {
  private eh!: GameEventHandler;
  private settings: IGameSettingsManager;
  private root: HTMLElement;
  /** Swappable view layer (cleared on every view change). */
  private view: HTMLElement;
  /** Persistent control bar (END TURN + MENU), present in every view. */
  private footer: HTMLElement;
  private endTurnBtn!: HTMLElement;
  onHelp: (() => void) | null = null;
  onQuit: (() => void) | null = null;

  constructor(parent: HTMLElement, settings: IGameSettingsManager) {
    this.settings = settings;
    this.root = document.createElement('div');
    this.root.className = 'cp-root cp-menu';
    this.root.style.left = `${settings.getMapWidth()}px`;
    this.root.style.width = `${settings.getMenuWidth()}px`;
    this.root.style.height = `${settings.getMapHeight()}px`;
    parent.appendChild(this.root);

    const footerPx = FOOTER_CELLS * CELL;
    // The swappable views fill the whole panel and keep their original 1:1 cell
    // coordinates (the view layer starts at 0,0). The control bar floats over the
    // bottom strip — the per-view layouts leave that strip empty, so nothing is
    // clipped and tall views never need a scrollbar.
    this.view = document.createElement('div');
    this.view.style.position = 'absolute';
    this.view.style.inset = '0';
    this.view.style.overflow = 'hidden';
    this.root.appendChild(this.view);

    this.footer = document.createElement('div');
    this.footer.className = 'cp-footer';
    this.footer.style.position = 'absolute';
    this.footer.style.left = '0';
    this.footer.style.right = '0';
    this.footer.style.bottom = '0';
    this.footer.style.height = `${footerPx}px`;
    this.footer.style.zIndex = '2';
    this.root.appendChild(this.footer);
    this.buildFooter();
  }

  setEventHandler(eh: GameEventHandler): void {
    this.eh = eh;
  }

  destroy(): void {
    this.root.remove();
  }

  /** The always-visible MENU (new game) + END TURN bar. Built once, never cleared. */
  private buildFooter(): void {
    const menuBtn = document.createElement('div');
    menuBtn.className = 'cp-btn cp-footer-menu';
    menuBtn.style.fontSize = '9px';
    menuBtn.innerHTML = '<span class="cp-lbl">MENU</span>';
    menuBtn.addEventListener('click', () => this.confirmQuit());

    const end = document.createElement('div');
    end.className = 'cp-btn cp-footer-end';
    end.style.fontSize = '12px';
    end.innerHTML = '<span class="cp-lbl">END TURN</span>';
    end.addEventListener('click', () => {
      if (end.classList.contains('cp-disabled')) return;
      this.eh.endTurn();
    });
    this.endTurnBtn = end;

    this.footer.appendChild(menuBtn);
    this.footer.appendChild(end);
  }

  /** Enable END TURN only on a human player's own active turn. */
  private setEndTurnEnabled(enabled: boolean): void {
    this.endTurnBtn.classList.toggle('cp-disabled', !enabled);
  }

  /** Styled "discard the current game?" confirmation before quitting to the menu. */
  private confirmQuit(): void {
    const overlay = document.createElement('div');
    overlay.className = 'cp-root cp-overlay';
    const dialog = document.createElement('div');
    dialog.className = 'cp-dialog';
    dialog.style.textAlign = 'center';
    dialog.innerHTML =
      `<h2>New Game?</h2>` +
      `<p style="font-size:9px;line-height:1.7;margin:0 0 4px">This will end the current game and return to the start menu.</p>` +
      `<div class="cp-actions"><button id="cp-confirm-yes" class="cp-primary">New Game</button><button id="cp-confirm-no">Keep Playing</button></div>`;
    overlay.appendChild(dialog);
    document.body.appendChild(overlay);
    const close = () => overlay.remove();
    dialog.querySelector('#cp-confirm-yes')!.addEventListener('click', () => {
      close();
      this.onQuit?.();
    });
    dialog.querySelector('#cp-confirm-no')!.addEventListener('click', close);
    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) close();
    });
  }

  private get player(): PlayerBase {
    return this.eh.getPlayerManager().getCurrentPlayer();
  }

  // --- low-level builders ---------------------------------------------------

  /** Clear the swappable view layer (the persistent footer survives) and set whether
   *  END TURN is pressable for the view about to be drawn. */
  private reset(endTurnEnabled = true): HTMLElement {
    this.view.innerHTML = '';
    this.view.scrollTop = 0;
    this.setEndTurnEnabled(endTurnEnabled);
    return this.view;
  }

  private place(el: HTMLElement, x: number, y: number, w: number, h: number): HTMLElement {
    el.style.left = `${x * CELL}px`;
    el.style.top = `${y * CELL}px`;
    el.style.width = `${w * CELL}px`;
    el.style.height = `${h * CELL}px`;
    return el;
  }

  private container(parent: HTMLElement, x: number, y: number, w: number, h: number): HTMLElement {
    const frame = document.createElement('div');
    frame.className = 'cp-el cp-container';
    this.place(frame, x, y, w, h);
    parent.appendChild(frame);
    // Inner content layer aligned to the frame's BORDER box, so child cell
    // coordinates match the original (which measures from the container corner,
    // not inside the 9-patch border).
    const inner = document.createElement('div');
    inner.style.position = 'absolute';
    inner.style.left = `${-CONTAINER_BORDER}px`;
    inner.style.top = `${-CONTAINER_BORDER}px`;
    inner.style.width = `${w * CELL}px`;
    inner.style.height = `${h * CELL}px`;
    inner.style.zIndex = '1'; // above the ::before 9-slice frame
    frame.appendChild(inner);
    return inner;
  }

  private label(
    parent: HTMLElement,
    x: number,
    y: number,
    w: number,
    h: number,
    html: string,
    align: 'left' | 'left-center' | 'center' = 'left',
    fontSize = 10,
    color = 'rgb(200,200,200)',
  ): HTMLElement {
    const d = document.createElement('div');
    d.className = `cp-el cp-label ${align}`;
    d.style.fontSize = `${fontSize}px`;
    d.style.color = color;
    d.style.padding = '3px';
    this.place(d, x, y, w, h);
    d.innerHTML = html;
    parent.appendChild(d);
    return d;
  }

  private button(
    parent: HTMLElement,
    x: number,
    y: number,
    w: number,
    h: number,
    text: string,
    fontSize: number,
    onClick: () => void,
    disabled = false,
  ): HTMLElement {
    const d = document.createElement('div');
    d.className = 'cp-el cp-btn' + (disabled ? ' cp-disabled' : '');
    d.style.fontSize = `${fontSize}px`;
    this.place(d, x, y, w, h);
    // Label sits above the 9-slice frame drawn by the ::before pseudo-element.
    d.innerHTML = `<span class="cp-lbl">${text}</span>`;
    if (!disabled) d.addEventListener('click', onClick);
    parent.appendChild(d);
    return d;
  }

  /** A thin horizontal divider line (matches the original menu's section separators). */
  private divider(parent: HTMLElement, x: number, y: number, w: number): void {
    const d = document.createElement('div');
    d.className = 'cp-el cp-divider';
    d.style.left = `${x * CELL}px`;
    d.style.top = `${y * CELL}px`;
    d.style.width = `${w * CELL}px`;
    d.style.height = '2px';
    parent.appendChild(d);
  }

  private icon(parent: HTMLElement, x: number, y: number, w: number, h: number, key: string): HTMLElement {
    const img = document.createElement('img');
    img.className = 'cp-el cp-img';
    img.src = `assets/images/${key}.png`;
    this.place(img as unknown as HTMLElement, x, y, w, h);
    parent.appendChild(img);
    return img;
  }

  private colorBall(parent: HTMLElement, x: number, y: number, playerNum: number): void {
    this.icon(parent, x, y, 2, 2, COLOR_BALL[playerNum - 1]);
  }

  private playerTitle(parent: HTMLElement): void {
    const p = this.player;
    this.colorBall(parent, 1, 1, p.getPlayerNum());
    this.label(parent, 3, 1, 18, 2, p.getName(), 'left-center', 16);
  }

  private closeButton(parent: HTMLElement): void {
    this.button(parent, 19, 1, 2, 2, 'X', 10, () => this.eh.openDefaultMenuView());
  }

  private resourceMenu(parent: HTMLElement, x: number, y: number): void {
    const c = this.container(parent, x, y, 20, 7);
    const res = this.player.getResources();
    const cells: Array<[BasicResource, number, number]> = [
      [BasicResource.MONEY, 1, 1],
      [BasicResource.WOOD, 10, 1],
      [BasicResource.STONE, 1, 4],
      [BasicResource.METAL, 10, 4],
    ];
    for (const [r, cx, cy] of cells) {
      this.icon(c, cx, cy, 2, 2, RES_ICON[r]);
      this.label(c, cx + 2, cy, 7, 2, String(amount(res, r)), 'left-center', 14);
    }
  }

  /**
   * A titled resource readout for any player (used to spy on an enemy HQ). Lays the
   * resources out exactly like the main {@link resourceMenu} (money/wood on top,
   * stone/metal below, value right of each icon), with a colour-ball + title row
   * above — all sitting cleanly inside the card frame.
   */
  private resourceMenuFor(parent: HTMLElement, x: number, y: number, player: PlayerBase, title: string): void {
    const c = this.container(parent, x, y, 20, 11);
    this.colorBall(c, 1, 1, player.getPlayerNum());
    this.label(c, 4, 1, 15, 2, title, 'left-center', 9);
    const res = player.getResources();
    const cells: Array<[BasicResource, number, number]> = [
      [BasicResource.MONEY, 1, 4],
      [BasicResource.WOOD, 10, 4],
      [BasicResource.STONE, 1, 7],
      [BasicResource.METAL, 10, 7],
    ];
    for (const [r, cx, cy] of cells) {
      this.icon(c, cx, cy, 2, 2, RES_ICON[r]);
      this.label(c, cx + 2, cy, 7, 2, String(amount(res, r)), 'left-center', 14);
    }
  }

  private netMenu(parent: HTMLElement, x: number, y: number): void {
    const c = this.container(parent, x, y, 20, 11);
    const rev = this.eh.getCurrentRevenue();
    const exp = this.eh.getCurrentExpences();
    const net = this.eh.getCurrentNet();
    this.label(c, 3, 0, 5, 2, 'Revenue', 'center', 8);
    this.label(c, 8, 0, 5, 2, 'Expenses', 'center', 8);
    this.label(c, 13, 0, 6, 2, '<u>Net</u>', 'center', 10);
    let yy = 2;
    for (const r of RES_ORDER) {
      this.icon(c, 1, yy, 2, 2, RES_ICON[r]);
      const rv = amount(rev, r);
      const ev = amount(exp, r);
      const nv = amount(net, r);
      this.label(c, 3, yy, 5, 2, rv !== 0 ? String(rv) : '-', 'center', 10);
      this.label(c, 8, yy, 5, 2, ev !== 0 ? String(ev) : '-', 'center', 10);
      this.label(c, 13, yy, 6, 2, nv !== 0 ? String(nv) : '-', 'center', 10);
      yy += 2;
    }
  }

  private unitMenu(parent: HTMLElement, x: number, y: number): void {
    const c = this.container(parent, x, y, 20, 9);
    const p = this.player;
    this.label(c, 1, 0, 12, 3, `(${p.getCurrentUnitAmount()}/${p.getMaxUnitAmount()})`, 'center', 10);
    this.label(c, 13, 0, 6, 3, `(${p.getCurrentSoldierAmount()}/${p.getMaxSoldierAmount()})`, 'center', 10);
    this.icon(c, 3, 3, 2, 3, 'basicworker_1');
    this.icon(c, 9, 3, 2, 3, 'expert_1');
    this.icon(c, 15, 3, 2, 3, 'soldier_1');
    this.label(c, 1, 6, 6, 2, `x${p.getCurrentBasicWorkerAmount()}`, 'center', 10);
    this.label(c, 7, 6, 6, 2, `x${p.getCurrentExpertAmount()}`, 'center', 10);
    this.label(c, 13, 6, 6, 2, `x${p.getCurrentSoldierAmount()}`, 'center', 10);
  }

  // --- IMenuObjectManager views --------------------------------------------

  selectFirstTileMenuView(player: PlayerBase): void {
    const root = this.reset(false); // must place the HQ before ending the turn
    this.colorBall(root, 1, 1, player.getPlayerNum());
    this.label(root, 3, 1, 18, 2, player.getName(), 'left-center', 16);
    const c = this.container(root, 1, 4, 20, 16);
    this.label(
      c,
      1,
      1,
      18,
      12,
      `${player.getName()} choose your starting tile! Starting tile must be a grassland. ` +
        `You will also get all of the other tiles next to the chosen tile. Choose carefully.<br><br>Good luck!`,
      'left',
      12,
    );
  }

  setDefaultMenuView(): void {
    const root = this.reset();
    this.playerTitle(root);
    this.resourceMenu(root, 1, 4);
    this.divider(root, 1, 11.5, 20);
    this.button(root, 1, 12, 6, 3, 'UNIT<br>SHOP', 10, () => this.eh.openUnitBuyMenu());
    this.button(root, 8, 12, 6, 3, 'STATS', 10, () => this.eh.openStatsMenuView());
    this.button(root, 15, 12, 6, 3, 'HELP', 10, () => this.onHelp?.());
    this.netMenu(root, 1, 16);
    this.unitMenu(root, 1, 28);
    // END TURN now lives in the persistent footer bar (always reachable).
  }

  setCpuTurnMenuView(player: PlayerBase): void {
    const root = this.reset(false); // not the human's turn
    this.colorBall(root, 1, 1, player.getPlayerNum());
    this.label(root, 3, 1, 18, 2, player.getName(), 'left-center', 16);
    const diff = player.getDifficulty();
    const diffLabel = diff.charAt(0).toUpperCase() + diff.slice(1);
    this.label(root, 1, 3, 20, 2, `Computer player &middot; ${diffLabel}`, 'center', 9, 'rgb(150,150,150)');
    this.resourceMenu(root, 1, 6);
    const c = this.container(root, 1, 14, 20, 6);
    this.label(c, 1, 1, 18, 4, 'Thinking&hellip;<br>Watch the map.', 'center', 12);
  }

  setUnitShopMenuView(): void {
    const root = this.reset();
    this.label(root, 3, 1, 18, 2, this.player.getName(), 'left-center', 16);
    this.closeButton(root);
    this.resourceMenu(root, 1, 4);
    this.divider(root, 1, 11.5, 20);
    const c = this.container(root, 1, 12, 20, 26);

    const units: Array<[string, string, string, number, number]> = [
      ['WORKER', 'BasicWorker', 'basicworker_1', 1, 0],
      ['EXPERT', 'Expert', 'expert_1', 9, 8],
      ['SOLDIER', 'Soldier', 'soldier_1', 17, 16],
    ];
    const p = this.player;
    const costFor = (t: string) => (t === 'BasicWorker' ? R.BASIC_WORKER_COST : t === 'Expert' ? R.EXPERT_COST : R.SOLDIER_COST);
    const salaryFor = (t: string) =>
      t === 'BasicWorker' ? R.BASIC_WORKER_SALARY : t === 'Expert' ? R.EXPERT_SALARY : R.SOLDIER_SALARY;
    const desc: Record<string, string> = {
      BasicWorker: 'A worker can work in mines, farms and power plants. He can also cut down forests.',
      Expert: 'An expert can work in power plants and mines. He boosts efficiency a lot.',
      Soldier: 'A soldier can defend your area and sometimes even conquer tiles from other players.',
    };
    for (const [label, type, iconKey, , relY] of units) {
      this.label(c, 1, relY + 1, 6, 2, label, 'center', 12);
      this.icon(c, 3, relY + 3, 2, 3, iconKey);
      const cost = costFor(type);
      const free = type === 'Soldier' ? p.getFreeSoldierAmount() : p.getFreeUnitAmount();
      const canBuy = p.hasEnoughResources(cost) && free > 0;
      this.button(c, 2, relY + 6, 4, 2, 'BUY', 12, () => this.eh.createUnit(type), !canBuy);
      // Each resource is coloured red only when this player can't afford that one.
      const costStr = `<u>Cost:</u>${type === 'Soldier' ? '<br>' : ' '}${this.coloredCost(cost, 'Coins')}`;
      const salaryStr = `<u>Salary:</u> ${-amount(salaryFor(type), BasicResource.MONEY)} Coins/round`;
      this.label(c, 7, relY + 2, 12, 7, `${costStr}<br>${salaryStr}<br><br>${desc[type]}`, 'left', 8);
    }
  }

  setStatMenuView(): void {
    const root = this.reset();
    this.playerTitle(root);
    this.closeButton(root);
    const rc = this.container(root, 1, 4, 20, 4);
    this.label(rc, 1, 1, 18, 2, `Rounds played: ${this.eh.getPlayerManager().getRoundsPlayed()}`, 'left-center', 12);
    this.label(root, 1, 9, 18, 2, 'Tiles owned:', 'left-center', 14);
    const om = this.eh.getObjectManager();
    const total = om.getTileCount();
    const players = this.eh.getPlayerManager().getPlayers();
    const c = this.container(root, 1, 11, 20, players.length > 2 ? 10 : 7);
    const positions: Array<[number, number]> = [
      [1, 1],
      [10, 1],
      [1, 4],
      [10, 4],
    ];
    players.forEach((pl, i) => {
      const [cx, cy] = positions[i];
      this.colorBall(c, cx, cy, pl.getPlayerNum());
      const pct = total > 0 ? Math.trunc((om.getTileCountForPlayer(pl) * 100) / total) : 0;
      this.label(c, cx + 2, cy, 7, 2, `${pct}%`, 'left-center', 14);
    });
    const neutralPct = total > 0 ? Math.trunc((om.getNeutralTiles() * 100) / total) : 0;
    this.label(c, 1, players.length > 2 ? 7 : 4, 18, 2, `Neutral tiles: ${neutralPct}%`, 'left-center', 10);
  }

  setTileInspectionMenuView(tile: TileBase, indexForBuildings = 0): void {
    const root = this.reset();
    this.playerTitle(root);
    this.closeButton(root);

    const building = tile.getBuilding();
    const titleName = abbreviate(building ? building.getType() : tile.getType());

    const c = this.container(root, 1, 4, 20, 14);
    // tile image + owner bar
    const terrainKey = tile.getImageFiles()[0] ?? 'grassland';
    this.icon(c, 15, 1, 4, 4, terrainKey);
    if (building) this.icon(c, 15, 1, 4, 4, building.getImageFiles()[0] ?? terrainKey);
    this.icon(c, 15, 1, 4, 4, 'tile_cover_border');
    const owner = tile.getOwner();
    this.icon(c, 15, 5, 4, 2, owner ? COLOR_BAR[owner.getPlayerNum() - 1] : 'color_bar_neutral');

    this.label(c, 1, 0, 14, 2, `<u>${titleName}</u>`, 'left-center', 12);
    // A building tile shows the building's own description (Mikontalo, Farm, HQ…),
    // not the description of the terrain underneath it — matches menuobjectmanager.cpp.
    const basicDescription = building ? building.getBasicDescription() : tile.getBasicDescription();
    this.label(c, 1, 2, 14, 6, basicDescription, 'left', 8);

    const net = tile.getNetDescription();
    const extra = tile.getExtraDescription();
    if (net) this.label(c, 1, 8, 9, 5, net, 'left', 8);
    if (extra) this.label(c, 10, 8, 9, 5, extra, 'left', 8);

    let yBuild = 19;
    // Units present.
    const showUnits = owner === this.player ? tile.getUnits() : tile.getConqueringUnits();
    if (showUnits.length > 0) {
      const uc = this.container(root, 1, 19, 20, 9);
      showUnits.forEach((unit, idx) => {
        const baseX = 3 + idx + 4 * idx;
        const iconKey = unit.getImageFiles()[0] ?? 'basicworker_1';
        this.icon(uc, baseX + 1, 1, 2, 3, iconKey);
        this.button(uc, baseX, 4, 4, 2, 'MOVE', 8, () => this.eh.moveUnitFromTile(idx, tile));
        this.button(uc, baseX, 6, 4, 2, 'DEL', 8, () => this.eh.deleteUnitFromTileByIndex(idx, tile));
      });
      yBuild = 29;
    }

    // Buildable buildings (only on own, building-less tiles).
    if (owner === this.player && building === null) {
      const buildable = tile.getBuildableBuildings();
      if (buildable.length > 0) {
        const idx = ((indexForBuildings % buildable.length) + buildable.length) % buildable.length;
        const name = buildable[idx];
        const info = BUILD_INFO[name];
        const bc = this.container(root, 1, yBuild, 20, 13);
        const arrowColor = buildable.length > 1 ? 'rgb(200,200,200)' : 'rgb(80,80,80)';
        this.button(bc, 1, 1, 3, 2, '&lt;', 10, () => this.eh.setTileInspectionMenuView(tile, idx - 1), buildable.length <= 1);
        this.label(bc, 4, 1, 12, 2, abbreviate(name), 'center', 10, arrowColor);
        this.button(bc, 16, 1, 3, 2, '&gt;', 10, () => this.eh.setTileInspectionMenuView(tile, idx + 1), buildable.length <= 1);

        // preview image: terrain + building sprite + cover border (as in the original)
        const previewTerrain = tile.getType() === 'River' ? terrainKey : tile.getType() === 'Mountain' ? 'mountain' : 'grassland';
        this.icon(bc, 1, 4, 4, 4, previewTerrain);
        const previewBuilding = BUILD_PREVIEW[name];
        if (previewBuilding) this.icon(bc, 1, 4, 4, 4, previewBuilding);
        this.icon(bc, 1, 4, 4, 4, 'tile_cover_border');

        if (info) {
          const canAfford = this.player.hasEnoughResources(info.cost);
          this.label(bc, 5, 4, 6, 8, this.buildInfoHtml(name, info), 'left', 8);
          this.label(bc, 11, 4, 8, 8, info.shop, 'left', 8);
          this.button(bc, 1, 8, 4, 4, 'BUY', 12, () => this.eh.buildBuilding(name, tile), !canAfford);
        }
      }
    } else if (building?.getType() === 'Headquarters' && owner && owner !== this.player) {
      // Spy on a rival by clicking their headquarters: show their resources.
      this.resourceMenuFor(root, 1, yBuild, owner, `${owner.getName()}'s resources`);
    }
  }

  /**
   * Render a cost with each resource coloured individually: white when the current
   * player can afford that specific resource, red only for the ones they're short on.
   * (Previously the whole cost line went red, which hid *which* resource was lacking.)
   */
  private coloredCost(cost: ResourceMap, moneyLabel = 'Money'): string {
    const res = this.player.getResources();
    const parts: string[] = [];
    for (const r of RES_ORDER) {
      const v = amount(cost, r);
      if (v >= 0) continue; // not a cost for this resource
      const need = -v;
      const color = amount(res, r) >= need ? 'rgb(200,200,200)' : 'rgb(255,80,80)';
      const label = r === BasicResource.MONEY ? moneyLabel : RES_NAME[r];
      parts.push(`<span style="color:${color}">${need} ${label}</span>`);
    }
    return parts.join(', ');
  }

  private buildInfoHtml(name: string, info: { cost: ResourceMap; prod: ResourceMap }): string {
    const cost = `<u>Cost:</u><br>${this.coloredCost(info.cost)}`;
    // The Strange Device has no per-round product — its "effect" is the win countdown and
    // the soldier-cap penalty, so we surface that instead of an empty Products list.
    if (name === 'Strange Device') {
      return `${cost}<br><u>Effect:</u><br>Win if it survives.<br>−2 soldier cap.`;
    }
    let prod = `<br><u>Products:</u><br>`;
    if (name === 'Farm') {
      prod += `${amount(info.prod, BasicResource.MONEY)} Money every ${R.FARM_GROW_TIME} rounds`;
    } else {
      const prodParts: string[] = [];
      for (const r of RES_ORDER) {
        const v = amount(info.prod, r);
        if (v !== 0) prodParts.push(`${v} ${RES_NAME[r]}/r`);
      }
      prod += prodParts.join('<br>');
      if (name === 'Mine' || name === 'Nuclear Power Plant' || name === 'Hydroelectric Power Plant') {
        prod += '<br>(for each unit)';
      }
    }
    return `${cost}${prod}`;
  }

  setWinMenu(player: PlayerBase): void {
    const root = this.reset(false);
    const c = this.container(root, 1, 10, 20, 9);
    this.colorBall(c, 1, 1, player.getPlayerNum());
    this.label(c, 3, 1, 16, 9, `${player.getName()} is the winner!<br><br>Congratulations!`, 'left', 12);
    this.button(c, 3, 11, 6, 3, 'New Game', 10, () => this.eh.restartGame());
    this.button(c, 11, 11, 6, 3, 'Quit', 10, () => this.onQuit?.());
  }

  setPlayerLostMenu(players: PlayerBase[], reasons: string[]): void {
    const root = this.reset(false);
    const yOffset = 10;
    players.forEach((player, i) => {
      const c = this.container(root, 1, 10 + yOffset * i, 20, 9);
      this.colorBall(c, 1, 1, player.getPlayerNum());
      this.label(c, 3, 1, 16, 3, `${player.getName()} lost the game.`, 'left', 12);
      const reasonText = reasons[i] === 'noresources' ? 'Player ran out of resources.' : 'Players headquarters got conquered.';
      this.label(c, 3, 4, 16, 5, `<u>Reason:</u><br>${reasonText}`, 'left', 8);
    });
    this.button(root, 8, 10 + yOffset * players.length, 6, 3, 'OK', 14, () => this.eh.openDefaultMenuView());
  }

  setTieMenu(players: PlayerBase[], reasons: string[]): void {
    const root = this.reset(false);
    this.label(root, 1, 1, 20, 3, 'It is a tie.', 'center', 14);
    const yOffset = 9;
    players.forEach((player, i) => {
      const c = this.container(root, 1, 4 + yOffset * i, 20, 8);
      this.colorBall(c, 1, 1, player.getPlayerNum());
      this.label(c, 3, 1, 16, 3, `${player.getName()} lost the game.`, 'left', 12);
      const reasonText = reasons[i] === 'noresources' ? 'Player ran out of resources.' : 'Players headquarters got conquered.';
      this.label(c, 3, 4, 16, 4, `<u>Reason:</u><br>${reasonText}`, 'left', 8);
    });
    this.button(root, 4, 4 + yOffset * players.length, 6, 3, 'New Game', 10, () => this.eh.restartGame());
    this.button(root, 13, 4 + yOffset * players.length, 6, 3, 'Quit', 10, () => this.onQuit?.());
  }
}
