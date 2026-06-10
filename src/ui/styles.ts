// Injected CSS for the DOM-rendered menu panel, start dialog and help window.
//
// Faithful to the original Qt build: the menu was drawn from a 9-slice sprite set
// (multi_0..8.png, 8px pieces) tiled at the 16px grid — raised frames for buttons,
// sunken (180°-rotated) frames for containers. We compose those pieces into two
// sheets (multi_frame.png / multi_frame_inv.png) and render them with CSS
// border-image at 2× (16px) with pixelated scaling, exactly matching the original's
// chunky, angular pixel look. The frame is painted by a ::before layer so text and
// child cells overlay the full box (as the original paints text over the rect).

let injected = false;

const FILL = 'rgb(31,31,31)'; // the 9-slice centre colour (#1f1f1f)
const PANEL_BG = '#151515'; // panel background, a touch darker so cards read clearly
const TEXT = 'rgb(208,208,208)';
const RAISED = 'url(assets/images/multi_frame.png) 8 fill repeat'; // buttons (light top-left)
const SUNKEN = 'url(assets/images/multi_frame_inv.png) 8 fill repeat'; // containers (recessed)

export function injectStyles(): void {
  if (injected) return;
  injected = true;
  const css = `
  /* Keep the Phaser canvas pixel-crisp when the stage is scaled up to fill big screens. */
  #cp-stage canvas { image-rendering: pixelated; }
  .cp-root, .cp-root * { box-sizing: border-box; font-family: 'PressStart2P', monospace; }
  .cp-menu {
    position: absolute; top: 0;
    background: ${PANEL_BG};
    color: ${TEXT}; overflow: hidden;
  }
  .cp-el { position: absolute; }
  .cp-img { image-rendering: pixelated; }

  /* Shared 9-slice frame, drawn behind content by a ::before layer. The host must be
     positioned for the ::before's inset to anchor to it; menu cards/buttons already
     are via .cp-el (absolute), so we must NOT re-declare position here (it would
     override .cp-el and drop them into normal flow). The footer's buttons aren't
     .cp-el, so they get position below. */
  .cp-container, .cp-btn { background: transparent; }
  .cp-footer .cp-btn { position: relative; }
  .cp-container::before, .cp-btn::before {
    content: ''; position: absolute; inset: 0; z-index: 0; pointer-events: none;
    border: 16px solid transparent; image-rendering: pixelated;
  }
  .cp-container::before { border-image: ${SUNKEN}; } /* in-game HUD panels: recessed/non-raised */
  .cp-btn::before { border-image: ${RAISED}; }      /* buttons: raised */

  .cp-divider { background: #0c0c0c; box-shadow: 0 1px 0 #383838; }

  /* Pixel-art progress bar (e.g. forest cut / regrowth) — sunken track + hard fill. */
  .cp-bar {
    width: 100%; height: 12px; margin-top: 5px; background: #141414;
    box-shadow: inset 2px 2px 0 #050505, inset -2px -2px 0 #383838; image-rendering: pixelated;
  }
  .cp-bar-fill { height: 100%; image-rendering: pixelated; transition: width 0.2s ease; }
  .cp-bar-cut { background: #d08a2c; }  /* amber: wood being cut down */
  .cp-bar-grow { background: #4caf50; } /* green: forest regrowing */

  .cp-btn {
    display: flex; align-items: center; justify-content: center;
    text-align: center; cursor: pointer; color: ${TEXT};
    user-select: none; line-height: 1.2;
  }
  .cp-btn .cp-lbl { position: relative; z-index: 1; }
  .cp-btn::before { transition: filter 0.1s ease; }
  .cp-btn:hover::before { filter: brightness(1.35); }
  .cp-btn:active::before { filter: brightness(0.8); }
  .cp-btn.cp-disabled { cursor: default; }
  .cp-btn.cp-disabled .cp-lbl { color: rgb(96,96,96); }
  .cp-btn.cp-disabled::before, .cp-btn.cp-disabled:hover::before { filter: brightness(0.65); }

  .cp-label { line-height: 1.5; overflow: hidden; word-break: break-word; }
  .cp-label.left { display: block; text-align: left; }
  .cp-label.left-center { display: flex; align-items: center; justify-content: flex-start; text-align: left; }
  .cp-label.center { display: flex; align-items: center; justify-content: center; text-align: center; }
  .cp-label u { text-decoration: underline; }

  /* Persistent control bar pinned to the bottom of the menu panel. */
  .cp-footer {
    display: flex; gap: 8px; padding: 6px 8px; box-sizing: border-box;
    background: ${PANEL_BG};
  }
  .cp-footer-menu { flex: 0 0 80px; }
  .cp-footer-end { flex: 1 1 auto; }
  .cp-footer-end .cp-lbl { color: #b6f7a8; }
  .cp-footer-end.cp-disabled .cp-lbl { color: rgb(96,96,96); }

  /* Start dialog + help window (modal overlays) — same 9-slice frames as the HUD. */
  .cp-overlay {
    position: fixed; inset: 0; background: rgba(0,0,0,0.72);
    display: flex; align-items: center; justify-content: center; z-index: 50;
    padding: 16px;
  }
  .cp-dialog {
    position: relative; background: ${FILL}; color: ${TEXT};
    border: 16px solid transparent; border-image: ${RAISED}; image-rendering: pixelated;
    width: min(500px, 94vw); max-height: 90vh; overflow-y: auto;
  }
  .cp-dialog > * { position: relative; z-index: 1; image-rendering: auto; }
  .cp-dialog h2 { font-size: 15px; margin: 2px 0 16px; text-align: center; color:#ffd479; }
  .cp-row { display: flex; align-items: center; justify-content: space-between; margin: 9px 0; font-size: 9px; gap: 8px; }
  .cp-row label { flex: 1; line-height: 1.5; }
  /* Sunken pixel bevel for inputs/selects. */
  .cp-root input, .cp-root select {
    font-family: 'PressStart2P', monospace; font-size: 9px; color:#eee;
    background:#141414; border: 0; padding: 6px;
    box-shadow: inset 2px 2px 0 #050505, inset -2px -2px 0 #383838, 0 0 0 2px #000;
  }
  .cp-root input:focus, .cp-root select:focus { outline: 2px solid #ffd479; outline-offset: 1px; }
  .cp-row input { width: 130px; }
  .cp-row input[type=number] { width: 74px; }
  .cp-section { font-size: 9px; color:#9a9a9a; margin: 16px 0 8px; border-top:2px solid #000; padding-top:12px; }
  .cp-prow { gap: 8px; }
  .cp-prow input { flex: 1; width: auto; min-width: 0; }
  /* Opponent dropdowns need room for the longest label ("Gunnar (AlphaZero XL)"). */
  .cp-prow select { flex: 0 0 auto; width: 210px; }
  .cp-prow input.cp-locked { color:#9a9a9a; cursor: not-allowed; }
  .cp-type { width: 134px; }
  .cp-dialog .cp-actions { display: flex; gap: 14px; margin-top: 20px; justify-content: center; }
  /* Dialog buttons use the raised frame as a real border (they auto-size to text). */
  .cp-dialog button, .cp-mini {
    font-family: 'PressStart2P', monospace; font-size: 10px; color: ${TEXT};
    background: transparent; cursor: pointer;
    border: 12px solid transparent; border-image: ${RAISED}; image-rendering: pixelated;
    padding: 2px 12px; transition: filter 0.1s ease;
  }
  .cp-mini { font-size: 8px; border-width: 10px; padding: 0 8px; }
  .cp-dialog button:hover, .cp-mini:hover { filter: brightness(1.35); }
  .cp-dialog button:active, .cp-mini:active { filter: brightness(0.8); }
  .cp-dialog button:focus-visible, .cp-mini:focus-visible { outline: 2px solid #ffd479; outline-offset: 2px; }
  .cp-dialog button.cp-primary { color: #b6f7a8; }
  .cp-error { color:#ff6a6a; font-size:8px; text-align:center; min-height:12px; margin-top:10px; line-height:1.5; }
  .cp-help { width: min(580px, 94vw); }
  .cp-help h3 { font-size: 11px; margin: 16px 0 6px; color:#ffd479; }
  .cp-help p { font-size: 9px; line-height: 1.7; }
  .cp-help ul { margin: 4px 0 8px; padding-left: 16px; }
  .cp-help li { font-size: 9px; line-height: 1.6; margin: 3px 0; }
  .cp-help u { color:#ffe0a0; }

  /* Transient "it's your turn" banner over the map. */
  .cp-banner {
    position: absolute; left: 50%; top: 14px; transform: translateX(-50%);
    z-index: 40; pointer-events: none;
    font-family: 'PressStart2P', monospace; font-size: 12px; color:#fff;
    background: ${FILL}; border: 16px solid transparent; border-image: ${RAISED};
    image-rendering: pixelated; white-space: nowrap;
    display: flex; align-items: center; gap: 10px;
    opacity: 0; transition: opacity 0.25s ease;
  }
  .cp-banner > * { image-rendering: auto; }
  .cp-banner.cp-show { opacity: 1; }
  .cp-banner .cp-dot { width: 14px; height: 14px; image-rendering: pixelated; }

  /* ---- Replay dashboard (browse + step through recorded games) ------------ */
  /* The panel reuses the raised dialog frame but is wide and column-flexed, with
     three sunken (recessed) sub-panels: the game list, the transport controls and
     the per-seat metrics — same 9-slice language as the in-game HUD. */
  .cp-replay-overlay { padding: 16px; }
  .cp-replay {
    width: min(1140px, 97vw); height: min(840px, 94vh);
    display: flex; flex-direction: column; overflow: hidden;
  }
  .cp-replay-head { display: flex; align-items: center; justify-content: space-between; }
  .cp-replay-head h2 { margin: 0; }
  .cp-replay-body { flex: 1; display: flex; gap: 14px; margin-top: 14px; min-height: 0; }

  /* Sunken sub-panels. .cp-container draws its recessed ::before frame; the host
     must be positioned and its content inset past the 16px frame border. */
  .cp-replay-list, .cp-replay-controls, .cp-replay-metrics { position: relative; }
  .cp-replay-list { flex: 0 0 300px; overflow: hidden; }
  .cp-replay-list-inner {
    position: relative; z-index: 1; height: 100%; overflow-y: auto; padding: 14px;
    display: flex; flex-direction: column; gap: 8px;
  }
  .cp-replay-view { flex: 1; display: flex; flex-direction: column; gap: 14px; min-width: 0; }
  .cp-replay-board {
    flex: 1; min-height: 240px; background: #0c0c0c; overflow: hidden;
    display: flex; align-items: center; justify-content: center;
    box-shadow: inset 2px 2px 0 #050505, inset -2px -2px 0 #2a2a2a;
  }
  .cp-replay-board #cp-stage { transform-origin: center center; }
  .cp-replay-controls-inner, .cp-replay-metrics-inner { position: relative; z-index: 1; padding: 12px 14px; }

  .cp-replay-title { font-size: 9px; line-height: 1.6; color: ${TEXT}; margin-bottom: 10px; }
  .cp-rp-matchup { color: #ffd479; }
  .cp-replay-transport { display: flex; align-items: center; gap: 10px; }
  .cp-replay-transport .cp-mini { min-width: 30px; }
  .cp-replay-turn { font-size: 9px; color: #9a9a9a; font-variant-numeric: tabular-nums; white-space: nowrap; }
  #cp-rp-slider {
    flex: 1; min-width: 0; height: 14px; -webkit-appearance: none; appearance: none;
    background: #141414; box-shadow: inset 2px 2px 0 #050505, inset -2px -2px 0 #383838, 0 0 0 2px #000;
    accent-color: #ffd479; cursor: pointer;
  }
  #cp-rp-slider::-webkit-slider-thumb {
    -webkit-appearance: none; appearance: none; width: 12px; height: 18px; background: #ffd479;
    border: 2px solid #000; cursor: pointer;
  }
  #cp-rp-slider::-moz-range-thumb { width: 12px; height: 18px; background: #ffd479; border: 2px solid #000; }
  #cp-rp-slider:disabled { opacity: 0.4; cursor: default; }

  /* Game list cards — inset bevel; raised highlight on the selected one. */
  .cp-rp-card {
    background: #1a1a1a; padding: 8px 9px; cursor: pointer; line-height: 1.5;
    box-shadow: inset 1px 1px 0 #383838, inset -1px -1px 0 #050505; transition: filter 0.1s ease;
  }
  .cp-rp-card:hover { filter: brightness(1.3); }
  .cp-rp-card.cp-rp-sel { box-shadow: inset 0 0 0 2px #ffd479; background: #232014; }
  .cp-rp-card-matchup { font-size: 8px; color: #d0d0d0; word-break: break-word; }
  .cp-rp-card-meta { font-size: 7px; color: #b6f7a8; margin-top: 5px; }
  .cp-rp-card-date { font-size: 7px; color: #777; margin-top: 3px; }
  .cp-rp-empty { font-size: 8px; color: #9a9a9a; line-height: 1.8; padding: 6px; text-align: center; }

  /* Per-seat metrics. Each seat gets its own full-width row (a steady scoreboard
     layout) and its stat chips never wrap — so the box height stays constant as
     the numbers change turn-to-turn, instead of reflowing and jumping. */
  .cp-replay-metrics-inner { display: flex; flex-direction: column; gap: 8px; }
  .cp-rp-seat {
    background: #141414; padding: 8px 10px;
    box-shadow: inset 2px 2px 0 #050505, inset -2px -2px 0 #2a2a2a;
  }
  .cp-rp-seat-head { display: flex; align-items: center; gap: 7px; font-size: 9px; color: #eee; margin-bottom: 7px; }
  .cp-rp-dot { width: 12px; height: 12px; image-rendering: pixelated; }
  .cp-rp-seat-name { word-break: break-word; }
  .cp-rp-device { margin-left: auto; font-size: 7px; color: #ff9a5a; }
  .cp-rp-seat-stats {
    display: flex; flex-wrap: nowrap; gap: 6px 10px; font-size: 8px; color: #d0d0d0;
    font-variant-numeric: tabular-nums; overflow-x: auto;
  }
  .cp-rp-res { display: inline-flex; align-items: center; gap: 4px; flex: none; }
  .cp-rp-res img { width: 12px; height: 12px; }
  .cp-rp-stat { color: #9a9a9a; flex: none; }
  `;
  const style = document.createElement('style');
  style.textContent = css;
  document.head.appendChild(style);
}
