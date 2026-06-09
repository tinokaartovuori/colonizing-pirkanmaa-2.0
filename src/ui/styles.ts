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
  .cp-container::before { border-image: ${SUNKEN}; }
  .cp-btn::before { border-image: ${RAISED}; }

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
    border: 16px solid transparent; border-image: ${SUNKEN}; image-rendering: pixelated;
    width: min(440px, 92vw); max-height: 90vh; overflow-y: auto;
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
  .cp-prow input { flex: 1; width: auto; }
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
  `;
  const style = document.createElement('style');
  style.textContent = css;
  document.head.appendChild(style);
}
