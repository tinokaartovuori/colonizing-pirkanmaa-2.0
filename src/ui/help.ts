// Port of helpwindow.ui content — the in-game rules window.

const HELP_HTML = `
<p>Colonizing Pirkanmaa is a turn-based strategy game for two to four players (human or computer). Build an economy, raise an army, and become the sole ruler of Pirkanmaa.</p>

<h3>How to win</h3>
<p>There are three ways to win:</p>
<ul>
  <li><b>Conquest</b> &mdash; be the last player standing by conquering every rival's headquarters.</li>
  <li><b>Domination</b> &mdash; own <b>70%</b> of all tiles.</li>
  <li><b>Strange Device</b> &mdash; build the Device and keep it standing until its countdown reaches zero (see below).</li>
</ul>

<h3>How you lose</h3>
<p>You are out of the game if any of these happen:</p>
<ul>
  <li>Your resources go negative (you can't pay your wages).</li>
  <li>Your headquarters is conquered by an enemy soldier.</li>
  <li>Another player reaches 70% of the map.</li>
  <li>An opponent's Strange Device finishes its countdown.</li>
</ul>

<h3>Getting started</h3>
<p>On the first round each player picks a starting <u>grassland</u> tile. Your headquarters is placed there and you also claim every unowned tile around it. After that, end your turn and the next player goes.</p>

<h3>Controls</h3>
<ul>
  <li><b>Click a tile</b> to inspect it, build, or buy &amp; place a unit.</li>
  <li><b>Click one of your units</b> on the map to pick it up, then click a neighbouring tile to move it there. (The MOVE button in a tile's panel does the same thing.)</li>
  <li><b>END TURN</b> collects income, pays wages, resolves battles, then passes play on.</li>
</ul>

<h3>Resources</h3>
<ul>
  <li><u>Money</u> &mdash; pays for everything and for unit wages each round.</li>
  <li><u>Wood</u> &mdash; harvested from forests; needed by almost every building.</li>
  <li><u>Stone</u> &mdash; mined from mountains; used in buildings.</li>
  <li><u>Metal</u> &mdash; mined from mountains; required for soldiers and advanced buildings.</li>
</ul>

<h3>Tiles</h3>
<ul>
  <li><u>Grassland</u> &mdash; the workhorse tile. Build a Farm, Village, Outpost or Nuclear Plant.</li>
  <li><u>Forest</u> &mdash; the only source of wood (plus some stone). Harvest it 6 times; it regrows in 5 turns, and once fully cut you can build on it.</li>
  <li><u>Mountain</u> &mdash; the only place for a Mine.</li>
  <li><u>River</u> &mdash; units may enter but cannot cross until you build a Bridge or Hydroelectric Plant.</li>
  <li><u>Abundant Forest</u> &mdash; station one worker to forage fruit for a little money; no building needed.</li>
</ul>

<h3>Units</h3>
<p>Bought from the UNIT SHOP, up to your unit limit (soldiers have a separate limit). A tile holds 3 units &mdash; or 6 while it is being contested.</p>
<ul>
  <li><u>Worker</u> &mdash; runs buildings and harvests forests. Cheap, with a small wage.</li>
  <li><u>Expert</u> &mdash; required by power plants and doubles a mine's output. Pricey, with a high wage.</li>
  <li><u>Soldier</u> &mdash; defends your land and conquers tiles. Costs metal and has the highest wage.</li>
</ul>

<h3>Buildings</h3>
<ul>
  <li><u>Headquarters</u> &mdash; the heart of your region (free, placed at the start). Lose it and you lose. +3 max units, +1 max soldier.</li>
  <li><u>Farm</u> &mdash; pays money every 4 rounds; keep a worker on it or the crops die. Grassland / cut forest.</li>
  <li><u>Mine</u> &mdash; mountains only; steady stone and metal. Each worker adds output; an expert doubles it.</li>
  <li><u>Village</u> &mdash; +3 max units. Grassland / cut forest.</li>
  <li><u>Outpost</u> &mdash; cannot be conquered; +3 max soldiers. Grassland / cut forest. <b>Cannot be built next to your HQ or another Outpost</b>, so place it a tile or more out from them.</li>
  <li><u>Nuclear Plant</u> &mdash; the top money maker; needs an expert plus workers. Grassland / cut forest.</li>
  <li><u>Hydroelectric Plant</u> &mdash; makes money and doubles as a bridge; straight river only; needs an expert and a worker.</li>
  <li><u>Bridge</u> &mdash; lets units cross a straight river; costs a little wood each round.</li>
  <li><u>Mikontalo</u> &mdash; one spawns at random; conquer it for +2 max units.</li>
  <li><u>Strange Device</u> &mdash; an alternate way to win (see below). Very expensive; only one can ever be built. Grassland / cut forest.</li>
</ul>

<h3>The Strange Device</h3>
<p>The Device is a third path to victory, separate from war and expansion. Build it on one of your own empty tiles &mdash; it is very costly, and only one Device exists in the whole game. Once built it starts a <u>countdown</u>; if the Device is still standing when the countdown hits zero, its owner <b>wins instantly</b>.</p>
<p>Building it is a gamble. While the Device stands your <b>maximum soldiers drop by 2</b>, leaving you exposed, and the Device tile can hold only <b>one</b> defender. If an enemy captures that tile the Device is destroyed and the race is off. So the builder must hold on while everyone else rushes to crack it.</p>

<h3>Conquering</h3>
<p>You can act one tile beyond your border. An unowned tile is claimed by moving any unit onto it. An enemy tile falls only if your soldiers there outnumber the defenders &mdash; otherwise your attackers are wiped out. Outposts can never be taken. Cut an enemy off from their HQ and the stranded tiles go neutral, their units lost, but the buildings remain for the taking.</p>

<h3>Computer players</h3>
<p>Any seat can be filled by one of three named AI opponents. Picking one locks that seat to its character name.</p>
<ul>
  <li><u>Jorma</u> &mdash; a hand-written strategy bot. Expands steadily, fields soldiers, and presses weak neighbours.</li>
  <li><u>Kalevi</u> &mdash; a neural network trained by self-play (AlphaZero-style). The strongest all-round opponent.</li>
  <li><u>Gunnar</u> &mdash; a larger neural network. A touch weaker than Kalevi overall but more aggressive &mdash; he builds a bigger army.</li>
</ul>
<p>While an AI takes its turn a banner reads "&hellip;is playing" &mdash; watch its units and buildings appear on the map.</p>
`;

export function showHelpWindow(): void {
  const overlay = document.createElement('div');
  overlay.className = 'cp-root cp-overlay';
  const dialog = document.createElement('div');
  dialog.className = 'cp-dialog cp-help';
  dialog.innerHTML = `<h2>Help</h2>${HELP_HTML}<div class="cp-actions"><button id="cp-help-close">Close</button></div>`;
  overlay.appendChild(dialog);
  document.body.appendChild(overlay);
  const close = () => overlay.remove();
  dialog.querySelector('#cp-help-close')!.addEventListener('click', close);
  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) close();
  });
}
