// TS side of the forward-parity check for the bundled spatial-CNN roster.
//
// Feeds the IDENTICAL deterministic synthetic planes / value_scalars / candidate
// features that rust-trainer's `cnn_fwd_parity` bin uses to each roster net's
// SpatialNetTS forward, and prints board_embed_sum / global_embed_sum /
// score_target / score_pass / value to 10 decimals. Compare against the Rust
// goldens (run: target-agent/release/cnn_fwd_parity <net>.json) to confirm the
// TS port matches to >=6 decimals for BOTH the small (no conv3) and large
// (residual conv3) arch.
//
//   vite-node training/spatial-roster-parity.ts

import { SpatialNetTS } from '../src/ai/nn/spatial_net';
import { SPATIAL_ROSTER } from '../src/ai/nn/models_spatial_roster';

function synthPlanes(pc: number, h: number, w: number): number[] {
  const p = new Array<number>(pc * h * w).fill(0);
  for (let c = 0; c < pc; c++) {
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const idx = (c * h + y) * w + x;
        p[idx] = (((c * 7 + y * 3 + x * 5) % 11) / 11) * 2 - 1;
      }
    }
  }
  return p;
}

for (const m of SPATIAL_ROSTER) {
  const w = m.weights;
  const net = new SpatialNetTS(w);
  const [pc, h, ww] = [w.plane_count, 4, 5];
  const planes = synthPlanes(pc, h, ww);

  const vsd = w.value_scalar_dim ?? 0;
  const vscal: number[] = [];
  for (let i = 0; i < vsd; i++) vscal.push(((i % 7) / 7) * 2 - 1);

  const cache = net.forwardBoard(planes, h, ww, vscal);

  const local: number[] = [];
  for (let i = 0; i < w.local_dim; i++) local.push((i % 5) * 0.1 - 0.2);
  const intent = new Array<number>(w.intent_dim).fill(0);
  intent[3] = 1;

  const sTarget = net.scoreCandidate(cache, { x: 2, y: 1 }, local, intent);
  const sPass = net.scoreCandidate(cache, null, local, intent);
  const value = net.valueFrom(cache);
  const beSum = cache.boardEmbed.reduce((a, b) => a + b, 0);
  const geSum = cache.globalEmbed.reduce((a, b) => a + b, 0);

  console.log(
    `TS_FWD id=${m.id} board_embed_sum=${beSum.toFixed(10)} ` +
      `global_embed_sum=${geSum.toFixed(10)} score_target=${sTarget.toFixed(10)} ` +
      `score_pass=${sPass.toFixed(10)} value=${value.toFixed(10)}`,
  );
}
