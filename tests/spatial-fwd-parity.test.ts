import { describe, it, expect } from 'vitest';
import {
  SpatialNetTS, SpatialWeights, PLANE_COUNT,
} from '../src/ai/nn/spatial_net';
import { SPATIAL_CHAMPION_WEIGHTS } from '../src/ai/nn/models_spatial';

// Numeric forward-parity probe — the TS twin of rust-trainer's `cnn_fwd_parity`
// bin. Builds the IDENTICAL deterministic synthetic planes / value_scalars /
// candidate features and checks the policy scores AND the value-head output
// against the Rust golden, proving the conv/dense/pool/score/value forward is a
// faithful f64 port. ENGINE-INDEPENDENT (planes are a fixed formula).
//
// Golden produced by:
//   CARGO_TARGET_DIR=target-agent cargo run -p cp-train --bin cnn_fwd_parity \
//     --release -- models/sd4/az/sd4-az-002/weights.json
const RUST = {
  board_embed_sum: 6.9047137446,
  global_embed_sum: 0.3452356872,
  score_target: -0.2955589415,
  score_pass: -0.3225061547,
  value: -0.7509915859,
};

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

describe('CNN forward parity (TS twin of cnn_fwd_parity)', () => {
  it('matches the Rust forward (policy + value head) to ≥6 decimals', () => {
    const w = SPATIAL_CHAMPION_WEIGHTS as SpatialWeights;
    const net = new SpatialNetTS(w);
    const [pc, h, ww] = [w.plane_count, 4, 5];
    expect(pc).toBe(PLANE_COUNT);
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

    // ≥6 decimals (the Rust golden is printed to 10).
    expect(beSum).toBeCloseTo(RUST.board_embed_sum, 6);
    expect(geSum).toBeCloseTo(RUST.global_embed_sum, 6);
    expect(sTarget).toBeCloseTo(RUST.score_target, 6);
    expect(sPass).toBeCloseTo(RUST.score_pass, 6);
    expect(value).toBeCloseTo(RUST.value, 6);
  });
});
