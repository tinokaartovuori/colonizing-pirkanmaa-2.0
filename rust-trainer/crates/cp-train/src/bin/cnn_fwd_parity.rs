//! Numeric forward-parity probe for the TS deploy of the CNN champion.
//!
//! Loads a SpatialNet weights.json, runs `forward_board` + `score_candidate` on a
//! DETERMINISTIC synthetic planes tensor and two synthetic candidates (one with a
//! target tile, one with None / Pass), and prints the resulting policy scores +
//! a checksum of board_embed/global_embed. The TS twin (`spatial_net.ts`) runs
//! the IDENTICAL synthetic input; the scores must match to f64 precision, proving
//! the TS conv/dense/pool/score forward is a faithful port. ENGINE-INDEPENDENT —
//! the planes are a fixed formula, not derived from a Game.

use std::env;
use std::fs;

use cp_ai::spatial_net::SpatialNet;

fn synth_planes(pc: usize, h: usize, w: usize) -> Vec<f64> {
    let mut p = vec![0.0f64; pc * h * w];
    for c in 0..pc {
        for y in 0..h {
            for x in 0..w {
                let idx = (c * h + y) * w + x;
                // Deterministic, bounded in ~[-1,1]: a smooth function of (c,y,x).
                p[idx] = (((c * 7 + y * 3 + x * 5) % 11) as f64 / 11.0) * 2.0 - 1.0;
            }
        }
    }
    p
}

fn main() {
    let path = env::args().nth(1).expect("usage: cnn_fwd_parity <weights.json>");
    let net: SpatialNet = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();

    let (pc, h, w) = (net.plane_count, 4usize, 5usize);
    let planes = synth_planes(pc, h, w);

    // Deterministic synthetic value_scalars (length = net.value_scalar_dim),
    // bounded ~[-1,1] — same formula the TS twin uses.
    let vsd = net.value_scalar_dim;
    let vscal: Vec<f64> = (0..vsd).map(|i| ((i % 7) as f64) / 7.0 * 2.0 - 1.0).collect();
    let cache = net.forward_board_scalars(&planes, h, w, &vscal);

    // Synthetic candidate features (deterministic).
    let local: Vec<f64> = (0..net.local_dim).map(|i| ((i % 5) as f64) * 0.1 - 0.2).collect();
    let mut intent = vec![0.0; net.intent_dim];
    intent[3] = 1.0; // BuildOutpost-ish

    let s_target = net.score_candidate(&cache, Some((2, 1)), &local, &intent);
    let s_pass = net.score_candidate(&cache, None, &local, &intent);
    let value = net.value_from(&cache);

    let be_sum: f64 = cache.board_embed.iter().sum();
    let ge_sum: f64 = cache.global_embed.iter().sum();

    println!("RUST_FWD board_embed_sum={:.10} global_embed_sum={:.10} score_target={:.10} score_pass={:.10} value={:.10}",
        be_sum, ge_sum, s_target, s_pass, value);
}
