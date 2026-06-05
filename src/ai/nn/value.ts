// Learned VALUE NET — a separate small MLP used as the MCTS leaf evaluator, to
// recover rollout-MCTS strength at static-eval speed. 1:1 port of the Rust
// `cp-ai/src/value.rs` forward pass.
//
// This is a brand-new, separate artifact: it does NOT touch the policy Genome
// (mlp.ts), its arch, candidates, features, or the policy numerics. The
// search-OFF / parity path never constructs a value net, so the parity gate
// stays byte-identical.
//
// Network: [GLOBAL_DIM(36), 32, 16, 1]. Hidden activations tanh; the OUTPUT
// activation is ALSO tanh (the one difference from the policy MLP, whose output
// is linear) so the value lives in [-1, 1] — directly comparable to the exact
// terminal ±1 / 0 the search uses. Input is the 36-dim global feature vector for
// the to-move (root) player.

import { GLOBAL_DIM } from './features';

/** The fixed value-net architecture: [36, 32, 16, 1]. */
export const VALUE_ARCH = [GLOBAL_DIM, 32, 16, 1];

/** A value-net genome: architecture + flat parameter vector (own JSON). */
export interface ValueNet {
  arch: number[];
  params: number[];
}

/**
 * Forward pass → a scalar value in [-1, 1]. `input.length` must equal `arch[0]`.
 * EVERY layer (hidden AND output) is tanh, mirroring value.rs::forward. Same
 * flat-param layout as mlp.ts::forward (weights row-major, then biases, per
 * layer).
 */
export function valueForward(net: ValueNet, input: number[]): number {
  const { arch, params } = net;
  let act = input;
  let offset = 0;
  for (let l = 0; l < arch.length - 1; l++) {
    const nin = arch[l];
    const nout = arch[l + 1];
    const out = new Array<number>(nout);
    for (let j = 0; j < nout; j++) {
      let sum = params[offset + nin * nout + j];
      const base = offset + j * nin;
      for (let i = 0; i < nin; i++) {
        sum += params[base + i] * act[i];
      }
      out[j] = Math.tanh(sum); // hidden AND output → value ∈ [-1, 1]
    }
    offset += nin * nout + nout;
    act = out;
  }
  return act[0];
}
