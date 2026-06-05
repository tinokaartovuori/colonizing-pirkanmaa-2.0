// A tiny, dependency-free dense multi-layer perceptron.
//
// This is the *entire* trained artifact: training (neuroevolution in
// `training/`) and client-side inference both call `forward()` here, so a
// genome is just the flat list of weights this file knows how to consume. No
// matrix library, no autodiff — a few loops over `number[]`, which runs as
// happily in the browser as in Node.
//
// Architecture is a list of layer sizes, e.g. [inputDim, 24, 16, 1]. Hidden
// layers use tanh; the output layer is linear (the network scores a candidate
// action, so a real-valued head is what we want). Weights are stored row-major
// per layer: for a layer with `nin` inputs and `nout` outputs we store the
// `nin*nout` weights followed by the `nout` biases.

export type LayerSizes = number[];

/** A genome: the architecture plus the flat parameter vector that fills it. */
export interface Genome {
  arch: LayerSizes;
  params: number[];
}

/** Number of trainable parameters for a given architecture. */
export function paramCount(arch: LayerSizes): number {
  let n = 0;
  for (let l = 0; l < arch.length - 1; l++) {
    const nin = arch[l];
    const nout = arch[l + 1];
    n += nin * nout + nout;
  }
  return n;
}

/**
 * Forward pass. `input.length` must equal `arch[0]`. Returns the output layer
 * activations (length `arch[arch.length-1]`). Hidden layers: tanh. Output:
 * linear. Allocations are kept minimal; this is called once per candidate
 * action per decision, so it stays cheap.
 */
export function forward(genome: Genome, input: number[]): number[] {
  const { arch, params } = genome;
  let act = input;
  let offset = 0;
  for (let l = 0; l < arch.length - 1; l++) {
    const nin = arch[l];
    const nout = arch[l + 1];
    const out = new Array<number>(nout);
    const isHidden = l < arch.length - 2;
    for (let j = 0; j < nout; j++) {
      // bias first conceptually, but we lay weights then biases per layer.
      let sum = params[offset + nin * nout + j];
      const base = offset + j * nin;
      for (let i = 0; i < nin; i++) {
        sum += params[base + i] * act[i];
      }
      out[j] = isHidden ? Math.tanh(sum) : sum;
    }
    offset += nin * nout + nout;
    act = out;
  }
  return act;
}

/** Convenience: a network whose only output is a single scalar score. */
export function score(genome: Genome, input: number[]): number {
  return forward(genome, input)[0];
}

/** A fresh genome with small random weights (used to seed a population). */
export function randomGenome(arch: LayerSizes, rand: () => number, scale = 0.5): Genome {
  const n = paramCount(arch);
  const params = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    // Box–Muller normal, scaled. Small init keeps early play sane.
    const u1 = Math.max(rand(), 1e-9);
    const u2 = rand();
    params[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) * scale;
  }
  return { arch, params };
}

/** A genome of all-zero weights — forward() then returns all zeros, so the
 *  policy falls back to candidate enumeration order. Used as a safe default
 *  before any weights are trained. */
export function zeroGenome(arch: LayerSizes): Genome {
  return { arch, params: new Array<number>(paramCount(arch)).fill(0) };
}
