//! Stage-B LEARNED VALUE NET — a SEPARATE small MLP used as the MCTS leaf
//! evaluator, to recover rollout-MCTS strength at static-eval speed.
//!
//! This is a brand-new, separate artifact: it does NOT touch the policy
//! [`crate::mlp::Genome`], its arch, candidates, features, or the policy
//! numerics. The parity path never constructs a value net, so the parity gate
//! stays byte-identical.
//!
//! ## Network
//! - Architecture: `[GLOBAL_DIM(36), 32, 16, 1]`.
//! - Hidden activations: `tanh`.
//! - **Output activation: `tanh`** (the one difference from the policy MLP,
//!   whose output layer is linear) so the value lives in `[-1, 1]` — directly
//!   comparable to the exact terminal `±1 / 0` the search uses.
//! - Input: the 36-dim global feature vector for the to-move player
//!   ([`crate::features::global_features`]).
//!
//! The forward math reuses the same flat-param layout as [`crate::mlp`] (weights
//! row-major then biases, per layer); only the final scalar is tanh'd.
//!
//! ## Training
//! Hand-coded backprop (the net is tiny — no external autodiff). MSE loss
//! against the game-outcome target `z ∈ {+1, -1, 0}`, mini-batch SGD with Adam.
//! See [`ValueTrainer`].

use serde::{Deserialize, Serialize};

use crate::features::GLOBAL_DIM;

/// The fixed value-net architecture: `[36, 32, 16, 1]`.
pub const VALUE_ARCH: [usize; 4] = [GLOBAL_DIM, 32, 16, 1];
/// SPATIAL value-net architecture: `[41, 32, 16, 1]` — global + spatial summaries
/// (input = [`crate::features::value_features_spatial`]). Targets the conversion
/// ceiling with positional awareness in the leaf evaluator.
pub const VALUE_ARCH_SPATIAL: [usize; 4] = [crate::features::VALUE_SPATIAL_DIM, 32, 16, 1];

/// A value-net genome: architecture + flat parameter vector. Serialised to its
/// OWN JSON (`value.json`) — separate from the policy genome.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueNet {
    pub arch: Vec<usize>,
    pub params: Vec<f64>,
}

impl ValueNet {
    /// A zero-initialised net of the default [`VALUE_ARCH`].
    pub fn zeros() -> ValueNet {
        ValueNet {
            arch: VALUE_ARCH.to_vec(),
            params: vec![0.0; crate::mlp::param_count(&VALUE_ARCH)],
        }
    }

    /// Randomly initialise the default-arch net.
    pub fn random(seed: u64) -> ValueNet {
        ValueNet::random_arch(&VALUE_ARCH, seed)
    }

    /// Randomly initialise an arbitrary-arch net with He-ish uniform weights
    /// (scaled by `1/sqrt(nin)`), zero biases, using a SplitMix64 stream.
    pub fn random_arch(arch_in: &[usize], seed: u64) -> ValueNet {
        let arch = arch_in.to_vec();
        let mut params = vec![0.0; crate::mlp::param_count(&arch)];
        let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
        let mut next = || {
            s = s.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            // Uniform in [-1, 1).
            ((z >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        };
        let mut offset = 0usize;
        for l in 0..arch.len() - 1 {
            let nin = arch[l];
            let nout = arch[l + 1];
            let scale = 1.0 / (nin as f64).sqrt();
            for j in 0..nout {
                let base = offset + j * nin;
                for i in 0..nin {
                    params[base + i] = next() * scale;
                }
            }
            // Biases (laid down after the weights) stay 0.
            offset += nin * nout + nout;
        }
        ValueNet { arch, params }
    }

    pub fn from_json(s: &str) -> Result<ValueNet, serde_json::Error> {
        serde_json::from_str(s)
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("ValueNet serialises")
    }

    pub fn from_file(path: &str) -> std::io::Result<ValueNet> {
        let s = std::fs::read_to_string(path)?;
        ValueNet::from_json(&s).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    pub fn to_file(&self, path: &str) -> std::io::Result<()> {
        std::fs::write(path, self.to_json())
    }

    /// Forward pass → a scalar value in `[-1, 1]`. `input.len()` must equal
    /// `arch[0]`. Hidden layers `tanh`, **output layer `tanh`** (the value
    /// squashing). Same flat-param layout as [`crate::mlp::forward`].
    pub fn forward(&self, input: &[f64]) -> f64 {
        let arch = &self.arch;
        let params = &self.params;
        let mut act: Vec<f64> = input.to_vec();
        let mut offset = 0usize;
        for l in 0..arch.len() - 1 {
            let nin = arch[l];
            let nout = arch[l + 1];
            let mut out = vec![0.0f64; nout];
            for j in 0..nout {
                let mut sum = params[offset + nin * nout + j];
                let base = offset + j * nin;
                for i in 0..nin {
                    sum += params[base + i] * act[i];
                }
                // Every layer (hidden AND output) is tanh → value ∈ [-1, 1].
                out[j] = sum.tanh();
            }
            offset += nin * nout + nout;
            act = out;
        }
        act[0]
    }
}

// ===========================================================================
// Gradient regression trainer — hand-coded backprop through tanh layers.
// ===========================================================================

/// One training example: a 36-dim global feature vector + its outcome target z.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueExample {
    /// 36-dim global features for the to-move player at a turn start.
    pub x: Vec<f64>,
    /// Final game outcome from that player's perspective: +1 win, -1 loss, 0 tie.
    pub z: f64,
}

/// Adam-based MSE regression trainer for a [`ValueNet`]. The net is tiny
/// (~1.7k params) so a dense full-backprop per example is cheap.
pub struct ValueTrainer {
    pub net: ValueNet,
    // Adam moment estimates (same length as params).
    m: Vec<f64>,
    v: Vec<f64>,
    t: u64,
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
}

impl ValueTrainer {
    pub fn new(net: ValueNet, lr: f64) -> ValueTrainer {
        let n = net.params.len();
        ValueTrainer {
            net,
            m: vec![0.0; n],
            v: vec![0.0; n],
            t: 0,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }

    /// Mean-squared error of the current net over `data` (no gradient).
    pub fn mse(&self, data: &[ValueExample]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        let mut sum = 0.0;
        for ex in data {
            let yhat = self.net.forward(&ex.x);
            let e = yhat - ex.z;
            sum += e * e;
        }
        sum / data.len() as f64
    }

    /// One Adam step over a mini-batch. Returns the batch mean-squared error
    /// (computed at the pre-update params, the standard convention). Accumulates
    /// the gradient of `MSE = mean((yhat - z)^2)` across the batch, then applies
    /// a single Adam update.
    pub fn step(&mut self, batch: &[ValueExample]) -> f64 {
        if batch.is_empty() {
            return 0.0;
        }
        let nparams = self.net.params.len();
        let mut grad = vec![0.0f64; nparams];
        let mut loss_sum = 0.0f64;
        let inv_b = 1.0 / batch.len() as f64;

        for ex in batch {
            let (yhat, g) = self.backprop_example(&ex.x, ex.z);
            let e = yhat - ex.z;
            loss_sum += e * e;
            for k in 0..nparams {
                grad[k] += g[k] * inv_b;
            }
        }

        // Adam update.
        self.t += 1;
        let t = self.t as f64;
        let bc1 = 1.0 - self.beta1.powf(t);
        let bc2 = 1.0 - self.beta2.powf(t);
        for k in 0..nparams {
            self.m[k] = self.beta1 * self.m[k] + (1.0 - self.beta1) * grad[k];
            self.v[k] = self.beta2 * self.v[k] + (1.0 - self.beta2) * grad[k] * grad[k];
            let mhat = self.m[k] / bc1;
            let vhat = self.v[k] / bc2;
            self.net.params[k] -= self.lr * mhat / (vhat.sqrt() + self.eps);
        }

        loss_sum * inv_b
    }

    /// Backprop one example. Returns `(yhat, grad)` where `grad` is the gradient
    /// of the per-example squared error `(yhat - z)^2` w.r.t. every param, in the
    /// SAME flat layout as the params. tanh derivative: `1 - a^2`.
    fn backprop_example(&self, input: &[f64], z: f64) -> (f64, Vec<f64>) {
        let arch = &self.net.arch;
        let params = &self.net.params;
        let nlayers = arch.len() - 1;

        // Forward, caching pre-activations are unnecessary since tanh'(s)=1-a^2.
        // Cache per-layer activations: acts[0]=input, acts[l+1]=layer l output.
        let mut acts: Vec<Vec<f64>> = Vec::with_capacity(nlayers + 1);
        acts.push(input.to_vec());
        let mut offset = 0usize;
        // Remember each layer's (offset, nin, nout) for the backward pass.
        let mut layer_meta: Vec<(usize, usize, usize)> = Vec::with_capacity(nlayers);
        for l in 0..nlayers {
            let nin = arch[l];
            let nout = arch[l + 1];
            layer_meta.push((offset, nin, nout));
            let prev = &acts[l];
            let mut out = vec![0.0f64; nout];
            for j in 0..nout {
                let mut sum = params[offset + nin * nout + j];
                let base = offset + j * nin;
                for i in 0..nin {
                    sum += params[base + i] * prev[i];
                }
                out[j] = sum.tanh();
            }
            offset += nin * nout + nout;
            acts.push(out);
        }

        let yhat = acts[nlayers][0];

        // Backward. dL/dyhat for L = (yhat - z)^2 → 2*(yhat - z).
        let mut grad = vec![0.0f64; params.len()];
        // delta[j] = dL/d(pre-activation s_j) for the current layer.
        let mut delta = vec![2.0 * (yhat - z)]; // output layer has 1 unit
        // Apply the output tanh derivative: dL/ds = dL/da * (1 - a^2).
        {
            let a = acts[nlayers][0];
            delta[0] *= 1.0 - a * a;
        }

        for l in (0..nlayers).rev() {
            let (off, nin, nout) = layer_meta[l];
            let prev = &acts[l];
            // Accumulate weight/bias grads for this layer.
            for j in 0..nout {
                let dj = delta[j];
                let base = off + j * nin;
                for i in 0..nin {
                    grad[base + i] += dj * prev[i];
                }
                grad[off + nin * nout + j] += dj; // bias
            }
            if l > 0 {
                // Propagate to the previous layer's activations, then through its
                // tanh derivative (using the cached previous-layer activations).
                let mut new_delta = vec![0.0f64; nin];
                for i in 0..nin {
                    let mut s = 0.0;
                    for j in 0..nout {
                        let base = off + j * nin;
                        s += params[base + i] * delta[j];
                    }
                    let a = prev[i];
                    new_delta[i] = s * (1.0 - a * a);
                }
                delta = new_delta;
            }
        }

        (yhat, grad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_in_range() {
        let net = ValueNet::random(42);
        let x = vec![0.5; GLOBAL_DIM];
        let y = net.forward(&x);
        assert!(y > -1.0 && y < 1.0, "value {y} out of (-1,1)");
    }

    #[test]
    fn zeros_outputs_zero() {
        let net = ValueNet::zeros();
        let x = vec![0.7; GLOBAL_DIM];
        assert!(net.forward(&x).abs() < 1e-12);
    }

    #[test]
    fn json_roundtrip() {
        let net = ValueNet::random(7);
        let net2 = ValueNet::from_json(&net.to_json()).unwrap();
        assert_eq!(net.arch, net2.arch);
        assert_eq!(net.params.len(), net2.params.len());
        // serde_json's shortest-round-trip float formatting is accurate to f64
        // precision; assert close rather than bit-identical (the value net is
        // NOT a parity artifact).
        for (a, b) in net.params.iter().zip(&net2.params) {
            assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        }
    }

    #[test]
    fn gradient_reduces_loss_on_one_example() {
        // A single (x, z) target: training should drive forward(x) → z.
        let mut tr = ValueTrainer::new(ValueNet::random(1), 0.01);
        let x: Vec<f64> = (0..GLOBAL_DIM).map(|i| ((i as f64) / 36.0) - 0.5).collect();
        let data = vec![ValueExample { x: x.clone(), z: 0.8 }];
        let before = tr.mse(&data);
        for _ in 0..500 {
            tr.step(&data);
        }
        let after = tr.mse(&data);
        assert!(after < before, "loss did not drop: {before} -> {after}");
        assert!(after < 1e-3, "did not converge: {after}");
    }

    #[test]
    fn finite_difference_gradient_matches_backprop() {
        // Numerically verify backprop against finite differences for a few params.
        let net = ValueNet::random(99);
        let tr = ValueTrainer::new(net, 0.01);
        let x: Vec<f64> = (0..GLOBAL_DIM).map(|i| (i as f64 * 0.013).sin()).collect();
        let z = -0.3;
        let (_y, grad) = tr.backprop_example(&x, z);

        let h = 1e-6;
        let loss = |net: &ValueNet| -> f64 {
            let e = net.forward(&x) - z;
            e * e
        };
        for &k in &[0usize, 100, 500, 1000, tr.net.params.len() - 1] {
            let mut np = tr.net.clone();
            np.params[k] += h;
            let lp = loss(&np);
            np.params[k] -= 2.0 * h;
            let lm = loss(&np);
            let fd = (lp - lm) / (2.0 * h);
            assert!(
                (fd - grad[k]).abs() < 1e-4,
                "param {k}: fd={fd} backprop={}",
                grad[k]
            );
        }
    }
}
