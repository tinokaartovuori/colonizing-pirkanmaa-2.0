//! AlphaZero POLICY-net gradient trainer — the one genuinely new training piece.
//!
//! The GA never needed backprop (it evolved). AlphaZero trains the policy
//! [`crate::mlp::Genome`] by **gradient descent** so the candidate-softmax matches
//! the MCTS visit-count distribution π. This module is the policy twin of
//! [`crate::value::ValueTrainer`] (whose backprop is finite-difference-verified):
//! same flat-param layout, same Adam, but the output layer is **linear** (the
//! policy head is a scalar score per candidate) and the loss is **cross-entropy
//! of softmax(scores) against π** over the candidate set.
//!
//! Decoupled from feature dimensions on purpose: a [`PolicyExample`] stores the
//! already-assembled per-candidate input vectors, so this trainer works unchanged
//! whether the input is the current 63-dim layout or the future AlphaZero
//! assembly. Additive: nothing here touches the parity path.

use serde::{Deserialize, Serialize};

use crate::mlp::{param_count, Genome};

/// One self-play decision: the per-candidate network inputs and the MCTS
/// visit-count target distribution `pi` over those candidates (sums to 1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyExample {
    /// `inputs[c]` = the policy network input vector for candidate `c`.
    pub inputs: Vec<Vec<f64>>,
    /// MCTS visit-count distribution over the candidates (same length, Σ = 1).
    pub pi: Vec<f64>,
}

/// He-ish random policy genome (weights ~ U[-1,1)/sqrt(nin), zero biases),
/// SplitMix64 stream — so gradient training starts from a non-degenerate net.
pub fn random_genome(arch: &[usize], seed: u64) -> Genome {
    let mut params = vec![0.0; param_count(arch)];
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    let mut next = || {
        s = s.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
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
        offset += nin * nout + nout;
    }
    Genome { arch: arch.to_vec(), params }
}

/// Numerically stable softmax over candidate scores.
pub fn softmax(scores: &[f64]) -> Vec<f64> {
    if scores.is_empty() {
        return Vec::new();
    }
    let mx = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut exps: Vec<f64> = scores.iter().map(|&s| (s - mx).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum > 0.0 {
        for e in &mut exps {
            *e /= sum;
        }
    }
    exps
}

/// Adam cross-entropy trainer for the policy [`Genome`].
pub struct PolicyTrainer {
    pub genome: Genome,
    m: Vec<f64>,
    v: Vec<f64>,
    t: u64,
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    /// L2 weight decay (0 = off).
    pub l2: f64,
    /// Optional FROZEN reference policy for a KL trust-region anchor. When set
    /// (with `kl_coeff > 0`), each step adds `kl_coeff * (p_c - q_c)` to the
    /// per-candidate upstream, where `q = softmax(reference scores)` over the
    /// same candidate inputs. This is the gradient of `kl_coeff * KL(q || p)`:
    /// it pulls the trained policy toward the reference (the warm-start), which
    /// in self-play prevents the drift to passivity (the draw-attractor). The
    /// combined target is equivalent to training CE against the blended
    /// `(pi + kl_coeff*q)`. None / 0.0 = exact legacy behaviour (parity-safe).
    pub ref_genome: Option<Genome>,
    pub kl_coeff: f64,
}

impl PolicyTrainer {
    pub fn new(genome: Genome, lr: f64) -> PolicyTrainer {
        let n = genome.params.len();
        PolicyTrainer {
            genome,
            m: vec![0.0; n],
            v: vec![0.0; n],
            t: 0,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            l2: 0.0,
            ref_genome: None,
            kl_coeff: 0.0,
        }
    }

    /// Forward one candidate input → scalar score (linear output head).
    fn score(&self, input: &[f64]) -> f64 {
        crate::mlp::score(&self.genome, input)
    }

    /// Cross-entropy loss `-Σ π_c log p_c` of the current net over `data`.
    pub fn loss(&self, data: &[PolicyExample]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        let mut sum = 0.0;
        for ex in data {
            let scores: Vec<f64> = ex.inputs.iter().map(|x| self.score(x)).collect();
            let p = softmax(&scores);
            for c in 0..p.len() {
                if ex.pi[c] > 0.0 {
                    sum += -ex.pi[c] * p[c].max(1e-12).ln();
                }
            }
        }
        sum / data.len() as f64
    }

    /// One Adam step over a mini-batch; returns the pre-update mean CE loss.
    pub fn step(&mut self, batch: &[PolicyExample]) -> f64 {
        if batch.is_empty() {
            return 0.0;
        }
        let nparams = self.genome.params.len();
        let mut grad = vec![0.0f64; nparams];
        let mut loss_sum = 0.0f64;
        let inv_b = 1.0 / batch.len() as f64;

        for ex in batch {
            // Forward all candidates, softmax, accumulate per-candidate grads
            // with upstream dL/dscore_c = (p_c - pi_c).
            let scores: Vec<f64> = ex.inputs.iter().map(|x| self.score(x)).collect();
            let p = softmax(&scores);
            // KL trust-region anchor: reference distribution q over the same
            // candidates from the frozen reference policy (None when disabled).
            let qref: Option<Vec<f64>> = if self.kl_coeff != 0.0 {
                self.ref_genome.as_ref().map(|rg| {
                    let rs: Vec<f64> = ex.inputs.iter().map(|x| crate::mlp::score(rg, x)).collect();
                    softmax(&rs)
                })
            } else {
                None
            };
            for c in 0..p.len() {
                if ex.pi[c] > 0.0 {
                    loss_sum += -ex.pi[c] * p[c].max(1e-12).ln();
                }
                let mut upstream = p[c] - ex.pi[c];
                if let Some(q) = &qref {
                    upstream += self.kl_coeff * (p[c] - q[c]);
                }
                if upstream != 0.0 {
                    self.backprop_into(&ex.inputs[c], upstream, inv_b, &mut grad);
                }
            }
        }

        // L2 decay on weights (added to the averaged grad).
        if self.l2 > 0.0 {
            for k in 0..nparams {
                grad[k] += self.l2 * self.genome.params[k];
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
            self.genome.params[k] -= self.lr * mhat / (vhat.sqrt() + self.eps);
        }

        loss_sum * inv_b
    }

    /// Backprop a single candidate with a scalar upstream gradient on the LINEAR
    /// output (`dL/dscore`), scaling by `w`, accumulating into `grad`. Mirrors
    /// `value.rs::backprop_example` but: output layer is linear (no tanh deriv),
    /// hidden layers tanh. Same flat-param layout as `mlp`.
    fn backprop_into(&self, input: &[f64], upstream: f64, w: f64, grad: &mut [f64]) {
        let arch = &self.genome.arch;
        let params = &self.genome.params;
        let nlayers = arch.len() - 1;

        // Forward, caching per-layer activations.
        let mut acts: Vec<Vec<f64>> = Vec::with_capacity(nlayers + 1);
        acts.push(input.to_vec());
        let mut offset = 0usize;
        let mut layer_meta: Vec<(usize, usize, usize)> = Vec::with_capacity(nlayers);
        for l in 0..nlayers {
            let nin = arch[l];
            let nout = arch[l + 1];
            layer_meta.push((offset, nin, nout));
            let prev = &acts[l];
            let is_hidden = l < nlayers - 1;
            let mut out = vec![0.0f64; nout];
            for j in 0..nout {
                let mut sum = params[offset + nin * nout + j];
                let base = offset + j * nin;
                for i in 0..nin {
                    sum += params[base + i] * prev[i];
                }
                out[j] = if is_hidden { sum.tanh() } else { sum }; // linear output
            }
            offset += nin * nout + nout;
            acts.push(out);
        }

        // Backward. Output layer linear → delta = upstream (no tanh deriv).
        let mut delta = vec![upstream]; // 1 output unit
        for l in (0..nlayers).rev() {
            let (off, nin, nout) = layer_meta[l];
            let prev = &acts[l];
            for j in 0..nout {
                let dj = delta[j];
                let base = off + j * nin;
                for i in 0..nin {
                    grad[base + i] += w * dj * prev[i];
                }
                grad[off + nin * nout + j] += w * dj; // bias
            }
            if l > 0 {
                let mut new_delta = vec![0.0f64; nin];
                for i in 0..nin {
                    let mut s = 0.0;
                    for j in 0..nout {
                        let base = off + j * nin;
                        s += params[base + i] * delta[j];
                    }
                    let a = prev[i];
                    new_delta[i] = s * (1.0 - a * a); // tanh' on the hidden activation
                }
                delta = new_delta;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arch() -> Vec<usize> {
        vec![6, 8, 1]
    }

    fn example(seed: u64) -> PolicyExample {
        // 3 candidates, fixed pseudo-random inputs; target favours candidate 1.
        let mut s = seed;
        let mut rnd = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) as f64 / (1u64 << 31) as f64) - 1.0
        };
        let inputs: Vec<Vec<f64>> = (0..3).map(|_| (0..6).map(|_| rnd()).collect()).collect();
        PolicyExample { inputs, pi: vec![0.1, 0.8, 0.1] }
    }

    #[test]
    fn softmax_sums_to_one() {
        let p = softmax(&[1.0, 2.0, 3.0]);
        let sum: f64 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
        assert!(p[2] > p[1] && p[1] > p[0]);
    }

    #[test]
    fn cross_entropy_reduces_and_matches_target() {
        let mut tr = PolicyTrainer::new(random_genome(&arch(), 3), 0.05);
        let data = vec![example(11), example(22), example(33)];
        let before = tr.loss(&data);
        for _ in 0..400 {
            tr.step(&data);
        }
        let after = tr.loss(&data);
        assert!(after < before, "CE loss did not drop: {before} -> {after}");
        // The net should put most mass on candidate 1 for at least one example.
        let scores: Vec<f64> = data[0].inputs.iter().map(|x| crate::mlp::score(&tr.genome, x)).collect();
        let p = softmax(&scores);
        assert!(p[1] > p[0] && p[1] > p[2], "did not learn the argmax: {p:?}");
    }

    #[test]
    fn kl_anchor_pulls_policy_toward_reference() {
        // The MCTS target pi favours candidate 0, but a strong KL anchor to a
        // reference policy q should keep the trained policy near q instead of
        // collapsing fully onto pi. We verify the anchored policy ends up closer
        // to q (smaller cross-entropy H(q, p)) than an identical unanchored run.
        let data = vec![example(11), example(22), example(33)];
        let reference = random_genome(&arch(), 4242); // frozen reference policy
        let qs: Vec<Vec<f64>> = data
            .iter()
            .map(|ex| softmax(&ex.inputs.iter().map(|x| crate::mlp::score(&reference, x)).collect::<Vec<_>>()))
            .collect();
        let h_q_given = |tr: &PolicyTrainer| -> f64 {
            data.iter().zip(&qs).map(|(ex, q)| {
                let p = softmax(&ex.inputs.iter().map(|x| crate::mlp::score(&tr.genome, x)).collect::<Vec<_>>());
                -q.iter().zip(&p).map(|(qc, pc)| qc * pc.max(1e-12).ln()).sum::<f64>()
            }).sum()
        };

        // Identical init for a fair comparison.
        let init = random_genome(&arch(), 7);
        let mut anchored = PolicyTrainer::new(init.clone(), 0.05);
        anchored.ref_genome = Some(reference.clone());
        anchored.kl_coeff = 5.0;
        let mut free = PolicyTrainer::new(init, 0.05);
        for _ in 0..400 {
            anchored.step(&data);
            free.step(&data);
        }
        let h_anchored = h_q_given(&anchored);
        let h_free = h_q_given(&free);
        assert!(
            h_anchored < h_free,
            "KL anchor did not keep policy near reference: anchored H(q,p)={h_anchored} >= free {h_free}"
        );
    }

    #[test]
    fn finite_difference_gradient_matches_backprop() {
        // Verify the CE gradient w.r.t. params against finite differences.
        let g = random_genome(&arch(), 99);
        let tr = PolicyTrainer::new(g, 0.01);
        let ex = example(7);

        // Analytic grad: accumulate (p_c - pi_c) backprop, weight 1.0.
        let scores: Vec<f64> = ex.inputs.iter().map(|x| crate::mlp::score(&tr.genome, x)).collect();
        let p = softmax(&scores);
        let mut grad = vec![0.0f64; tr.genome.params.len()];
        for c in 0..p.len() {
            tr.backprop_into(&ex.inputs[c], p[c] - ex.pi[c], 1.0, &mut grad);
        }

        let h = 1e-6;
        let loss = |g: &Genome| -> f64 {
            let sc: Vec<f64> = ex.inputs.iter().map(|x| crate::mlp::score(g, x)).collect();
            let pp = softmax(&sc);
            let mut l = 0.0;
            for c in 0..pp.len() {
                if ex.pi[c] > 0.0 {
                    l += -ex.pi[c] * pp[c].max(1e-12).ln();
                }
            }
            l
        };
        for &k in &[0usize, 5, 20, tr.genome.params.len() - 1] {
            let mut gp = tr.genome.clone();
            gp.params[k] += h;
            let lp = loss(&gp);
            gp.params[k] -= 2.0 * h;
            let lm = loss(&gp);
            let fd = (lp - lm) / (2.0 * h);
            assert!((fd - grad[k]).abs() < 1e-5, "param {k}: fd={fd} backprop={}", grad[k]);
        }
    }
}
