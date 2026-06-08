//! Dual-head spatial neural net for the hand-rolled AlphaZero trainer.
//!
//! ADDITIVE / AZ-ONLY: this module is *not* wired into the parity-locked
//! feature / policy / search path. It composes the CNN primitives in
//! [`crate::cnn`] into a size-agnostic value + policy network:
//!
//! - Trunk: `planes(PC,H,W) -> Conv2d(PC->D1,3,1) -> tanh -> Conv2d(D1->D,3,1)
//!   -> tanh = trunk2(D,H,W)`. When the optional RESIDUAL block is enabled
//!   (`conv3: Some`, the round-3 higher-capacity arch), one more depth-preserving
//!   block is applied with an identity skip: `board_embed = tanh(conv3(trunk2)) +
//!   trunk2`; otherwise `board_embed = trunk2`. `global_embed =
//!   GlobalAvgPool(board_embed)`.
//! - Value head: `Dense(D->HV) -> tanh -> Dense(HV->1) -> tanh` scalar in [-1,1].
//! - Policy head (scored per candidate): per-candidate feature
//!   `concat( target_embed(D), global_embed(D), local(LOCAL), intent(II) )`
//!   `-> Dense(2D+LOCAL+II -> HP) -> tanh -> Dense(HP->1)` linear score.
//!   `target_embed` is the `board_embed` column at `(x,y)` (zeros for a None /
//!   "pass" target).
//!
//! The trunk is shared by both heads. The training method scores all candidates,
//! computes cross-entropy(softmax(scores), pi) + (value - z)^2, accumulates every
//! candidate's gradient back into `board_embed` (scattering the target column and
//! routing `global_embed` through the pool), adds the value head's contribution,
//! and backprops the *summed* `board_embed` grad through the trunk exactly once.
//!
//! All math is `f64`. The net derives serde so weights persist.

use serde::{Deserialize, Serialize};

use crate::cnn::{
    idx, tanh_backward, tanh_forward, tanh_forward_into, Conv2d, Dense, GlobalAvgPool,
};

// ---------------------------------------------------------------------------
// Net
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialNet {
    pub plane_count: usize,
    pub local_dim: usize,
    pub intent_dim: usize,
    /// Number of per-state SCALAR economy/strategy features concatenated onto the
    /// pooled `global_embed` BEFORE the value head (income, staffing, filled
    /// capacity, treasury-toward-Device, tile-lead, device-window, device
    /// countdown). 0 = legacy behaviour (value head pools planes only). These are
    /// *inputs* to `value_d1` (no parameters of their own, like `local`/`intent`
    /// for the policy head); the gradient does not flow back into them. Defaulted
    /// for backward-compatible deserialisation of pre-scalar checkpoints.
    #[serde(default)]
    pub value_scalar_dim: usize,
    /// Trunk hidden channels (first conv out).
    pub d1: usize,
    /// Embedding channels (second conv out = board_embed depth).
    pub d: usize,
    /// Value head hidden width.
    pub hv: usize,
    /// Policy head hidden width.
    pub hp: usize,

    // Trunk.
    pub conv1: Conv2d,
    pub conv2: Conv2d,
    /// OPTIONAL residual block (round-3 higher-capacity arch): a depth-preserving
    /// `Conv2d(D->D,3,1)` whose `tanh` output is ADDED to its input (identity skip)
    /// to form `board_embed`. `None` = legacy 2-conv trunk (`board_embed = trunk2`),
    /// for backward-compatible deserialisation of pre-residual checkpoints.
    #[serde(default)]
    pub conv3: Option<Conv2d>,
    #[serde(default)]
    pub pool: GlobalAvgPool,

    // Value head: Dense(D->HV) -> tanh -> Dense(HV->1) -> tanh.
    pub value_d1: Dense,
    pub value_d2: Dense,

    // Policy head: Dense(2D+LOCAL+II -> HP) -> tanh -> Dense(HP->1).
    pub policy_d1: Dense,
    pub policy_d2: Dense,
}

impl SpatialNet {
    /// Fully configurable constructor. Each layer gets a distinct seed derived
    /// from the base `seed` so init is deterministic but layers differ.
    #[allow(clippy::too_many_arguments)]
    pub fn new_seeded(
        plane_count: usize,
        local_dim: usize,
        intent_dim: usize,
        value_scalar_dim: usize,
        d1: usize,
        d: usize,
        hv: usize,
        hp: usize,
        seed: u64,
    ) -> Self {
        Self::new_seeded_arch(
            plane_count,
            local_dim,
            intent_dim,
            value_scalar_dim,
            d1,
            d,
            hv,
            hp,
            false,
            seed,
        )
    }

    /// Full constructor with an explicit `use_residual` toggle for the trunk's
    /// optional depth-preserving residual block (`Conv2d(D->D,3,1)` + identity skip).
    #[allow(clippy::too_many_arguments)]
    pub fn new_seeded_arch(
        plane_count: usize,
        local_dim: usize,
        intent_dim: usize,
        value_scalar_dim: usize,
        d1: usize,
        d: usize,
        hv: usize,
        hp: usize,
        use_residual: bool,
        seed: u64,
    ) -> Self {
        let conv1 = Conv2d::new_seeded(plane_count, d1, 3, 1, seed.wrapping_add(1));
        let conv2 = Conv2d::new_seeded(d1, d, 3, 1, seed.wrapping_add(2));
        let conv3 = if use_residual {
            Some(Conv2d::new_seeded(d, d, 3, 1, seed.wrapping_add(7)))
        } else {
            None
        };
        // Value head sees the pooled embedding PLUS the per-state scalar features.
        let value_d1 = Dense::new_seeded(d + value_scalar_dim, hv, seed.wrapping_add(3));
        let value_d2 = Dense::new_seeded(hv, 1, seed.wrapping_add(4));
        let policy_in = 2 * d + local_dim + intent_dim;
        let policy_d1 = Dense::new_seeded(policy_in, hp, seed.wrapping_add(5));
        let policy_d2 = Dense::new_seeded(hp, 1, seed.wrapping_add(6));
        SpatialNet {
            plane_count,
            local_dim,
            intent_dim,
            value_scalar_dim,
            d1,
            d,
            hv,
            hp,
            conv1,
            conv2,
            conv3,
            pool: GlobalAvgPool,
            value_d1,
            value_d2,
            policy_d1,
            policy_d2,
        }
    }

    /// Convenience constructor with the ROUND-3 higher-capacity arch
    /// (D1=32, D=48, HV=64, HP=64, + residual trunk block) ≈ 53.7k params.
    pub fn default_for(plane_count: usize, local_dim: usize, intent_dim: usize, seed: u64) -> Self {
        Self::new_seeded_arch(
            plane_count, local_dim, intent_dim, 0, 32, 48, 64, 64, true, seed,
        )
    }

    /// Like [`default_for`](Self::default_for) but with `value_scalar_dim` per-state
    /// scalar economy features fed into the value head. This is the constructor the
    /// `cnn_train` trainer uses, so its widths ARE the deployed arch.
    ///
    /// ROUND-3 capacity bump: the old default was the tiny `D1=16,D=24,HV=24,HP=24`
    /// 2-conv trunk (~9.8k params). Two full rounds confirmed win-rate-vs-HARD was
    /// pinned ~0.46 by NET CAPACITY, not the value-squash (which round 2 resolved
    /// without moving win-rate). The new default — `D1=32,D=48,HV=64,HP=64` + a
    /// depth-preserving residual block — is ≈ 53.7k params (~5.5×) at the trainer's
    /// 24-plane / local-18 / intent-12 / vsd-12 I/O (UNCHANGED).
    pub fn default_with_value_scalars(
        plane_count: usize,
        local_dim: usize,
        intent_dim: usize,
        value_scalar_dim: usize,
        seed: u64,
    ) -> Self {
        let mut net = Self::new_seeded_arch(
            plane_count,
            local_dim,
            intent_dim,
            value_scalar_dim,
            32, // d1
            48, // d
            64, // hv
            64, // hp
            true, // residual trunk block
            seed,
        );
        // DILATED RESIDUAL BLOCK: keep conv1/conv2 as dense k3-pad1 (so they ride the
        // bit-reproducible fast path) and dilate ONLY the conv3 residual block
        // (k3, dilation2, pad2 -> same HxW, RF 9x9 once stacked on the dense 5x5 trunk).
        // Same seed offset as the dense conv3 in `new_seeded_arch`.
        let d = net.d;
        // Dilation DISABLED for perf: a dilated conv routes through the slow general
        // path (the k3/pad1 fast path is dilation=1-only) -> ~7x slower self-play. The
        // C_DIST_TO_ENEMY_HQ/DEVICE gradient planes already give board-spanning vision
        // per-cell at zero cost, so the wider RF is redundant. Primitive kept + grad-
        // checked for future use; flip back to new_seeded_dilated(d,d,3,2,2,..) to re-enable.
        net.conv3 = Some(Conv2d::new_seeded(d, d, 3, 1, seed.wrapping_add(7)));
        net
    }

    /// The PRE-round-3 SMALL arch (`D1=16,D=24,HV=24,HP=24`, **no** residual block)
    /// with `value_scalar_dim` value-head scalars — the proven round-1/2 ~9.8k-param
    /// net. TRAINING-APPROACH §2.5 wants this for fast curriculum/reward iteration (the
    /// 5.5× capacity test was confounded by passivity, so it proved nothing). Same I/O
    /// as [`default_with_value_scalars`](Self::default_with_value_scalars); only the
    /// trunk widths and the absence of the residual block differ — a COLD-START arch.
    pub fn default_small_with_value_scalars(
        plane_count: usize,
        local_dim: usize,
        intent_dim: usize,
        value_scalar_dim: usize,
        seed: u64,
    ) -> Self {
        let mut net = Self::new_seeded_arch(
            plane_count,
            local_dim,
            intent_dim,
            value_scalar_dim,
            16,    // d1
            24,    // d
            24,    // hv
            24,    // hp
            false, // NO residual trunk block (the round-1/2 arch)
            seed,
        );
        // DILATED conv2: keep conv1 dense (k3-pad1) for the cheap first layer, dilate
        // the second conv (k3, dilation2, pad2 -> same HxW). Per-layer RF 5x5, stacked
        // on the dense conv1 3x3 -> effective 7x7 with no extra params/depth. Same
        // seed offset as the dense conv2 in `new_seeded_arch`.
        let (d1, d) = (net.d1, net.d);
        // Dilation DISABLED for perf (see default_with_value_scalars): distance planes
        // deliver board-spanning vision; primitive kept + grad-checked. Re-enable via
        // new_seeded_dilated(d1,d,3,2,2,..).
        let _ = d;
        net.conv2 = Conv2d::new_seeded(d1, d, 3, 1, seed.wrapping_add(2));
        net
    }

    /// Total scalar parameter count (all conv + dense weights and biases).
    pub fn param_count(&self) -> usize {
        self.conv1.weights.len()
            + self.conv1.bias.len()
            + self.conv2.weights.len()
            + self.conv2.bias.len()
            + self
                .conv3
                .as_ref()
                .map(|c| c.weights.len() + c.bias.len())
                .unwrap_or(0)
            + self.value_d1.weights.len()
            + self.value_d1.bias.len()
            + self.value_d2.weights.len()
            + self.value_d2.bias.len()
            + self.policy_d1.weights.len()
            + self.policy_d1.bias.len()
            + self.policy_d2.weights.len()
            + self.policy_d2.bias.len()
    }

    #[inline]
    fn policy_in_dim(&self) -> usize {
        2 * self.d + self.local_dim + self.intent_dim
    }

    // ----- Inference --------------------------------------------------------

    /// Run the shared trunk on a board, caching everything the backward pass
    /// needs. `planes` is `plane_count*h*w` in (C,H,W) layout.
    pub fn forward_board(&self, planes: &[f64], h: usize, w: usize) -> BoardCache {
        self.forward_board_scalars(planes, h, w, &[])
    }

    /// Like [`forward_board`](Self::forward_board) but also stows the per-state
    /// `value_scalars` (length must equal `self.value_scalar_dim`) in the cache so
    /// the value head can concatenate them onto the pooled embedding. The trunk /
    /// policy path is identical and ignores the scalars.
    pub fn forward_board_scalars(
        &self,
        planes: &[f64],
        h: usize,
        w: usize,
        value_scalars: &[f64],
    ) -> BoardCache {
        debug_assert_eq!(planes.len(), self.plane_count * h * w);
        debug_assert_eq!(value_scalars.len(), self.value_scalar_dim);
        // Reuse one scratch buffer for both conv pre-activations: conv1_pre is
        // consumed into conv1_act, then the buffer is reused for conv2_pre which
        // is consumed into board_embed. Numerically identical to the allocating
        // path (same conv math, same tanh).
        let mut pre = Vec::new();
        self.conv1.forward_into(planes, h, w, &mut pre); // D1*H*W
        let conv1_act = tanh_forward(&pre); // tanh
        self.conv2.forward_into(&conv1_act, h, w, &mut pre); // D*H*W
        let trunk2 = tanh_forward(&pre); // tanh(conv2) = trunk2 (D*H*W)
        // Optional residual block: res_act = tanh(conv3(trunk2));
        // board_embed = res_act + trunk2 (identity skip). Without it,
        // board_embed == trunk2 (legacy 2-conv trunk).
        let (res_act, board_embed) = match &self.conv3 {
            Some(conv3) => {
                conv3.forward_into(&trunk2, h, w, &mut pre); // D*H*W
                let res_act = tanh_forward(&pre); // tanh
                let board_embed: Vec<f64> = res_act
                    .iter()
                    .zip(trunk2.iter())
                    .map(|(&r, &t)| r + t)
                    .collect();
                (Some(res_act), board_embed)
            }
            None => (None, trunk2.clone()),
        };
        let global_embed = self.pool.forward(&board_embed, self.d, h, w); // D
        BoardCache {
            h,
            w,
            planes: planes.to_vec(),
            conv1_act,
            trunk2,
            res_act,
            board_embed,
            global_embed,
            value_scalars: value_scalars.to_vec(),
        }
    }

    /// Scalar value in [-1,1] from a cached board.
    pub fn value_from(&self, cache: &BoardCache) -> f64 {
        let v = self.value_forward(cache);
        v.value
    }

    /// Build the per-candidate policy input vector from a cached board.
    fn candidate_input(
        &self,
        cache: &BoardCache,
        target: Option<(usize, usize)>,
        local: &[f64],
        intent_onehot: &[f64],
    ) -> Vec<f64> {
        debug_assert_eq!(local.len(), self.local_dim);
        debug_assert_eq!(intent_onehot.len(), self.intent_dim);
        let d = self.d;
        let mut input = Vec::with_capacity(self.policy_in_dim());
        // target_embed: board_embed column at (x,y), or zeros for None.
        match target {
            Some((x, y)) => {
                for c in 0..d {
                    input.push(cache.board_embed[idx(c, y, x, cache.h, cache.w)]);
                }
            }
            None => input.extend(std::iter::repeat(0.0).take(d)),
        }
        // global_embed.
        input.extend_from_slice(&cache.global_embed);
        // local + intent.
        input.extend_from_slice(local);
        input.extend_from_slice(intent_onehot);
        input
    }

    /// Linear policy score for one candidate against a cached board.
    pub fn score_candidate(
        &self,
        cache: &BoardCache,
        target: Option<(usize, usize)>,
        local: &[f64],
        intent_onehot: &[f64],
    ) -> f64 {
        let input = self.candidate_input(cache, target, local, intent_onehot);
        let h1_pre = self.policy_d1.forward(&input);
        let h1 = tanh_forward(&h1_pre);
        self.policy_d2.forward(&h1)[0]
    }

    /// Allocation-free per-candidate score for the MCTS hot path. Reuses the
    /// caller-owned [`PolicyScratch`] buffers across many candidates of the same
    /// board. Numerically identical to [`score_candidate`](Self::score_candidate)
    /// — same concat input, same Dense+tanh math — just without per-call Vec
    /// allocations.
    pub fn score_candidate_into(
        &self,
        cache: &BoardCache,
        target: Option<(usize, usize)>,
        local: &[f64],
        intent_onehot: &[f64],
        scratch: &mut PolicyScratch,
    ) -> f64 {
        debug_assert_eq!(local.len(), self.local_dim);
        debug_assert_eq!(intent_onehot.len(), self.intent_dim);
        let d = self.d;
        let input = &mut scratch.input;
        input.clear();
        // target_embed: board_embed column at (x,y), or zeros for None.
        match target {
            Some((x, y)) => {
                for c in 0..d {
                    input.push(cache.board_embed[idx(c, y, x, cache.h, cache.w)]);
                }
            }
            None => input.extend(std::iter::repeat(0.0).take(d)),
        }
        input.extend_from_slice(&cache.global_embed);
        input.extend_from_slice(local);
        input.extend_from_slice(intent_onehot);
        self.policy_d1.forward_into(input, &mut scratch.h1_pre);
        tanh_forward_into(&scratch.h1_pre, &mut scratch.h1);
        self.policy_d2.forward_into(&scratch.h1, &mut scratch.out);
        scratch.out[0]
    }

    // ----- Internal forward helpers (retain activations) --------------------

    /// Value-head input = `global_embed (D) ⊕ value_scalars (value_scalar_dim)`.
    /// When `value_scalar_dim == 0` this is exactly `global_embed` (legacy path).
    fn value_input(&self, cache: &BoardCache) -> Vec<f64> {
        if self.value_scalar_dim == 0 {
            return cache.global_embed.clone();
        }
        let mut v = Vec::with_capacity(self.d + self.value_scalar_dim);
        v.extend_from_slice(&cache.global_embed);
        v.extend_from_slice(&cache.value_scalars);
        v
    }

    fn value_forward(&self, cache: &BoardCache) -> ValueFwd {
        let input = self.value_input(cache);
        let h1_pre = self.value_d1.forward(&input);
        let h1 = tanh_forward(&h1_pre);
        let out_pre = self.value_d2.forward(&h1); // len 1
        let value = out_pre[0].tanh();
        ValueFwd { h1, value }
    }

    fn policy_forward(&self, input: &[f64]) -> PolicyFwd {
        let h1_pre = self.policy_d1.forward(input);
        let h1 = tanh_forward(&h1_pre);
        let score = self.policy_d2.forward(&h1)[0];
        PolicyFwd { h1, score }
    }

    // ----- Training ---------------------------------------------------------

    /// One decision: a cached board (or recomputed planes), its candidates, an
    /// MCTS policy target `pi` (one prob per candidate, sums to 1) and a value
    /// target `z`. Returns the combined gradient w.r.t. *all* parameters plus
    /// the scalar (policy_loss, value_loss) for logging.
    ///
    /// `policy_loss = -sum_c pi_c * ln softmax(scores)_c`
    /// `value_loss  = (value - z)^2`
    /// The returned grad is of `policy_loss + value_loss`.
    pub fn train_grad(
        &self,
        planes: &[f64],
        h: usize,
        w: usize,
        candidates: &[(Option<(usize, usize)>, Vec<f64>, Vec<f64>)],
        pi: &[f64],
        z: f64,
    ) -> (SpatialGrad, f64, f64) {
        let cache = self.forward_board(planes, h, w);
        self.train_grad_cached(&cache, candidates, pi, z)
    }

    /// Like [`train_grad`](Self::train_grad) but with per-state value-head scalar
    /// features (length must equal `self.value_scalar_dim`). Use this whenever
    /// `value_scalar_dim > 0` so the value loss/grad sees the same scalars that
    /// inference does.
    pub fn train_grad_scalars(
        &self,
        planes: &[f64],
        h: usize,
        w: usize,
        value_scalars: &[f64],
        candidates: &[(Option<(usize, usize)>, Vec<f64>, Vec<f64>)],
        pi: &[f64],
        z: f64,
    ) -> (SpatialGrad, f64, f64) {
        let cache = self.forward_board_scalars(planes, h, w, value_scalars);
        self.train_grad_cached(&cache, candidates, pi, z)
    }

    /// VALUE-ONLY gradient for a state: trains the value head toward `z` and
    /// produces NO policy gradient (the policy head is left untouched). Use this for
    /// examples that have a clean value target but NO usable MCTS visit-policy — e.g.
    /// a scripted (HardAi) opponent seat's trajectory, whose winning states are a
    /// clean ±1 value signal but whose move choices are not the net's policy target.
    /// `candidates`/`pi` are NOT consulted; the returned `policy_loss` is 0.0.
    ///
    /// This is the safe alternative to passing an all-zero `pi` to
    /// [`train_grad_scalars`](Self::train_grad_scalars): with an all-zero `pi` the
    /// softmax cross-entropy gradient `p_c − pi_c = p_c` is NON-zero and would
    /// spuriously push every score down. This path skips the policy head entirely.
    pub fn train_grad_value_only_scalars(
        &self,
        planes: &[f64],
        h: usize,
        w: usize,
        value_scalars: &[f64],
        z: f64,
    ) -> (SpatialGrad, f64, f64) {
        let cache = self.forward_board_scalars(planes, h, w, value_scalars);
        self.train_grad_cached_inner(&cache, &[], &[], z, true, false)
    }

    /// POLICY-ONLY gradient: trains the policy head (+ shared trunk) toward the
    /// one-hot/visit policy `pi` and produces NO value gradient (value head + its
    /// contribution to the shared trunk are skipped). The mirror image of
    /// [`train_grad_value_only_scalars`]. Use for imitation/DAgger where the value
    /// target `z` is near-random (≈50/50 expert-vs-league) — a noisy value loss
    /// (stuck ≈0.78) keeps perturbing the SHARED conv trunk toward a useless
    /// constant, and with enough gradient steps that corruption dominates the policy
    /// head and re-collapses it to Pass (observed: 400-game round Pass 28%, but
    /// 800/2000-game rounds Pass 94-96%). Training the policy head alone removes that
    /// interference so the imitation signal survives at scale. `z`/value loss are
    /// reported as 0.0. Training-only (no forward-inference change) → parity-neutral.
    pub fn train_grad_policy_only_scalars(
        &self,
        planes: &[f64],
        h: usize,
        w: usize,
        value_scalars: &[f64],
        candidates: &[(Option<(usize, usize)>, Vec<f64>, Vec<f64>)],
        pi: &[f64],
    ) -> (SpatialGrad, f64, f64) {
        let cache = self.forward_board_scalars(planes, h, w, value_scalars);
        self.train_grad_cached_policy_only(&cache, candidates, pi)
    }

    /// Cached variant of [`train_grad_policy_only_scalars`].
    pub fn train_grad_cached_policy_only(
        &self,
        cache: &BoardCache,
        candidates: &[(Option<(usize, usize)>, Vec<f64>, Vec<f64>)],
        pi: &[f64],
    ) -> (SpatialGrad, f64, f64) {
        self.train_grad_cached_inner(cache, candidates, pi, 0.0, false, true)
    }

    /// Same as [`train_grad`](Self::train_grad) but reuses a prebuilt cache.
    pub fn train_grad_cached(
        &self,
        cache: &BoardCache,
        candidates: &[(Option<(usize, usize)>, Vec<f64>, Vec<f64>)],
        pi: &[f64],
        z: f64,
    ) -> (SpatialGrad, f64, f64) {
        self.train_grad_cached_inner(cache, candidates, pi, z, false, false)
    }

    /// Like [`train_grad_scalars`](Self::train_grad_scalars) but adds a
    /// FORWARD-KL anchor term to the policy loss. `anchor_pi` is the candidate-wise
    /// probability distribution from a FROZEN anchor net (`softmax` of its scores,
    /// same candidate ordering as `candidates`/`pi`). The added loss is:
    ///
    /// `kl_weight * KL(softmax(net_scores) || anchor_pi)`
    /// `       = kl_weight * sum_c p_c * (ln p_c - ln q_c)`
    ///
    /// The gradient wrt the net's logit `score_j` is the standard `p - pi`
    /// (cross-entropy of the hard target) PLUS `kl_weight * p_j * (ln(p_j/q_j) - KL)`
    /// (forward-KL term), accumulated into the same policy-head backward pass.
    /// `kl_weight = 0.0` is bit-identical to [`train_grad_scalars`].
    pub fn train_grad_scalars_kl_anchor(
        &self,
        planes: &[f64],
        h: usize,
        w: usize,
        value_scalars: &[f64],
        candidates: &[(Option<(usize, usize)>, Vec<f64>, Vec<f64>)],
        pi: &[f64],
        z: f64,
        anchor_pi: &[f64],
        kl_weight: f64,
    ) -> (SpatialGrad, f64, f64) {
        let cache = self.forward_board_scalars(planes, h, w, value_scalars);
        if kl_weight == 0.0 {
            return self.train_grad_cached_inner(&cache, candidates, pi, z, false, false);
        }
        debug_assert_eq!(anchor_pi.len(), candidates.len());
        self.train_grad_cached_kl_inner(&cache, candidates, pi, z, anchor_pi, kl_weight)
    }

    /// Forward-KL-augmented backward pass. Identical to `train_grad_cached_inner`
    /// (`value_only = false`) except: the policy gradient on each candidate's logit
    /// gets an additional `kl_weight * p_j * (ln(p_j/q_j) - KL)` term, and the
    /// returned `policy_loss` includes the `kl_weight * KL(p||q)` summand. The value
    /// head is unchanged.
    fn train_grad_cached_kl_inner(
        &self,
        cache: &BoardCache,
        candidates: &[(Option<(usize, usize)>, Vec<f64>, Vec<f64>)],
        pi: &[f64],
        z: f64,
        anchor_pi: &[f64],
        kl_weight: f64,
    ) -> (SpatialGrad, f64, f64) {
        debug_assert_eq!(candidates.len(), pi.len());
        debug_assert_eq!(candidates.len(), anchor_pi.len());
        let d = self.d;
        let (h, w) = (cache.h, cache.w);

        let mut grad = SpatialGrad::zeros_like(self);
        let mut grad_board_embed = vec![0.0f64; d * h * w];
        let mut grad_global = vec![0.0f64; d];

        // ----- Value head (identical to the standard backward) ---------------
        let vf = self.value_forward(cache);
        let value_loss = (vf.value - z) * (vf.value - z);
        let d_value = 2.0 * (vf.value - z);
        let grad_out_pre = vec![d_value * (1.0 - vf.value * vf.value)];
        let (grad_h1, vw2, vb2) = self.value_d2.backward(&vf.h1, &grad_out_pre);
        grad.value_d2_w = vw2;
        grad.value_d2_b = vb2;
        let grad_h1_pre = tanh_backward(&vf.h1, &grad_h1);
        let value_in = self.value_input(cache);
        let (grad_value_in, vw1, vb1) = self.value_d1.backward(&value_in, &grad_h1_pre);
        grad.value_d1_w = vw1;
        grad.value_d1_b = vb1;
        for c in 0..d {
            grad_global[c] += grad_value_in[c];
        }

        // ----- Policy head with CE + forward-KL ------------------------------
        let inputs: Vec<Vec<f64>> = candidates
            .iter()
            .map(|(tgt, local, intent)| self.candidate_input(cache, *tgt, local, intent))
            .collect();
        let fwds: Vec<PolicyFwd> = inputs.iter().map(|x| self.policy_forward(x)).collect();
        let scores: Vec<f64> = fwds.iter().map(|f| f.score).collect();
        let p = softmax(&scores);

        // CE loss + KL(p||q) loss.
        let mut ce_loss = 0.0f64;
        for c in 0..candidates.len() {
            if pi[c] > 0.0 {
                ce_loss += -pi[c] * p[c].max(1e-12).ln();
            }
        }
        // KL(p || q) = sum p (ln p - ln q). Floor q at 1e-12 for log stability.
        let mut kl_val = 0.0f64;
        for c in 0..candidates.len() {
            let qc = anchor_pi[c].max(1e-12);
            let pc = p[c].max(1e-12);
            kl_val += p[c] * (pc.ln() - qc.ln());
        }
        let policy_loss = ce_loss + kl_weight * kl_val;

        for c in 0..candidates.len() {
            // CE gradient: p - pi. KL forward gradient: kl_weight * p_c * (ln(p_c/q_c) - KL).
            let qc = anchor_pi[c].max(1e-12);
            let pc = p[c].max(1e-12);
            let kl_grad_c = kl_weight * p[c] * ((pc.ln() - qc.ln()) - kl_val);
            let upstream = (p[c] - pi[c]) + kl_grad_c;
            let grad_score = vec![upstream];
            let (grad_h1, pw2, pb2) = self.policy_d2.backward(&fwds[c].h1, &grad_score);
            accum(&mut grad.policy_d2_w, &pw2);
            accum(&mut grad.policy_d2_b, &pb2);
            let grad_h1_pre = tanh_backward(&fwds[c].h1, &grad_h1);
            let (grad_input, pw1, pb1) = self.policy_d1.backward(&inputs[c], &grad_h1_pre);
            accum(&mut grad.policy_d1_w, &pw1);
            accum(&mut grad.policy_d1_b, &pb1);
            if let Some((x, y)) = candidates[c].0 {
                for ch in 0..d {
                    grad_board_embed[idx(ch, y, x, h, w)] += grad_input[ch];
                }
            }
            for ch in 0..d {
                grad_global[ch] += grad_input[d + ch];
            }
        }

        // ----- Trunk backward (identical to the standard path) ---------------
        let grad_from_pool = self.pool.backward(&grad_global, d, h, w);
        for i in 0..grad_board_embed.len() {
            grad_board_embed[i] += grad_from_pool[i];
        }
        let grad_trunk2: Vec<f64> = match (&self.conv3, &cache.res_act) {
            (Some(conv3), Some(res_act)) => {
                let grad_res_pre = tanh_backward(res_act, &grad_board_embed);
                let (grad_into_trunk2, cw3, cb3) =
                    conv3.backward(&cache.trunk2, &grad_res_pre, h, w);
                grad.conv3_w = cw3;
                grad.conv3_b = cb3;
                grad_into_trunk2
                    .iter()
                    .zip(grad_board_embed.iter())
                    .map(|(&a, &b)| a + b)
                    .collect()
            }
            _ => grad_board_embed,
        };
        let grad_conv2_pre = tanh_backward(&cache.trunk2, &grad_trunk2);
        let (grad_conv1_act, cw2, cb2) =
            self.conv2.backward(&cache.conv1_act, &grad_conv2_pre, h, w);
        grad.conv2_w = cw2;
        grad.conv2_b = cb2;
        let grad_conv1_pre = tanh_backward(&cache.conv1_act, &grad_conv1_act);
        let (_grad_planes, cw1, cb1) = self.conv1.backward(&cache.planes, &grad_conv1_pre, h, w);
        grad.conv1_w = cw1;
        grad.conv1_b = cb1;

        (grad, policy_loss, value_loss)
    }

    /// PPO clipped-surrogate + entropy + value backward pass for ONE recorded
    /// decision (PPO-SPEC §3). TRAINING-ONLY / parity-FREE (the forward inference
    /// path is untouched). Modelled on [`train_grad_cached_kl_inner`] (same value
    /// head + trunk backward verbatim) but with a *custom policy upstream* derived
    /// from the PPO clipped objective instead of cross-entropy.
    ///
    /// Inputs (all from the FROZEN θ_old captured at collection time, except the
    /// net itself which is the CURRENT θ):
    ///   * `cache`        — `forward_board_scalars` cache of the recorded state under θ.
    ///   * `candidates`   — the per-candidate `(target, local, intent)` triples.
    ///   * `chosen`       — index of the action that was sampled at collection.
    ///   * `logp_old`     — ln π_old(chosen|s), captured under θ_old (τ=1 softmax).
    ///   * `adv`          — GAE advantage A_t (batch-normalised by the caller).
    ///   * `vtarg`        — GAE value target = A_t + V_old(s_t).
    ///   * `_v_old`       — V_old(s_t) (only used by the optional value-clip; off by
    ///                      default). Kept in the signature so the caller can pass it.
    ///   * `clip_eps`     — PPO ratio clip ε.
    ///   * `ent_coef`     — entropy bonus coefficient (subtracted from the loss).
    ///   * `val_coef`     — value-loss coefficient.
    ///   * `vclip`        — value-clip range (0 = OFF; the standard unclipped MSE).
    ///
    /// Returns `(grad, policy_loss, value_loss)` where:
    ///   * policy_loss = L_clip + L_ent (the reported surrogate incl. entropy term),
    ///   * value_loss  = (V_new − vtarg)^2  (unweighted, for logging),
    ///   * grad        = ∇θ of `L_clip + L_ent + val_coef·value_loss`.
    ///
    /// Policy math (PPO-SPEC §3):
    ///   r = exp(clamp(logp_new − logp_old, ±20));
    ///   L_clip = −min(r·A, clip(r,1−ε,1+ε)·A);
    ///   the clipped branch (grad 0) is active iff (A≥0 & r>1+ε) or (A<0 & r<1−ε),
    ///   else dL_clip/dlogp_new = −r·A.
    ///   ∂logp_new/∂s_c = [c==chosen] − p_c → policy upstream
    ///       g_c = (dL_clip/dlogp_new)·([c==chosen]−p_c) + ent_coef·p_c·(ln p_c + H).
    #[allow(clippy::too_many_arguments)]
    pub fn train_grad_ppo_cached(
        &self,
        cache: &BoardCache,
        candidates: &[(Option<(usize, usize)>, Vec<f64>, Vec<f64>)],
        chosen: usize,
        logp_old: f64,
        adv: f64,
        vtarg: f64,
        _v_old: f64,
        clip_eps: f64,
        ent_coef: f64,
        val_coef: f64,
        vclip: f64,
    ) -> (SpatialGrad, f64, f64) {
        debug_assert!(chosen < candidates.len());
        let d = self.d;
        let (h, w) = (cache.h, cache.w);

        let mut grad = SpatialGrad::zeros_like(self);
        let mut grad_board_embed = vec![0.0f64; d * h * w];
        let mut grad_global = vec![0.0f64; d];

        // ----- Value head ----------------------------------------------------
        // Standard MSE toward vtarg through the final tanh. Optional PPO value
        // clip (vclip>0): use the larger of the unclipped and clipped squared
        // error (the pessimistic bound), but ONLY when vclip is enabled.
        let vf = self.value_forward(cache);
        // Unclipped squared error (this is what we report regardless of vclip).
        let value_loss = (vf.value - vtarg) * (vf.value - vtarg);
        // d(value_loss_term)/d(value). With vclip off this is the plain MSE grad.
        let d_value = if vclip > 0.0 {
            // Clip V_new to V_old ± vclip; the gradient flows through whichever of
            // the unclipped / clipped squared error is LARGER (pessimistic max).
            let v_clipped = (vf.value).clamp(_v_old - vclip, _v_old + vclip);
            let err_unclipped = vf.value - vtarg;
            let err_clipped = v_clipped - vtarg;
            if err_unclipped * err_unclipped >= err_clipped * err_clipped {
                2.0 * err_unclipped
            } else {
                // Clipped branch: when v_clipped is at the clamp boundary the grad
                // wrt V_new is 0 (clamp saturated); inside the band it is the plain
                // 2·err. clamp saturates iff V_new is outside [V_old±vclip].
                if vf.value > _v_old + vclip || vf.value < _v_old - vclip {
                    0.0
                } else {
                    2.0 * err_clipped
                }
            }
        } else {
            2.0 * (vf.value - vtarg)
        };
        // Scale by val_coef and route through the tanh: value = tanh(out_pre).
        let grad_out_pre = vec![val_coef * d_value * (1.0 - vf.value * vf.value)];
        let (grad_h1, vw2, vb2) = self.value_d2.backward(&vf.h1, &grad_out_pre);
        grad.value_d2_w = vw2;
        grad.value_d2_b = vb2;
        let grad_h1_pre = tanh_backward(&vf.h1, &grad_h1);
        let value_in = self.value_input(cache);
        let (grad_value_in, vw1, vb1) = self.value_d1.backward(&value_in, &grad_h1_pre);
        grad.value_d1_w = vw1;
        grad.value_d1_b = vb1;
        for c in 0..d {
            grad_global[c] += grad_value_in[c];
        }

        // ----- Policy head: PPO clipped surrogate + entropy -------------------
        let inputs: Vec<Vec<f64>> = candidates
            .iter()
            .map(|(tgt, local, intent)| self.candidate_input(cache, *tgt, local, intent))
            .collect();
        let fwds: Vec<PolicyFwd> = inputs.iter().map(|x| self.policy_forward(x)).collect();
        let scores: Vec<f64> = fwds.iter().map(|f| f.score).collect();
        let p = softmax(&scores);

        // logp_new under the CURRENT net (τ=1 softmax). Clamp the log-ratio to ±20
        // before exp (PPO-SPEC §8 ratio-explosion guard).
        let logp_new = p[chosen].max(1e-12).ln();
        let log_ratio = (logp_new - logp_old).clamp(-20.0, 20.0);
        let r = log_ratio.exp();

        // Clipped surrogate. L_clip = −min(r·A, clip(r,1−ε,1+ε)·A).
        let lo = 1.0 - clip_eps;
        let hi = 1.0 + clip_eps;
        // The clipped branch is ACTIVE (zero policy gradient) iff:
        //   (A ≥ 0 & r > 1+ε)  or  (A < 0 & r < 1−ε).
        let clip_active = (adv >= 0.0 && r > hi) || (adv < 0.0 && r < lo);
        let l_clip = {
            let unclipped = r * adv;
            let clipped = r.clamp(lo, hi) * adv;
            -unclipped.min(clipped)
        };
        // dL_clip/dlogp_new: 0 in the clipped branch, else −r·A
        // (since d r/d logp_new = r).
        let dlclip_dlogpnew = if clip_active { 0.0 } else { -r * adv };

        // Entropy bonus: H = −Σ p_c ln p_c; L_ent = −ent_coef·H (subtract entropy
        // so MINIMISING the loss MAXIMISES entropy). ∂L_ent/∂s_j = ent_coef·p_j·(ln p_j + H).
        let mut entropy = 0.0f64;
        for &pc in &p {
            if pc > 0.0 {
                entropy -= pc * pc.ln();
            }
        }
        let l_ent = -ent_coef * entropy;
        let policy_loss = l_clip + l_ent;

        // Per-candidate policy upstream g_c on its logit score_c:
        //   clip term : dlclip_dlogpnew · ([c==chosen] − p_c)
        //   entropy   : ent_coef · p_c · (ln p_c + H)
        for c in 0..candidates.len() {
            let indicator = if c == chosen { 1.0 } else { 0.0 };
            let g_clip = dlclip_dlogpnew * (indicator - p[c]);
            let g_ent = if p[c] > 0.0 {
                ent_coef * p[c] * (p[c].ln() + entropy)
            } else {
                0.0
            };
            let upstream = g_clip + g_ent;
            let grad_score = vec![upstream];
            let (grad_h1, pw2, pb2) = self.policy_d2.backward(&fwds[c].h1, &grad_score);
            accum(&mut grad.policy_d2_w, &pw2);
            accum(&mut grad.policy_d2_b, &pb2);
            let grad_h1_pre = tanh_backward(&fwds[c].h1, &grad_h1);
            let (grad_input, pw1, pb1) = self.policy_d1.backward(&inputs[c], &grad_h1_pre);
            accum(&mut grad.policy_d1_w, &pw1);
            accum(&mut grad.policy_d1_b, &pb1);
            if let Some((x, y)) = candidates[c].0 {
                for ch in 0..d {
                    grad_board_embed[idx(ch, y, x, h, w)] += grad_input[ch];
                }
            }
            for ch in 0..d {
                grad_global[ch] += grad_input[d + ch];
            }
        }

        // ----- Trunk backward (identical to the standard path) ---------------
        let grad_from_pool = self.pool.backward(&grad_global, d, h, w);
        for i in 0..grad_board_embed.len() {
            grad_board_embed[i] += grad_from_pool[i];
        }
        let grad_trunk2: Vec<f64> = match (&self.conv3, &cache.res_act) {
            (Some(conv3), Some(res_act)) => {
                let grad_res_pre = tanh_backward(res_act, &grad_board_embed);
                let (grad_into_trunk2, cw3, cb3) =
                    conv3.backward(&cache.trunk2, &grad_res_pre, h, w);
                grad.conv3_w = cw3;
                grad.conv3_b = cb3;
                grad_into_trunk2
                    .iter()
                    .zip(grad_board_embed.iter())
                    .map(|(&a, &b)| a + b)
                    .collect()
            }
            _ => grad_board_embed,
        };
        let grad_conv2_pre = tanh_backward(&cache.trunk2, &grad_trunk2);
        let (grad_conv1_act, cw2, cb2) =
            self.conv2.backward(&cache.conv1_act, &grad_conv2_pre, h, w);
        grad.conv2_w = cw2;
        grad.conv2_b = cb2;
        let grad_conv1_pre = tanh_backward(&cache.conv1_act, &grad_conv1_act);
        let (_grad_planes, cw1, cb1) = self.conv1.backward(&cache.planes, &grad_conv1_pre, h, w);
        grad.conv1_w = cw1;
        grad.conv1_b = cb1;

        (grad, policy_loss, value_loss)
    }

    /// Compute the policy-head probabilities (softmax over candidate scores) for an
    /// anchor net evaluation of `(planes, value_scalars, candidates)`. Read-only —
    /// used by the KL-anchor training path to produce frozen targets per batch.
    pub fn policy_probs_scalars(
        &self,
        planes: &[f64],
        h: usize,
        w: usize,
        value_scalars: &[f64],
        candidates: &[(Option<(usize, usize)>, Vec<f64>, Vec<f64>)],
    ) -> Vec<f64> {
        let cache = self.forward_board_scalars(planes, h, w, value_scalars);
        let mut scratch = PolicyScratch::new();
        let scores: Vec<f64> = candidates
            .iter()
            .map(|(tgt, local, intent)| {
                self.score_candidate_into(&cache, *tgt, local, intent, &mut scratch)
            })
            .collect();
        softmax(&scores)
    }

    /// Shared implementation of the cached backward pass. When `value_only` is true
    /// the policy head is skipped entirely (no policy loss, no policy gradient) and
    /// `candidates`/`pi` are ignored — only the value head trains toward `z`.
    fn train_grad_cached_inner(
        &self,
        cache: &BoardCache,
        candidates: &[(Option<(usize, usize)>, Vec<f64>, Vec<f64>)],
        pi: &[f64],
        z: f64,
        value_only: bool,
        policy_only: bool,
    ) -> (SpatialGrad, f64, f64) {
        debug_assert!(value_only || candidates.len() == pi.len());
        debug_assert!(!(value_only && policy_only));
        let d = self.d;
        let (h, w) = (cache.h, cache.w);

        let mut grad = SpatialGrad::zeros_like(self);

        // grad accumulated into board_embed (D*H*W) from every head/candidate.
        let mut grad_board_embed = vec![0.0f64; d * h * w];
        // grad accumulated into global_embed (D) — routed through the pool at the
        // end into grad_board_embed.
        let mut grad_global = vec![0.0f64; d];

        // ----- Value head -----------------------------------------------------
        // Skipped entirely for policy-only examples (no value target / no value grad,
        // and crucially no value contribution to the SHARED trunk via grad_global).
        let mut value_loss = 0.0f64;
        if !policy_only {
        let vf = self.value_forward(cache);
        value_loss = (vf.value - z) * (vf.value - z);
        // d(value_loss)/d(value) = 2*(value - z); through final tanh:
        // value = tanh(out_pre), d value/d out_pre = 1 - value^2.
        let d_value = 2.0 * (vf.value - z);
        let grad_out_pre = vec![d_value * (1.0 - vf.value * vf.value)]; // len 1
        let (grad_h1, vw2, vb2) = self.value_d2.backward(&vf.h1, &grad_out_pre);
        grad.value_d2_w = vw2;
        grad.value_d2_b = vb2;
        // through tanh of value hidden.
        let grad_h1_pre = tanh_backward(&vf.h1, &grad_h1);
        // value_d1 input = global_embed (D) ⊕ value_scalars (value_scalar_dim).
        // Backprop through it, then take only the first D of the input-grad to route
        // into global_embed; the scalar segment is a non-parameter caller input and
        // is dropped (no input-grad exposed), exactly like `local`/`intent` in the
        // policy head.
        let value_in = self.value_input(cache);
        let (grad_value_in, vw1, vb1) = self.value_d1.backward(&value_in, &grad_h1_pre);
        grad.value_d1_w = vw1;
        grad.value_d1_b = vb1;
        for c in 0..d {
            grad_global[c] += grad_value_in[c];
        }
        } // end `if !policy_only` (value head)

        // ----- Policy head (per candidate) ------------------------------------
        // Scores then softmax cross-entropy: dL/d score_c = p_c - pi_c.
        // Skipped entirely for value-only examples (no usable policy target).
        let mut policy_loss = 0.0f64;
        if !value_only {
        let inputs: Vec<Vec<f64>> = candidates
            .iter()
            .map(|(tgt, local, intent)| self.candidate_input(cache, *tgt, local, intent))
            .collect();
        let fwds: Vec<PolicyFwd> = inputs.iter().map(|x| self.policy_forward(x)).collect();
        let scores: Vec<f64> = fwds.iter().map(|f| f.score).collect();
        let p = softmax(&scores);

        for c in 0..candidates.len() {
            if pi[c] > 0.0 {
                policy_loss += -pi[c] * p[c].max(1e-12).ln();
            }
        }

        for c in 0..candidates.len() {
            let upstream = p[c] - pi[c]; // dL/d score_c
            // policy_d2: Dense(HP->1), score = policy_d2(h1).
            let grad_score = vec![upstream];
            let (grad_h1, pw2, pb2) = self.policy_d2.backward(&fwds[c].h1, &grad_score);
            accum(&mut grad.policy_d2_w, &pw2);
            accum(&mut grad.policy_d2_b, &pb2);
            // through tanh of policy hidden.
            let grad_h1_pre = tanh_backward(&fwds[c].h1, &grad_h1);
            let (grad_input, pw1, pb1) = self.policy_d1.backward(&inputs[c], &grad_h1_pre);
            accum(&mut grad.policy_d1_w, &pw1);
            accum(&mut grad.policy_d1_b, &pb1);

            // Split grad_input back into its concat segments:
            //   [0..d)            -> target_embed
            //   [d..2d)           -> global_embed
            //   [2d..2d+local)    -> local (caller feature, no param here)
            //   [2d+local..)      -> intent (no param)
            // target_embed grad scatters into grad_board_embed at (x,y) col.
            if let Some((x, y)) = candidates[c].0 {
                for ch in 0..d {
                    grad_board_embed[idx(ch, y, x, h, w)] += grad_input[ch];
                }
            }
            // global_embed grad accumulates; routed through pool below.
            for ch in 0..d {
                grad_global[ch] += grad_input[d + ch];
            }
            // local / intent grads are inputs from the caller; not parameters,
            // so we drop them (no input-grad is exposed by this API).
        }
        } // end `if !value_only` (policy head)

        // ----- Route global_embed grad through the pool into board_embed ------
        let grad_from_pool = self.pool.backward(&grad_global, d, h, w); // D*H*W
        for i in 0..grad_board_embed.len() {
            grad_board_embed[i] += grad_from_pool[i];
        }

        // ----- Trunk backward -------------------------------------------------
        // Residual block: board_embed = res_act + trunk2, res_act = tanh(conv3(trunk2)).
        //   d board_embed/d res_act = 1, d board_embed/d trunk2 = 1 (skip).
        // So grad flows to res_act (== grad_board_embed) through conv3, AND directly
        // to trunk2 via the identity skip. Without a residual block, grad_trunk2 ==
        // grad_board_embed (board_embed == trunk2).
        let grad_trunk2: Vec<f64> = match (&self.conv3, &cache.res_act) {
            (Some(conv3), Some(res_act)) => {
                // tanh backward of the residual conv: res_act = tanh(conv3_pre).
                let grad_res_pre = tanh_backward(res_act, &grad_board_embed);
                let (grad_into_trunk2, cw3, cb3) =
                    conv3.backward(&cache.trunk2, &grad_res_pre, h, w);
                grad.conv3_w = cw3;
                grad.conv3_b = cb3;
                // Sum the conv3 path's input-grad with the identity-skip grad.
                grad_into_trunk2
                    .iter()
                    .zip(grad_board_embed.iter())
                    .map(|(&a, &b)| a + b)
                    .collect()
            }
            _ => grad_board_embed,
        };
        // trunk2 = tanh(conv2_pre). Backprop tanh.
        let grad_conv2_pre = tanh_backward(&cache.trunk2, &grad_trunk2);
        let (grad_conv1_act, cw2, cb2) =
            self.conv2.backward(&cache.conv1_act, &grad_conv2_pre, h, w);
        grad.conv2_w = cw2;
        grad.conv2_b = cb2;
        // conv1_act = tanh(conv1_pre). Backprop tanh.
        let grad_conv1_pre = tanh_backward(&cache.conv1_act, &grad_conv1_act);
        let (_grad_planes, cw1, cb1) = self.conv1.backward(&cache.planes, &grad_conv1_pre, h, w);
        grad.conv1_w = cw1;
        grad.conv1_b = cb1;

        (grad, policy_loss, value_loss)
    }

    /// Plain SGD step with L2 weight decay: `p -= lr * (grad + l2 * p)`.
    /// L2 is applied to weights only (not biases), matching the usual idiom.
    pub fn apply_grad(&mut self, grad: &SpatialGrad, lr: f64, l2: f64) {
        sgd(&mut self.conv1.weights, &grad.conv1_w, lr, l2);
        sgd(&mut self.conv1.bias, &grad.conv1_b, lr, 0.0);
        sgd(&mut self.conv2.weights, &grad.conv2_w, lr, l2);
        sgd(&mut self.conv2.bias, &grad.conv2_b, lr, 0.0);
        if let Some(conv3) = self.conv3.as_mut() {
            sgd(&mut conv3.weights, &grad.conv3_w, lr, l2);
            sgd(&mut conv3.bias, &grad.conv3_b, lr, 0.0);
        }
        sgd(&mut self.value_d1.weights, &grad.value_d1_w, lr, l2);
        sgd(&mut self.value_d1.bias, &grad.value_d1_b, lr, 0.0);
        sgd(&mut self.value_d2.weights, &grad.value_d2_w, lr, l2);
        sgd(&mut self.value_d2.bias, &grad.value_d2_b, lr, 0.0);
        sgd(&mut self.policy_d1.weights, &grad.policy_d1_w, lr, l2);
        sgd(&mut self.policy_d1.bias, &grad.policy_d1_b, lr, 0.0);
        sgd(&mut self.policy_d2.weights, &grad.policy_d2_w, lr, l2);
        sgd(&mut self.policy_d2.bias, &grad.policy_d2_b, lr, 0.0);
    }
}

// ---------------------------------------------------------------------------
// Cache + intermediate-forward structs
// ---------------------------------------------------------------------------

/// Cached trunk output for a single board, holding everything the backward pass
/// needs (pre-pool activations + the input planes).
#[derive(Debug, Clone)]
pub struct BoardCache {
    pub h: usize,
    pub w: usize,
    /// Input planes (C,H,W) — retained for the conv1 backward.
    pub planes: Vec<f64>,
    /// tanh(conv1) activation (D1,H,W) — input to conv2 backward.
    pub conv1_act: Vec<f64>,
    /// trunk2 = tanh(conv2) (D,H,W) — the residual block's INPUT and skip path,
    /// retained for conv3/skip backward. When there is no residual block this is
    /// identical to `board_embed`.
    pub trunk2: Vec<f64>,
    /// Residual-block activation res_act = tanh(conv3(trunk2)) (D,H,W). `None` when
    /// the net has no residual block. Retained for the conv3 tanh backward.
    pub res_act: Option<Vec<f64>>,
    /// board_embed (D,H,W). With a residual block: `res_act + trunk2`; else `trunk2`.
    pub board_embed: Vec<f64>,
    /// GlobalAvgPool(board_embed) (D).
    pub global_embed: Vec<f64>,
    /// Per-state scalar economy/strategy features concatenated onto `global_embed`
    /// before the value head. Length == `SpatialNet::value_scalar_dim` (empty for
    /// legacy planes-only value heads). Inputs only — not backpropped into.
    pub value_scalars: Vec<f64>,
}

/// Reusable scratch buffers for [`SpatialNet::score_candidate_into`]. Construct
/// once per node (or per thread) and reuse across all candidates to avoid
/// per-candidate Vec allocation. Buffers auto-resize on first use.
#[derive(Debug, Clone, Default)]
pub struct PolicyScratch {
    input: Vec<f64>,
    h1_pre: Vec<f64>,
    h1: Vec<f64>,
    out: Vec<f64>,
}

impl PolicyScratch {
    pub fn new() -> Self {
        Self::default()
    }
}

struct ValueFwd {
    h1: Vec<f64>,
    value: f64,
}

struct PolicyFwd {
    h1: Vec<f64>,
    score: f64,
}

// ---------------------------------------------------------------------------
// Gradient struct (mirrors the param layout)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SpatialGrad {
    pub conv1_w: Vec<f64>,
    pub conv1_b: Vec<f64>,
    pub conv2_w: Vec<f64>,
    pub conv2_b: Vec<f64>,
    /// Residual-block conv grads (empty when the net has no residual block).
    pub conv3_w: Vec<f64>,
    pub conv3_b: Vec<f64>,
    pub value_d1_w: Vec<f64>,
    pub value_d1_b: Vec<f64>,
    pub value_d2_w: Vec<f64>,
    pub value_d2_b: Vec<f64>,
    pub policy_d1_w: Vec<f64>,
    pub policy_d1_b: Vec<f64>,
    pub policy_d2_w: Vec<f64>,
    pub policy_d2_b: Vec<f64>,
}

impl SpatialGrad {
    /// Zero-initialised grad matching `net`'s parameter shapes.
    pub fn zeros_like(net: &SpatialNet) -> Self {
        SpatialGrad {
            conv1_w: vec![0.0; net.conv1.weights.len()],
            conv1_b: vec![0.0; net.conv1.bias.len()],
            conv2_w: vec![0.0; net.conv2.weights.len()],
            conv2_b: vec![0.0; net.conv2.bias.len()],
            conv3_w: vec![0.0; net.conv3.as_ref().map(|c| c.weights.len()).unwrap_or(0)],
            conv3_b: vec![0.0; net.conv3.as_ref().map(|c| c.bias.len()).unwrap_or(0)],
            value_d1_w: vec![0.0; net.value_d1.weights.len()],
            value_d1_b: vec![0.0; net.value_d1.bias.len()],
            value_d2_w: vec![0.0; net.value_d2.weights.len()],
            value_d2_b: vec![0.0; net.value_d2.bias.len()],
            policy_d1_w: vec![0.0; net.policy_d1.weights.len()],
            policy_d1_b: vec![0.0; net.policy_d1.bias.len()],
            policy_d2_w: vec![0.0; net.policy_d2.weights.len()],
            policy_d2_b: vec![0.0; net.policy_d2.bias.len()],
        }
    }

    /// In-place add of another grad (for batch accumulation).
    pub fn add(&mut self, other: &SpatialGrad) {
        accum(&mut self.conv1_w, &other.conv1_w);
        accum(&mut self.conv1_b, &other.conv1_b);
        accum(&mut self.conv2_w, &other.conv2_w);
        accum(&mut self.conv2_b, &other.conv2_b);
        accum(&mut self.conv3_w, &other.conv3_w);
        accum(&mut self.conv3_b, &other.conv3_b);
        accum(&mut self.value_d1_w, &other.value_d1_w);
        accum(&mut self.value_d1_b, &other.value_d1_b);
        accum(&mut self.value_d2_w, &other.value_d2_w);
        accum(&mut self.value_d2_b, &other.value_d2_b);
        accum(&mut self.policy_d1_w, &other.policy_d1_w);
        accum(&mut self.policy_d1_b, &other.policy_d1_b);
        accum(&mut self.policy_d2_w, &other.policy_d2_w);
        accum(&mut self.policy_d2_b, &other.policy_d2_b);
    }

    /// Scale all grads by a scalar (e.g. 1/batch).
    pub fn scale(&mut self, s: f64) {
        for v in [
            &mut self.conv1_w,
            &mut self.conv1_b,
            &mut self.conv2_w,
            &mut self.conv2_b,
            &mut self.conv3_w,
            &mut self.conv3_b,
            &mut self.value_d1_w,
            &mut self.value_d1_b,
            &mut self.value_d2_w,
            &mut self.value_d2_b,
            &mut self.policy_d1_w,
            &mut self.policy_d1_b,
            &mut self.policy_d2_w,
            &mut self.policy_d2_b,
        ] {
            for x in v.iter_mut() {
                *x *= s;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

#[inline]
fn accum(dst: &mut [f64], src: &[f64]) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d += *s;
    }
}

#[inline]
fn sgd(params: &mut [f64], grad: &[f64], lr: f64, l2: f64) {
    for (p, g) in params.iter_mut().zip(grad.iter()) {
        let gd = if l2 != 0.0 { *g + l2 * *p } else { *g };
        *p -= lr * gd;
    }
}

/// Numerically stable softmax.
fn softmax(scores: &[f64]) -> Vec<f64> {
    let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scores.iter().map(|&s| (s - m).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

// ---------------------------------------------------------------------------
// Tests — finite-difference gradient checks of the COMBINED training loss.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-5;
    const TOL: f64 = 1e-4;

    fn fill(n: usize, seed: u64) -> Vec<f64> {
        // Deterministic [-1,1) fill via a tiny xorshift (independent of cnn::Lcg).
        let mut s = seed ^ 0x9e37_79b9_7f4a_7c15;
        if s == 0 {
            s = 1;
        }
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
            })
            .collect()
    }

    fn assert_close(a: f64, b: f64, what: &str) {
        assert!(
            (a - b).abs() < TOL,
            "{what}: analytic {a} vs numeric {b} (diff {})",
            (a - b).abs()
        );
    }

    // ---- Test 1: shapes / sanity ----------------------------------------

    #[test]
    fn shapes_and_none_vs_target() {
        let (pc, h, w) = (4usize, 4usize, 5usize);
        let local_dim = 6;
        let intent_dim = 3;
        let net = SpatialNet::default_for(pc, local_dim, intent_dim, 1234);
        let planes = fill(pc * h * w, 1);
        let cache = net.forward_board(&planes, h, w);

        assert_eq!(cache.board_embed.len(), net.d * h * w);
        assert_eq!(cache.global_embed.len(), net.d);
        assert_eq!(cache.h, h);
        assert_eq!(cache.w, w);

        let val = net.value_from(&cache);
        assert!(val.is_finite());
        assert!(val > -1.0 && val < 1.0, "value in (-1,1): {val}");

        let local = fill(local_dim, 2);
        let intent = fill(intent_dim, 3);
        let s_target = net.score_candidate(&cache, Some((2, 1)), &local, &intent);
        let s_none = net.score_candidate(&cache, None, &local, &intent);
        assert!(s_target.is_finite() && s_none.is_finite());
        // With identical local/intent, the only difference is the target_embed
        // column vs zeros, so the scores should differ (board_embed is non-zero).
        assert!(
            (s_target - s_none).abs() > 1e-9,
            "None vs targeted should differ: {s_target} vs {s_none}"
        );
    }

    // ---- Test 2: combined-loss finite-difference gradient check ----------

    /// Recompute the combined loss for the current net params — used by the FD
    /// loop. Mirrors `train_grad_cached` losses exactly.
    fn combined_loss(
        net: &SpatialNet,
        planes: &[f64],
        h: usize,
        w: usize,
        cands: &[(Option<(usize, usize)>, Vec<f64>, Vec<f64>)],
        pi: &[f64],
        z: f64,
    ) -> f64 {
        let cache = net.forward_board(planes, h, w);
        // value loss
        let value = net.value_from(&cache);
        let value_loss = (value - z) * (value - z);
        // policy loss
        let scores: Vec<f64> = cands
            .iter()
            .map(|(t, l, i)| net.score_candidate(&cache, *t, l, i))
            .collect();
        let p = softmax(&scores);
        let mut policy_loss = 0.0;
        for c in 0..cands.len() {
            if pi[c] > 0.0 {
                policy_loss += -pi[c] * p[c].max(1e-12).ln();
            }
        }
        policy_loss + value_loss
    }

    #[test]
    fn combined_grad_finite_difference() {
        let (pc, h, w) = (3usize, 3usize, 4usize);
        let local_dim = 2usize;
        let intent_dim = 2usize;
        // Tiny net: D1=3, D=4, HV=4, HP=4.
        let mut net = SpatialNet::new_seeded(pc, local_dim, intent_dim, 0, 3, 4, 4, 4, 777);
        let planes = fill(pc * h * w, 11);

        // 3 candidates: mix of Some / None targets.
        let cands: Vec<(Option<(usize, usize)>, Vec<f64>, Vec<f64>)> = vec![
            (Some((1, 0)), fill(local_dim, 21), fill(intent_dim, 31)),
            (None, fill(local_dim, 22), fill(intent_dim, 32)),
            (Some((3, 2)), fill(local_dim, 23), fill(intent_dim, 33)),
        ];
        let pi = vec![0.5, 0.3, 0.2];
        let z = 0.4;

        let (grad, ploss, vloss) = net.train_grad(&planes, h, w, &cands, &pi, z);
        assert!(ploss.is_finite() && vloss.is_finite());
        assert!(ploss >= 0.0);

        // Closure to FD a single parameter slice given accessors.
        macro_rules! check_param {
            ($field:expr, $gradvec:expr, $name:expr) => {{
                let n = $field.len();
                // Sample up to a handful of indices to keep the test fast but
                // cover the slice.
                let stride = (n / 6).max(1);
                let mut j = 0;
                while j < n {
                    let save = $field[j];
                    $field[j] = save + EPS;
                    let lp = combined_loss(&net, &planes, h, w, &cands, &pi, z);
                    $field[j] = save - EPS;
                    let lm = combined_loss(&net, &planes, h, w, &cands, &pi, z);
                    $field[j] = save;
                    let num = (lp - lm) / (2.0 * EPS);
                    assert_close($gradvec[j], num, $name);
                    j += stride;
                }
            }};
        }

        // Both conv layers' weights + biases.
        check_param!(net.conv1.weights, grad.conv1_w, "conv1_w");
        check_param!(net.conv1.bias, grad.conv1_b, "conv1_b");
        check_param!(net.conv2.weights, grad.conv2_w, "conv2_w");
        check_param!(net.conv2.bias, grad.conv2_b, "conv2_b");
        // Value-head denses.
        check_param!(net.value_d1.weights, grad.value_d1_w, "value_d1_w");
        check_param!(net.value_d1.bias, grad.value_d1_b, "value_d1_b");
        check_param!(net.value_d2.weights, grad.value_d2_w, "value_d2_w");
        check_param!(net.value_d2.bias, grad.value_d2_b, "value_d2_b");
        // Policy-head denses.
        check_param!(net.policy_d1.weights, grad.policy_d1_w, "policy_d1_w");
        check_param!(net.policy_d1.bias, grad.policy_d1_b, "policy_d1_b");
        check_param!(net.policy_d2.weights, grad.policy_d2_w, "policy_d2_w");
        check_param!(net.policy_d2.bias, grad.policy_d2_b, "policy_d2_b");
    }

    // ---- Test 2a': FD gradient check with a DILATED trunk ----------------
    //
    // Builds a net whose conv2 is dilated (k3/dil2/pad2 -> same HxW), exactly like
    // `default_small_with_value_scalars`, and FD-checks the combined value+policy
    // loss gradient end-to-end through every parameter. This exercises the dilated
    // forward/backward inside the full trunk + heads.
    #[test]
    fn combined_grad_finite_difference_dilated() {
        let (pc, h, w) = (3usize, 5usize, 6usize); // >=5 so the 5x5 footprint has interior
        let local_dim = 2usize;
        let intent_dim = 2usize;
        // Tiny net (D1=3, D=4, HV=4, HP=4), NO residual, then dilate conv2.
        let mut net = SpatialNet::new_seeded(pc, local_dim, intent_dim, 0, 3, 4, 4, 4, 777);
        let (d1, d) = (net.d1, net.d);
        net.conv2 = Conv2d::new_seeded_dilated(d1, d, 3, 2, 2, 999);
        assert_eq!(net.conv2.dilation, 2, "conv2 must be dilated for this test");
        let planes = fill(pc * h * w, 11);

        let cands: Vec<(Option<(usize, usize)>, Vec<f64>, Vec<f64>)> = vec![
            (Some((1, 0)), fill(local_dim, 21), fill(intent_dim, 31)),
            (None, fill(local_dim, 22), fill(intent_dim, 32)),
            (Some((4, 3)), fill(local_dim, 23), fill(intent_dim, 33)),
        ];
        let pi = vec![0.5, 0.3, 0.2];
        let z = 0.4;

        let (grad, ploss, vloss) = net.train_grad(&planes, h, w, &cands, &pi, z);
        assert!(ploss.is_finite() && vloss.is_finite() && ploss >= 0.0);

        macro_rules! check_param {
            ($field:expr, $gradvec:expr, $name:expr) => {{
                let n = $field.len();
                let stride = (n / 6).max(1);
                let mut j = 0;
                while j < n {
                    let save = $field[j];
                    $field[j] = save + EPS;
                    let lp = combined_loss(&net, &planes, h, w, &cands, &pi, z);
                    $field[j] = save - EPS;
                    let lm = combined_loss(&net, &planes, h, w, &cands, &pi, z);
                    $field[j] = save;
                    let num = (lp - lm) / (2.0 * EPS);
                    assert_close($gradvec[j], num, $name);
                    j += stride;
                }
            }};
        }
        // The dilated conv2 weights/bias are the critical check; also FD the rest of
        // the chain so grad routes correctly through the dilated layer.
        check_param!(net.conv2.weights, grad.conv2_w, "dil_conv2_w");
        check_param!(net.conv2.bias, grad.conv2_b, "dil_conv2_b");
        check_param!(net.conv1.weights, grad.conv1_w, "dil_conv1_w");
        check_param!(net.value_d1.weights, grad.value_d1_w, "dil_value_d1_w");
        check_param!(net.policy_d1.weights, grad.policy_d1_w, "dil_policy_d1_w");
        check_param!(net.policy_d2.weights, grad.policy_d2_w, "dil_policy_d2_w");
    }

    // ---- Test 2b: FD gradient check at local_dim=18 ---------------------
    //
    // cnn_train now builds the SpatialNet with local_dim = LOCAL_DIM(16) + 2
    // remaining-capacity features = 18. The combined-loss gradient is already
    // checked for an arbitrary local_dim above (the math is dim-agnostic); this
    // pins the exact 18-dim policy-input width the trainer uses so a future
    // regression in the concat/scatter at that width is caught.
    #[test]
    fn combined_grad_finite_difference_local_dim_18() {
        let (pc, h, w) = (3usize, 3usize, 4usize);
        let local_dim = 18usize; // LOCAL_DIM(16) + 2 capacity features
        let intent_dim = 12usize;
        let mut net = SpatialNet::new_seeded(pc, local_dim, intent_dim, 0, 3, 4, 4, 4, 4242);
        let planes = fill(pc * h * w, 11);
        let cands: Vec<(Option<(usize, usize)>, Vec<f64>, Vec<f64>)> = vec![
            (Some((1, 0)), fill(local_dim, 21), fill(intent_dim, 31)),
            (None, fill(local_dim, 22), fill(intent_dim, 32)),
            (Some((3, 2)), fill(local_dim, 23), fill(intent_dim, 33)),
        ];
        let pi = vec![0.5, 0.3, 0.2];
        let z = 0.4;

        let (grad, ploss, vloss) = net.train_grad(&planes, h, w, &cands, &pi, z);
        assert!(ploss.is_finite() && vloss.is_finite() && ploss >= 0.0);

        macro_rules! check_param {
            ($field:expr, $gradvec:expr, $name:expr) => {{
                let n = $field.len();
                let stride = (n / 6).max(1);
                let mut j = 0;
                while j < n {
                    let save = $field[j];
                    $field[j] = save + EPS;
                    let lp = combined_loss(&net, &planes, h, w, &cands, &pi, z);
                    $field[j] = save - EPS;
                    let lm = combined_loss(&net, &planes, h, w, &cands, &pi, z);
                    $field[j] = save;
                    let num = (lp - lm) / (2.0 * EPS);
                    assert_close($gradvec[j], num, $name);
                    j += stride;
                }
            }};
        }
        // The policy head is the part whose input width changed (2D+18+12); FD it
        // plus the trunk (grad must route correctly through the wider concat).
        check_param!(net.policy_d1.weights, grad.policy_d1_w, "policy_d1_w@18");
        check_param!(net.policy_d1.bias, grad.policy_d1_b, "policy_d1_b@18");
        check_param!(net.policy_d2.weights, grad.policy_d2_w, "policy_d2_w@18");
        check_param!(net.conv1.weights, grad.conv1_w, "conv1_w@18");
        check_param!(net.conv2.weights, grad.conv2_w, "conv2_w@18");
        check_param!(net.value_d1.weights, grad.value_d1_w, "value_d1_w@18");
    }

    // ---- Test 2c: FD gradient check WITH value-head scalar features -------
    //
    // The value head now optionally takes per-state scalar economy features
    // concatenated onto the pooled global_embed (value_scalar_dim > 0). This
    // checks the combined-loss gradient at the exact width the trainer uses
    // (15 planes, local_dim 18, intent 12, value_scalar_dim 8) so a regression in
    // the value_d1 concat / backward split is caught. The scalars are non-param
    // INPUTS, so we FD only the parameters (incl. value_d1 whose input width grew).
    #[test]
    fn combined_grad_fd_value_scalars() {
        let (pc, h, w) = (3usize, 3usize, 4usize);
        let local_dim = 18usize;
        let intent_dim = 12usize;
        let vsd = 8usize; // value_scalar_dim used by the trainer
        let mut net = SpatialNet::new_seeded(pc, local_dim, intent_dim, vsd, 3, 5, 5, 5, 31337);
        let planes = fill(pc * h * w, 11);
        let vscalars = fill(vsd, 71);
        let cands: Vec<(Option<(usize, usize)>, Vec<f64>, Vec<f64>)> = vec![
            (Some((1, 0)), fill(local_dim, 21), fill(intent_dim, 31)),
            (None, fill(local_dim, 22), fill(intent_dim, 32)),
            (Some((3, 2)), fill(local_dim, 23), fill(intent_dim, 33)),
        ];
        let pi = vec![0.5, 0.3, 0.2];
        let z = -0.2;

        // Analytic grad from the cached training path (cache carries scalars).
        let cache = net.forward_board_scalars(&planes, h, w, &vscalars);
        let (grad, ploss, vloss) = net.train_grad_cached(&cache, &cands, &pi, z);
        assert!(ploss.is_finite() && vloss.is_finite() && ploss >= 0.0);

        // Scalar-aware combined loss for FD.
        let loss = |net: &SpatialNet| -> f64 {
            let cache = net.forward_board_scalars(&planes, h, w, &vscalars);
            let value = net.value_from(&cache);
            let value_loss = (value - z) * (value - z);
            let scores: Vec<f64> = cands
                .iter()
                .map(|(t, l, i)| net.score_candidate(&cache, *t, l, i))
                .collect();
            let p = softmax(&scores);
            let mut policy_loss = 0.0;
            for c in 0..cands.len() {
                if pi[c] > 0.0 {
                    policy_loss += -pi[c] * p[c].max(1e-12).ln();
                }
            }
            policy_loss + value_loss
        };

        macro_rules! check_param {
            ($field:expr, $gradvec:expr, $name:expr) => {{
                let n = $field.len();
                let stride = (n / 6).max(1);
                let mut j = 0;
                while j < n {
                    let save = $field[j];
                    $field[j] = save + EPS;
                    let lp = loss(&net);
                    $field[j] = save - EPS;
                    let lm = loss(&net);
                    $field[j] = save;
                    let num = (lp - lm) / (2.0 * EPS);
                    assert_close($gradvec[j], num, $name);
                    j += stride;
                }
            }};
        }
        // value_d1 input width is now D + vsd; check it plus the rest of the net.
        check_param!(net.value_d1.weights, grad.value_d1_w, "value_d1_w@vsd8");
        check_param!(net.value_d1.bias, grad.value_d1_b, "value_d1_b@vsd8");
        check_param!(net.value_d2.weights, grad.value_d2_w, "value_d2_w@vsd8");
        check_param!(net.conv1.weights, grad.conv1_w, "conv1_w@vsd8");
        check_param!(net.conv2.weights, grad.conv2_w, "conv2_w@vsd8");
        check_param!(net.policy_d1.weights, grad.policy_d1_w, "policy_d1_w@vsd8");
        check_param!(net.policy_d2.weights, grad.policy_d2_w, "policy_d2_w@vsd8");
    }

    // ---- Test 2d: VALUE-ONLY gradient (Lever C: scripted-opponent value) --
    //
    // `train_grad_value_only_scalars` must (a) produce ZERO policy gradient (the
    // policy head is untouched — scripted opponent has no usable MCTS pi), and (b)
    // match a finite-difference of the PURE value loss for the value head + trunk.
    #[test]
    fn value_only_grad_zero_policy_and_fd_value() {
        let (pc, h, w) = (3usize, 3usize, 4usize);
        let local_dim = 18usize;
        let intent_dim = 12usize;
        let vsd = 12usize;
        let mut net = SpatialNet::new_seeded(pc, local_dim, intent_dim, vsd, 3, 5, 5, 5, 0xA10E);
        let planes = fill(pc * h * w, 13);
        let vscalars = fill(vsd, 73);
        let z = 0.7;

        let (grad, ploss, vloss) = net.train_grad_value_only_scalars(&planes, h, w, &vscalars, z);
        // (a) NO policy loss and NO policy gradient.
        assert_eq!(ploss, 0.0, "value-only must have zero policy loss");
        for &g in &grad.policy_d1_w { assert_eq!(g, 0.0, "value-only policy_d1_w must be 0"); }
        for &g in &grad.policy_d2_w { assert_eq!(g, 0.0, "value-only policy_d2_w must be 0"); }
        for &g in &grad.policy_d1_b { assert_eq!(g, 0.0, "value-only policy_d1_b must be 0"); }
        for &g in &grad.policy_d2_b { assert_eq!(g, 0.0, "value-only policy_d2_b must be 0"); }
        assert!(vloss.is_finite() && vloss >= 0.0);

        // (b) FD of the PURE value loss.
        let vloss_fn = |net: &SpatialNet| -> f64 {
            let cache = net.forward_board_scalars(&planes, h, w, &vscalars);
            let value = net.value_from(&cache);
            (value - z) * (value - z)
        };
        macro_rules! check_param {
            ($field:expr, $gradvec:expr, $name:expr) => {{
                let n = $field.len();
                let stride = (n / 6).max(1);
                let mut j = 0;
                while j < n {
                    let save = $field[j];
                    $field[j] = save + EPS;
                    let lp = vloss_fn(&net);
                    $field[j] = save - EPS;
                    let lm = vloss_fn(&net);
                    $field[j] = save;
                    let num = (lp - lm) / (2.0 * EPS);
                    assert_close($gradvec[j], num, $name);
                    j += stride;
                }
            }};
        }
        check_param!(net.value_d1.weights, grad.value_d1_w, "vo_value_d1_w");
        check_param!(net.value_d2.weights, grad.value_d2_w, "vo_value_d2_w");
        check_param!(net.conv1.weights, grad.conv1_w, "vo_conv1_w");
        check_param!(net.conv2.weights, grad.conv2_w, "vo_conv2_w");
    }

    // ---- Test 2e: FD gradient check at the EXP-M width --------------------
    //
    // Exp M widens the net inputs to the CORRECTED eyes: planes 15→24 (split
    // owned/conquering soldiers per side, enemy/self frontier-reachability, att−def
    // diff, device-defenseless, river-block, broadcast enemy mobile budget) and
    // value_scalar_dim 8→12 (relative-army + soldier/worker headroom + enemy device
    // threat). The combined-loss gradient math is dim-agnostic, but this pins the
    // EXACT widths the Exp-M trainer constructs so a regression in the wider
    // value_d1 concat / backward split (vsd=12) or the trunk at 24 planes is caught.
    #[test]
    fn combined_grad_fd_expm_widths() {
        let (pc, h, w) = (24usize, 3usize, 4usize); // 24 planes (corrected eyes)
        let local_dim = 18usize;
        let intent_dim = 12usize;
        let vsd = 12usize; // Exp-M value_scalar_dim
        let mut net = SpatialNet::new_seeded(pc, local_dim, intent_dim, vsd, 4, 5, 5, 5, 0xEac);
        let planes = fill(pc * h * w, 11);
        let vscalars = fill(vsd, 71);
        let cands: Vec<(Option<(usize, usize)>, Vec<f64>, Vec<f64>)> = vec![
            (Some((1, 0)), fill(local_dim, 21), fill(intent_dim, 31)),
            (None, fill(local_dim, 22), fill(intent_dim, 32)),
            (Some((3, 2)), fill(local_dim, 23), fill(intent_dim, 33)),
        ];
        let pi = vec![0.5, 0.3, 0.2];
        let z = 0.15;

        let cache = net.forward_board_scalars(&planes, h, w, &vscalars);
        let (grad, ploss, vloss) = net.train_grad_cached(&cache, &cands, &pi, z);
        assert!(ploss.is_finite() && vloss.is_finite() && ploss >= 0.0);

        let loss = |net: &SpatialNet| -> f64 {
            let cache = net.forward_board_scalars(&planes, h, w, &vscalars);
            let value = net.value_from(&cache);
            let value_loss = (value - z) * (value - z);
            let scores: Vec<f64> = cands
                .iter()
                .map(|(t, l, i)| net.score_candidate(&cache, *t, l, i))
                .collect();
            let p = softmax(&scores);
            let mut policy_loss = 0.0;
            for c in 0..cands.len() {
                if pi[c] > 0.0 {
                    policy_loss += -pi[c] * p[c].max(1e-12).ln();
                }
            }
            policy_loss + value_loss
        };

        macro_rules! check_param {
            ($field:expr, $gradvec:expr, $name:expr) => {{
                let n = $field.len();
                let stride = (n / 6).max(1);
                let mut j = 0;
                while j < n {
                    let save = $field[j];
                    $field[j] = save + EPS;
                    let lp = loss(&net);
                    $field[j] = save - EPS;
                    let lm = loss(&net);
                    $field[j] = save;
                    let num = (lp - lm) / (2.0 * EPS);
                    assert_close($gradvec[j], num, $name);
                    j += stride;
                }
            }};
        }
        check_param!(net.value_d1.weights, grad.value_d1_w, "value_d1_w@expm");
        check_param!(net.value_d1.bias, grad.value_d1_b, "value_d1_b@expm");
        check_param!(net.value_d2.weights, grad.value_d2_w, "value_d2_w@expm");
        check_param!(net.conv1.weights, grad.conv1_w, "conv1_w@expm");
        check_param!(net.conv2.weights, grad.conv2_w, "conv2_w@expm");
        check_param!(net.policy_d1.weights, grad.policy_d1_w, "policy_d1_w@expm");
        check_param!(net.policy_d2.weights, grad.policy_d2_w, "policy_d2_w@expm");
    }

    // ---- Test 2f: FD gradient check WITH the RESIDUAL trunk block ---------
    //
    // ROUND-3 capacity bump adds an optional depth-preserving residual conv
    // (conv3: Conv2d(D->D,3,1), board_embed = tanh(conv3(trunk2)) + trunk2). The
    // identity skip + the extra conv path are new gradient routes through the
    // trunk; this FD-checks conv3's weights/bias AND that conv1/conv2 still get the
    // correct grad through BOTH the skip and the conv3 path. Uses small dims with
    // use_residual = true so the FD loop stays fast.
    #[test]
    fn combined_grad_fd_residual_block() {
        let (pc, h, w) = (3usize, 3usize, 4usize);
        let local_dim = 18usize;
        let intent_dim = 12usize;
        let vsd = 12usize;
        // use_residual = true, D=5 (conv3 is D->D = 5->5).
        let mut net =
            SpatialNet::new_seeded_arch(pc, local_dim, intent_dim, vsd, 4, 5, 5, 5, true, 0xDEAD);
        assert!(net.conv3.is_some(), "residual block must be present");
        let planes = fill(pc * h * w, 11);
        let vscalars = fill(vsd, 71);
        let cands: Vec<(Option<(usize, usize)>, Vec<f64>, Vec<f64>)> = vec![
            (Some((1, 0)), fill(local_dim, 21), fill(intent_dim, 31)),
            (None, fill(local_dim, 22), fill(intent_dim, 32)),
            (Some((3, 2)), fill(local_dim, 23), fill(intent_dim, 33)),
        ];
        let pi = vec![0.5, 0.3, 0.2];
        let z = 0.1;

        let cache = net.forward_board_scalars(&planes, h, w, &vscalars);
        let (grad, ploss, vloss) = net.train_grad_cached(&cache, &cands, &pi, z);
        assert!(ploss.is_finite() && vloss.is_finite() && ploss >= 0.0);
        assert_eq!(grad.conv3_w.len(), net.conv3.as_ref().unwrap().weights.len());

        let loss = |net: &SpatialNet| -> f64 {
            let cache = net.forward_board_scalars(&planes, h, w, &vscalars);
            let value = net.value_from(&cache);
            let value_loss = (value - z) * (value - z);
            let scores: Vec<f64> = cands
                .iter()
                .map(|(t, l, i)| net.score_candidate(&cache, *t, l, i))
                .collect();
            let p = softmax(&scores);
            let mut policy_loss = 0.0;
            for c in 0..cands.len() {
                if pi[c] > 0.0 {
                    policy_loss += -pi[c] * p[c].max(1e-12).ln();
                }
            }
            policy_loss + value_loss
        };

        macro_rules! check_param {
            ($field:expr, $gradvec:expr, $name:expr) => {{
                let n = $field.len();
                let stride = (n / 6).max(1);
                let mut j = 0;
                while j < n {
                    let save = $field[j];
                    $field[j] = save + EPS;
                    let lp = loss(&net);
                    $field[j] = save - EPS;
                    let lm = loss(&net);
                    $field[j] = save;
                    let num = (lp - lm) / (2.0 * EPS);
                    assert_close($gradvec[j], num, $name);
                    j += stride;
                }
            }};
        }
        // The residual conv: FD via short-lived mutable borrows so `loss(&net)` can
        // re-borrow immutably between perturbations (the in-place macro can't hold a
        // mutable borrow of net.conv3 across the loss calls).
        macro_rules! check_conv3 {
            ($sel:ident, $gradvec:expr, $name:expr) => {{
                let n = net.conv3.as_ref().unwrap().$sel.len();
                let stride = (n / 6).max(1);
                let mut j = 0;
                while j < n {
                    let save = net.conv3.as_ref().unwrap().$sel[j];
                    net.conv3.as_mut().unwrap().$sel[j] = save + EPS;
                    let lp = loss(&net);
                    net.conv3.as_mut().unwrap().$sel[j] = save - EPS;
                    let lm = loss(&net);
                    net.conv3.as_mut().unwrap().$sel[j] = save;
                    let num = (lp - lm) / (2.0 * EPS);
                    assert_close($gradvec[j], num, $name);
                    j += stride;
                }
            }};
        }
        check_conv3!(weights, grad.conv3_w, "conv3_w@res");
        check_conv3!(bias, grad.conv3_b, "conv3_b@res");
        check_param!(net.conv1.weights, grad.conv1_w, "conv1_w@res");
        check_param!(net.conv1.bias, grad.conv1_b, "conv1_b@res");
        check_param!(net.conv2.weights, grad.conv2_w, "conv2_w@res");
        check_param!(net.conv2.bias, grad.conv2_b, "conv2_b@res");
        check_param!(net.value_d1.weights, grad.value_d1_w, "value_d1_w@res");
        check_param!(net.value_d2.weights, grad.value_d2_w, "value_d2_w@res");
        check_param!(net.policy_d1.weights, grad.policy_d1_w, "policy_d1_w@res");
        check_param!(net.policy_d2.weights, grad.policy_d2_w, "policy_d2_w@res");
    }

    // ---- Test 2g: the DEPLOYED round-3 arch has the expected param count --
    #[test]
    fn round3_default_arch_param_count() {
        // Trainer I/O: 24 planes, local 18, intent 12, value_scalar_dim 12.
        let net = SpatialNet::default_with_value_scalars(24, 18, 12, 12, 7);
        assert!(net.conv3.is_some(), "round-3 default uses the residual trunk");
        assert_eq!(net.d1, 32);
        assert_eq!(net.d, 48);
        assert_eq!(net.hv, 64);
        assert_eq!(net.hp, 64);
        // conv1 32*24*9+32=6944, conv2 48*32*9+48=13872, conv3 48*48*9+48=20784,
        // value_d1 (48+12)*64+64=3904, value_d2 65, policy_d1 (2*48+18+12)*64+64=8128,
        // policy_d2 65  => 53762.
        assert_eq!(net.param_count(), 53762, "round-3 deployed param count");
    }

    // ---- Test 2h: the SMALL (pre-round-3) arch has the documented param count -
    #[test]
    fn small_default_arch_param_count() {
        // Trainer I/O: 24 planes, local 18, intent 12, value_scalar_dim 12.
        let net = SpatialNet::default_small_with_value_scalars(24, 18, 12, 12, 7);
        assert!(net.conv3.is_none(), "small arch has NO residual trunk");
        assert_eq!(net.d1, 16);
        assert_eq!(net.d, 24);
        assert_eq!(net.hv, 24);
        assert_eq!(net.hp, 24);
        // conv1 16*24*9+16=3472, conv2 24*16*9+24=3480, value_d1 (24+12)*24+24=888,
        // value_d2 25, policy_d1 (2*24+18+12)*24+24=1896, policy_d2 25 => 9786.
        assert_eq!(net.param_count(), 9786, "small (round-1/2) param count");
    }

    // ---- Test 2i: FD gradient check of the SMALL arch (no residual block) ----
    //
    // The `--net-size small` arch drops the residual conv3 (board_embed == trunk2)
    // and narrows the trunk. This FD-checks that conv1/conv2 + both heads still get
    // correct gradients with `use_residual = false` (the grad code's no-conv3 path).
    #[test]
    fn combined_grad_fd_small_arch() {
        let (pc, h, w) = (3usize, 3usize, 4usize);
        let local_dim = 18usize;
        let intent_dim = 12usize;
        let vsd = 12usize;
        // Small arch shape: use_residual = false (conv3 == None).
        let mut net =
            SpatialNet::new_seeded_arch(pc, local_dim, intent_dim, vsd, 4, 5, 5, 5, false, 0xBEEF);
        assert!(net.conv3.is_none(), "small arch must have NO residual block");
        let planes = fill(pc * h * w, 11);
        let vscalars = fill(vsd, 71);
        let cands: Vec<(Option<(usize, usize)>, Vec<f64>, Vec<f64>)> = vec![
            (Some((1, 0)), fill(local_dim, 21), fill(intent_dim, 31)),
            (None, fill(local_dim, 22), fill(intent_dim, 32)),
            (Some((3, 2)), fill(local_dim, 23), fill(intent_dim, 33)),
        ];
        let pi = vec![0.5, 0.3, 0.2];
        let z = 0.1;

        let cache = net.forward_board_scalars(&planes, h, w, &vscalars);
        let (grad, ploss, vloss) = net.train_grad_cached(&cache, &cands, &pi, z);
        assert!(ploss.is_finite() && vloss.is_finite() && ploss >= 0.0);
        assert!(grad.conv3_w.is_empty(), "no residual ⇒ empty conv3 grad");

        let loss = |net: &SpatialNet| -> f64 {
            let cache = net.forward_board_scalars(&planes, h, w, &vscalars);
            let value = net.value_from(&cache);
            let value_loss = (value - z) * (value - z);
            let scores: Vec<f64> = cands
                .iter()
                .map(|(t, l, i)| net.score_candidate(&cache, *t, l, i))
                .collect();
            let p = softmax(&scores);
            let mut policy_loss = 0.0;
            for c in 0..cands.len() {
                if pi[c] > 0.0 {
                    policy_loss += -pi[c] * p[c].max(1e-12).ln();
                }
            }
            policy_loss + value_loss
        };

        macro_rules! check_param {
            ($field:expr, $gradvec:expr, $name:expr) => {{
                let n = $field.len();
                let stride = (n / 6).max(1);
                let mut j = 0;
                while j < n {
                    let save = $field[j];
                    $field[j] = save + EPS;
                    let lp = loss(&net);
                    $field[j] = save - EPS;
                    let lm = loss(&net);
                    $field[j] = save;
                    let num = (lp - lm) / (2.0 * EPS);
                    assert_close($gradvec[j], num, $name);
                    j += stride;
                }
            }};
        }
        check_param!(net.conv1.weights, grad.conv1_w, "conv1_w@small");
        check_param!(net.conv1.bias, grad.conv1_b, "conv1_b@small");
        check_param!(net.conv2.weights, grad.conv2_w, "conv2_w@small");
        check_param!(net.conv2.bias, grad.conv2_b, "conv2_b@small");
        check_param!(net.value_d1.weights, grad.value_d1_w, "value_d1_w@small");
        check_param!(net.value_d2.weights, grad.value_d2_w, "value_d2_w@small");
        check_param!(net.policy_d1.weights, grad.policy_d1_w, "policy_d1_w@small");
        check_param!(net.policy_d2.weights, grad.policy_d2_w, "policy_d2_w@small");
    }

    // ---- Test 2d: value scalars actually CHANGE the value output ----------
    #[test]
    fn value_scalars_affect_value() {
        let (pc, h, w) = (4usize, 4usize, 5usize);
        let net = SpatialNet::default_with_value_scalars(pc, 6, 3, 4, 9001);
        let planes = fill(pc * h * w, 5);
        let a = net.value_from(&net.forward_board_scalars(&planes, h, w, &[0.0, 0.0, 0.0, 0.0]));
        let b = net.value_from(&net.forward_board_scalars(&planes, h, w, &[1.0, -1.0, 0.5, -0.5]));
        assert!(
            (a - b).abs() > 1e-9,
            "value scalars must move the value head: {a} vs {b}"
        );
    }

    // ---- Test 3: apply_grad actually moves params toward lower loss -------

    // ---- Micro-benchmark of the MCTS inference workload ------------------
    //
    // Ignored by default (timing, not correctness). Run with:
    //   cargo test -p cp-ai --release -- --ignored --nocapture bench_mcts_forward
    //
    // Mimics one MCTS node evaluation: forward_board once + score ~6 candidates
    // + value_from once, repeated N times. Exercises the cached-trunk + scratch
    // inference path (score_candidate_into) the trainer now uses.
    #[test]
    #[ignore]
    fn bench_mcts_forward() {
        use std::time::Instant;
        // Match the trainer's real workload: 24 planes, 14×12 board, local 18,
        // value_scalar_dim 12. Times one MCTS node eval (trunk once + 6 candidate
        // scores + value) for the ROUND-3 arch vs the OLD tiny 9786-param arch so the
        // throughput cost of the capacity bump is explicit.
        let (pc, h, w) = (24usize, 14usize, 12usize);
        let (local_dim, intent_dim, vsd) = (18usize, 12usize, 12usize);
        let planes = fill(pc * h * w, 123);
        let vscalars = fill(vsd, 41);
        let locals: Vec<Vec<f64>> = (0..6).map(|i| fill(local_dim, 200 + i)).collect();
        let intents: Vec<Vec<f64>> = (0..6).map(|i| fill(intent_dim, 300 + i)).collect();
        let targets: Vec<Option<(usize, usize)>> = vec![
            Some((3, 4)),
            Some((7, 2)),
            None,
            Some((11, 9)),
            Some((0, 0)),
            Some((6, 13)),
        ];
        let n = 20_000usize;

        let bench_one = |net: &SpatialNet| -> std::time::Duration {
            let mut scratch = PolicyScratch::new();
            let t0 = Instant::now();
            let mut acc = 0.0f64;
            for _ in 0..n {
                let cache = net.forward_board_scalars(&planes, h, w, &vscalars);
                for c in 0..6 {
                    acc += net.score_candidate_into(
                        &cache, targets[c], &locals[c], &intents[c], &mut scratch,
                    );
                }
                acc += net.value_from(&cache);
            }
            std::hint::black_box(acc);
            t0.elapsed()
        };

        // OLD tiny arch (the round-1/2 net): D1=16,D=24,HV=24,HP=24, no residual.
        let old =
            SpatialNet::new_seeded_arch(pc, local_dim, intent_dim, vsd, 16, 24, 24, 24, false, 7);
        // NEW round-3 arch (the deployed default).
        let new = SpatialNet::default_with_value_scalars(pc, local_dim, intent_dim, vsd, 7);

        let t_old = bench_one(&old);
        let t_new = bench_one(&new);
        eprintln!(
            "bench_mcts_forward N={n}: OLD ({} params) {t_old:?} ({:.0} ns/node)  NEW round-3 ({} params, D1={} D={} residual={}) {t_new:?} ({:.0} ns/node)  slowdown {:.2}x",
            old.param_count(),
            t_old.as_nanos() as f64 / n as f64,
            new.param_count(),
            new.d1, new.d, new.conv3.is_some(),
            t_new.as_nanos() as f64 / n as f64,
            t_new.as_secs_f64() / t_old.as_secs_f64()
        );
    }

    // ---- Equivalence golden test (performance-refactor guard) ------------
    //
    // Locks the EXACT forward (value + per-candidate scores) and backward
    // (every grad vector) of the DEPLOYED round-3 arch at the trainer's real
    // I/O (24 planes / local 18 / intent 12 / vsd 12) on a fixed-seed input.
    // Any speed refactor of the conv / dense / pool math MUST keep these
    // checksums bit-stable to within `EQ_TOL` (≈ f64 epsilon), proving the
    // optimized binary evaluates the LIVE run's checkpoint identically.
    //
    // The golden constants below were captured from the pre-optimization
    // implementation; the optimized math must reproduce them.
    const EQ_TOL: f64 = 1e-9;

    /// Order-stable scalar fingerprint of a slice: Σ vᵢ·(i+1)·φ mixing so a
    /// permutation or single-element change moves the sum well above EQ_TOL.
    fn fingerprint(v: &[f64]) -> f64 {
        let mut acc = 0.0f64;
        for (i, &x) in v.iter().enumerate() {
            acc += x * ((i as f64) * 0.6180339887498949 + 1.0);
        }
        acc
    }

    /// Build the deployed-arch net + a fixed-seed decision (planes, scalars,
    /// candidates, pi, z) used by the equivalence golden test.
    #[allow(clippy::type_complexity)]
    fn eq_fixture() -> (
        SpatialNet,
        Vec<f64>,
        usize,
        usize,
        Vec<f64>,
        Vec<(Option<(usize, usize)>, Vec<f64>, Vec<f64>)>,
        Vec<f64>,
        f64,
    ) {
        let (pc, h, w) = (24usize, 14usize, 12usize); // trainer board size
        let (local_dim, intent_dim, vsd) = (18usize, 12usize, 12usize);
        let net = SpatialNet::default_with_value_scalars(pc, local_dim, intent_dim, vsd, 0xC0FFEE);
        let planes = fill(pc * h * w, 0xBEEF);
        let vscalars = fill(vsd, 0xFEED);
        let cands: Vec<(Option<(usize, usize)>, Vec<f64>, Vec<f64>)> = vec![
            (Some((3, 4)), fill(local_dim, 1), fill(intent_dim, 11)),
            (None, fill(local_dim, 2), fill(intent_dim, 12)),
            (Some((11, 9)), fill(local_dim, 3), fill(intent_dim, 13)),
            (Some((0, 0)), fill(local_dim, 4), fill(intent_dim, 14)),
            (Some((6, 13)), fill(local_dim, 5), fill(intent_dim, 15)),
        ];
        let pi = vec![0.30, 0.10, 0.25, 0.20, 0.15];
        let z = -0.35;
        (net, planes, h, w, vscalars, cands, pi, z)
    }

    #[test]
    fn forward_backward_equivalence_golden() {
        let (net, planes, h, w, vscalars, cands, pi, z) = eq_fixture();

        // Forward: value + every candidate score.
        let cache = net.forward_board_scalars(&planes, h, w, &vscalars);
        let value = net.value_from(&cache);
        let scores: Vec<f64> = cands
            .iter()
            .map(|(t, l, i)| net.score_candidate(&cache, *t, l, i))
            .collect();
        // Also fingerprint the full board_embed + global_embed so a trunk
        // rounding change is caught even if it cancels in the heads.
        let fwd_fp = fingerprint(&cache.board_embed)
            + 7.0 * fingerprint(&cache.global_embed)
            + 13.0 * fingerprint(&scores)
            + 101.0 * value;

        // Backward: fingerprint every grad vector.
        let (grad, ploss, vloss) = net.train_grad_cached(&cache, &cands, &pi, z);
        let bwd_fp = fingerprint(&grad.conv1_w)
            + 1.1 * fingerprint(&grad.conv1_b)
            + 2.0 * fingerprint(&grad.conv2_w)
            + 2.1 * fingerprint(&grad.conv2_b)
            + 3.0 * fingerprint(&grad.conv3_w)
            + 3.1 * fingerprint(&grad.conv3_b)
            + 4.0 * fingerprint(&grad.value_d1_w)
            + 4.1 * fingerprint(&grad.value_d1_b)
            + 5.0 * fingerprint(&grad.value_d2_w)
            + 5.1 * fingerprint(&grad.value_d2_b)
            + 6.0 * fingerprint(&grad.policy_d1_w)
            + 6.1 * fingerprint(&grad.policy_d1_b)
            + 7.0 * fingerprint(&grad.policy_d2_w)
            + 7.1 * fingerprint(&grad.policy_d2_b)
            + 211.0 * (ploss + vloss);

        // Golden constants for the DEPLOYED `default_with_value_scalars` arch.
        // Re-emitted 2026-06-06 when the round-3 residual block (conv3) was made
        // DILATED (k3/dil2/pad2, RF 9x9); this self-consistency lock tracks the
        // deployed arch, so a deliberate arch change re-stamps it. (AZ-only; not a
        // parity golden.)
        const GOLD_FWD: f64 = -3.41774919043339760e4;
        const GOLD_BWD: f64 = -5.30645312791301239e3;
        if std::env::var("EMIT_GOLDEN").is_ok() {
            eprintln!("GOLD_FWD = {fwd_fp:.17e};");
            eprintln!("GOLD_BWD = {bwd_fp:.17e};");
            return;
        }
        assert!(
            (fwd_fp - GOLD_FWD).abs() < EQ_TOL,
            "forward fingerprint drifted: {fwd_fp:.17e} vs golden {GOLD_FWD:.17e} (diff {:.3e})",
            (fwd_fp - GOLD_FWD).abs()
        );
        assert!(
            (bwd_fp - GOLD_BWD).abs() < EQ_TOL,
            "backward fingerprint drifted: {bwd_fp:.17e} vs golden {GOLD_BWD:.17e} (diff {:.3e})",
            (bwd_fp - GOLD_BWD).abs()
        );
    }

    #[test]
    fn apply_grad_reduces_loss() {
        let (pc, h, w) = (3usize, 3usize, 4usize);
        let (local_dim, intent_dim) = (2usize, 2usize);
        let mut net = SpatialNet::new_seeded(pc, local_dim, intent_dim, 0, 4, 5, 5, 5, 99);
        let planes = fill(pc * h * w, 55);
        let cands: Vec<(Option<(usize, usize)>, Vec<f64>, Vec<f64>)> = vec![
            (Some((0, 0)), fill(local_dim, 1), fill(intent_dim, 2)),
            (None, fill(local_dim, 3), fill(intent_dim, 4)),
            (Some((2, 1)), fill(local_dim, 5), fill(intent_dim, 6)),
        ];
        let pi = vec![0.6, 0.1, 0.3];
        let z = -0.3;

        let l0 = combined_loss(&net, &planes, h, w, &cands, &pi, z);
        for _ in 0..50 {
            let (g, _, _) = net.train_grad(&planes, h, w, &cands, &pi, z);
            net.apply_grad(&g, 0.05, 0.0);
        }
        let l1 = combined_loss(&net, &planes, h, w, &cands, &pi, z);
        assert!(l1 < l0, "loss should decrease: {l0} -> {l1}");
    }

    // ---- PPO clipped-surrogate gradient finite-difference check -------------
    //
    // PPO-SPEC §3d (MANDATORY). FD-checks `train_grad_ppo_cached` against the
    // analytic gradient for the COMBINED `L_clip + L_ent + val_coef·MSE`, covering
    // both advantage signs AND r inside / outside the clip band — and asserting the
    // clipped branch produces ZERO policy gradient (the conv trunk + policy heads go
    // to exactly 0 there while the value head still trains).

    /// Recompute the PPO scalar loss `L_clip + L_ent + val_coef·(V − vtarg)^2` for
    /// the current net params (FD probe). Mirrors `train_grad_ppo_cached` exactly,
    /// including the ±20 log-ratio clamp and the value-clip (vclip) branch.
    #[allow(clippy::too_many_arguments)]
    fn ppo_loss(
        net: &SpatialNet,
        planes: &[f64],
        h: usize,
        w: usize,
        vs: &[f64],
        cands: &[(Option<(usize, usize)>, Vec<f64>, Vec<f64>)],
        chosen: usize,
        logp_old: f64,
        adv: f64,
        vtarg: f64,
        v_old: f64,
        clip_eps: f64,
        ent_coef: f64,
        val_coef: f64,
        vclip: f64,
    ) -> f64 {
        let cache = net.forward_board_scalars(planes, h, w, vs);
        // value
        let value = net.value_from(&cache);
        let value_loss = if vclip > 0.0 {
            let v_clipped = value.clamp(v_old - vclip, v_old + vclip);
            let eu = value - vtarg;
            let ec = v_clipped - vtarg;
            (eu * eu).max(ec * ec)
        } else {
            (value - vtarg) * (value - vtarg)
        };
        // policy
        let scores: Vec<f64> = cands
            .iter()
            .map(|(t, l, i)| net.score_candidate(&cache, *t, l, i))
            .collect();
        let p = softmax(&scores);
        let logp_new = p[chosen].max(1e-12).ln();
        let log_ratio = (logp_new - logp_old).clamp(-20.0, 20.0);
        let r = log_ratio.exp();
        let lo = 1.0 - clip_eps;
        let hi = 1.0 + clip_eps;
        let l_clip = -(r * adv).min(r.clamp(lo, hi) * adv);
        let mut entropy = 0.0;
        for &pc in &p {
            if pc > 0.0 {
                entropy -= pc * pc.ln();
            }
        }
        let l_ent = -ent_coef * entropy;
        l_clip + l_ent + val_coef * value_loss
    }

    /// Run one FD vs analytic comparison of the PPO gradient for a given (adv, the
    /// frozen logp_old chosen to land r inside or outside the clip band). When
    /// `expect_clipped` is set, additionally assert the POLICY-side grads are ~0.
    fn ppo_grad_fd_case(adv: f64, logp_old_offset: f64, expect_clipped: bool) {
        let (pc, h, w) = (3usize, 3usize, 4usize);
        let (local_dim, intent_dim, vs_dim) = (2usize, 2usize, 3usize);
        let mut net = SpatialNet::new_seeded_arch(
            pc, local_dim, intent_dim, vs_dim, 3, 4, 4, 4, false, 1357,
        );
        let planes = fill(pc * h * w, 11);
        let vs = fill(vs_dim, 71);
        let cands: Vec<(Option<(usize, usize)>, Vec<f64>, Vec<f64>)> = vec![
            (Some((1, 0)), fill(local_dim, 21), fill(intent_dim, 31)),
            (None, fill(local_dim, 22), fill(intent_dim, 32)),
            (Some((3, 2)), fill(local_dim, 23), fill(intent_dim, 33)),
        ];
        let chosen = 0usize;
        let (clip_eps, ent_coef, val_coef, vclip) = (0.2, 0.01, 0.5, 0.0);
        let vtarg = 0.3;
        let v_old = 0.1;

        // Pick logp_old = logp_new(θ) − offset so r = exp(offset). With offset 0 the
        // ratio is 1 (inside band); a large +/- offset pushes r outside the band.
        let cache0 = net.forward_board_scalars(&planes, h, w, &vs);
        let scores0: Vec<f64> = cands
            .iter()
            .map(|(t, l, i)| net.score_candidate(&cache0, *t, l, i))
            .collect();
        let p0 = softmax(&scores0);
        let logp_new0 = p0[chosen].max(1e-12).ln();
        let logp_old = logp_new0 - logp_old_offset;
        let r0 = logp_old_offset.exp();

        // Sanity: the case is configured as intended (inside vs outside band).
        let lo = 1.0 - clip_eps;
        let hi = 1.0 + clip_eps;
        let clip_active = (adv >= 0.0 && r0 > hi) || (adv < 0.0 && r0 < lo);
        assert_eq!(
            clip_active, expect_clipped,
            "test setup: adv={adv} r0={r0} expected clip_active={expect_clipped}"
        );

        let cache = net.forward_board_scalars(&planes, h, w, &vs);
        let (grad, _pl, _vl) = net.train_grad_ppo_cached(
            &cache, &cands, chosen, logp_old, adv, vtarg, v_old, clip_eps, ent_coef, val_coef, vclip,
        );

        // The policy heads + conv trunk receive policy gradient. When clipped, the
        // policy-CLIP upstream is 0; the ONLY policy-side signal is the entropy bonus
        // (tiny). To assert "ZERO policy gradient from the clip" we run the SAME case
        // with ent_coef=0 and check the policy heads vanish in the clipped branch.
        if expect_clipped {
            let (grad_noent, _, _) = net.train_grad_ppo_cached(
                &cache, &cands, chosen, logp_old, adv, vtarg, v_old, clip_eps, 0.0, val_coef, vclip,
            );
            // With no entropy + clipped clip term, the policy-d1/d2 grads must be 0
            // (the value head + its trunk contribution remain non-zero, so we only
            // check the policy-only Dense layers, which the value path never touches).
            for (j, &gv) in grad_noent.policy_d2_w.iter().enumerate() {
                assert!(gv.abs() < 1e-12, "clipped policy_d2_w[{j}] should be 0, got {gv}");
            }
            for (j, &gv) in grad_noent.policy_d1_w.iter().enumerate() {
                assert!(gv.abs() < 1e-12, "clipped policy_d1_w[{j}] should be 0, got {gv}");
            }
        }

        // FD-check every parameter slice against the analytic grad of the COMBINED loss.
        macro_rules! check_param {
            ($field:expr, $gradvec:expr, $name:expr) => {{
                let n = $field.len();
                let stride = (n / 6).max(1);
                let mut j = 0;
                while j < n {
                    let save = $field[j];
                    $field[j] = save + EPS;
                    let lp = ppo_loss(
                        &net, &planes, h, w, &vs, &cands, chosen, logp_old, adv, vtarg, v_old,
                        clip_eps, ent_coef, val_coef, vclip,
                    );
                    $field[j] = save - EPS;
                    let lm = ppo_loss(
                        &net, &planes, h, w, &vs, &cands, chosen, logp_old, adv, vtarg, v_old,
                        clip_eps, ent_coef, val_coef, vclip,
                    );
                    $field[j] = save;
                    let num = (lp - lm) / (2.0 * EPS);
                    assert_close($gradvec[j], num, $name);
                    j += stride;
                }
            }};
        }
        check_param!(net.conv1.weights, grad.conv1_w, "ppo_conv1_w");
        check_param!(net.conv1.bias, grad.conv1_b, "ppo_conv1_b");
        check_param!(net.conv2.weights, grad.conv2_w, "ppo_conv2_w");
        check_param!(net.conv2.bias, grad.conv2_b, "ppo_conv2_b");
        check_param!(net.value_d1.weights, grad.value_d1_w, "ppo_value_d1_w");
        check_param!(net.value_d1.bias, grad.value_d1_b, "ppo_value_d1_b");
        check_param!(net.value_d2.weights, grad.value_d2_w, "ppo_value_d2_w");
        check_param!(net.value_d2.bias, grad.value_d2_b, "ppo_value_d2_b");
        check_param!(net.policy_d1.weights, grad.policy_d1_w, "ppo_policy_d1_w");
        check_param!(net.policy_d1.bias, grad.policy_d1_b, "ppo_policy_d1_b");
        check_param!(net.policy_d2.weights, grad.policy_d2_w, "ppo_policy_d2_w");
        check_param!(net.policy_d2.bias, grad.policy_d2_b, "ppo_policy_d2_b");
    }

    #[test]
    fn ppo_grad_finite_difference() {
        // adv > 0, r inside band (offset 0 → r=1): NOT clipped, full gradient.
        ppo_grad_fd_case(0.8, 0.0, false);
        // adv > 0, r OUTSIDE band high (offset +1.0 → r≈2.72 > 1.2): CLIPPED, 0 policy grad.
        ppo_grad_fd_case(0.8, 1.0, true);
        // adv < 0, r inside band: NOT clipped, full gradient.
        ppo_grad_fd_case(-0.8, 0.0, false);
        // adv < 0, r OUTSIDE band low (offset −1.0 → r≈0.37 < 0.8): CLIPPED, 0 policy grad.
        ppo_grad_fd_case(-0.8, -1.0, true);
        // adv > 0, r outside band LOW (offset −1.0 → r<0.8): NOT clipped (clip only
        // bites adv≥0 on the HIGH side) — full gradient. Confirms the asymmetry.
        ppo_grad_fd_case(0.8, -1.0, false);
        // adv < 0, r outside band HIGH (offset +1.0 → r>1.2): NOT clipped (clip only
        // bites adv<0 on the LOW side) — full gradient.
        ppo_grad_fd_case(-0.8, 1.0, false);
    }
}
