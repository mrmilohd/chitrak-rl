"""RewardManager that reshapes the total reward WTW's published
`only_positive_rewards_ji22_style` way, instead of mjlab's default plain
unclipped weighted sum.

mjlab's stock RewardManager.compute() (managers/reward_manager.py) just
accumulates every term's value*weight*dt into one running total -- no
clipping, no reshaping. WTW's compute_reward() does this instead:

    rew_buf_pos[:] = 0   # accumulates terms classified "positive" this step
    rew_buf_neg[:] = 0   # accumulates terms classified "negative" this step
    for each term:
        rew = func() * weight
        rew_buf += rew
        if sum(rew) >= 0: rew_buf_pos += rew
        else:             rew_buf_neg += rew
    if only_positive_rewards_ji22_style:
        total = rew_buf_pos * exp(rew_buf_neg / sigma_rew_neg)

The real published WTW config (scripts/train.py) uses
only_positive_rewards=False, only_positive_rewards_ji22_style=True,
sigma_rew_neg=0.02 -- the ji22 squash, not the hard zero-clip.

Why this matters: an early, barely-functional policy racking up deeply
negative total reward (we saw -12.05 in the very first smoke-test iteration)
has an incentive to end its episode as fast as possible just to stop
accruing penalty, if reward is allowed to go arbitrarily negative. The ji22
squash multiplies the positive-term total by an exponentially-decaying
factor of the (always <=0) negative-term total instead -- heavy penalties
shrink the reward toward zero smoothly, without a hard floor and without
removing the gradient entirely the way a hard clip-at-zero would.

One subtlety worth being precise about: classification into the positive vs
negative bucket is by the SIGN OF THE COMPUTED VALUE (value*weight, summed
across envs) THIS STEP -- not by the term's configured weight sign. This
matters for wtw_jump / wtw_tracking_contacts_shaped_force /
wtw_tracking_contacts_shaped_vel (env_cfg.py): each of those has a
*positive* published weight but a function that already returns a negative
quantity internally (matching WTW's own corl_rewards.py), so their net
per-step contribution is negative -- they must land in the negative bucket,
not the positive one, despite the positive weight.

BUG FOUND VIA A REAL KAGGLE RUN, FIXED HERE: this class previously inherited
mjlab's default dt-scaling (`value * weight * dt`, dt ~= 0.02), but WTW's own
compute_reward() has NO dt multiplication anywhere -- `rew = func() * weight`,
full stop. sigma_rew_neg=0.02 was calibrated by WTW's authors against THAT
unscaled magnitude. Multiplying every term by an extra ~0.02 before the squash
made rew_buf_neg ~50x smaller than WTW ever intended it to be relative to
sigma_rew_neg, which (counterintuitively) does NOT make the squash gentler --
exp(rew_buf_neg / sigma_rew_neg) is still wildly negative for very ordinary
per-step penalty totals, crushing the entire reward signal to ~0 essentially
every step from the start of training. Confirmed in an actual ~990-iteration
Kaggle run: Mean reward stuck at 0.00, Mean value loss at exactly 0.0000, and
action std *growing* (1.0 -> 3.68) instead of shrinking -- a policy that never
received a usable gradient, with the entropy bonus pushing it toward pure
noise unopposed. Fix: this class now ignores mjlab's dt-scaling entirely and
always uses the raw `value * weight` magnitude, matching WTW's compute_reward()
exactly -- not just in the squash, but in the episode-sum/step-reward logging
too, so every number this class produces is now WTW-scale, not mjlab-scale.

SECOND ISSUE FOUND, AFTER THE DT FIX, VIA A SECOND REAL KAGGLE RUN: fixing the
dt-scaling alone did NOT fix training -- Mean value loss was still exactly
0.0000 and Mean action std still exploding (1.0 -> 3.72). With dt-scaling
correctly removed, the real (now properly WTW-scaled) per-step rew_buf_neg
turned out to be approximately -13 (dominated by wtw_action_smoothness_1/_2,
which together were ~85% of the entire negative bucket -- consistent with an
exploded, near-random action distribution producing huge action-to-action
differences). exp(-13 / 0.02) = exp(-665) -- still numerically zero, every
step, regardless of dt. sigma_rew_neg=0.02 was tuned by WTW's authors against
THEIR pipeline's typical magnitudes (their own actuator-net torque model,
their own action scale), which this mjlab port's analytical-PD actuator model
and GO1_ACTION_SCALE evidently don't reproduce -- same configured weight,
different raw magnitude, because the rest of the simulation pipeline differs.

`apply_squash` below makes this togglable via the MJLAB_DISABLE_SQUASH env
var, specifically to A/B test whether sigma_rew_neg=0.02 itself is the active
bottleneck, independent of anything else: with it set, compute() returns the
plain (dt-scaling-fixed, WTW-magnitude) sum instead of the ji22 squash. If
training is learnable with the squash off, the fix is recalibrating
sigma_rew_neg for this pipeline's actual magnitudes, not WTW's literal 0.02.
If it's still broken with the squash off too, the action-distribution
explosion itself is the deeper problem, independent of reward shaping.
"""

from __future__ import annotations

import os

import torch

from mjlab.managers.reward_manager import RewardManager


class Ji22RewardManager(RewardManager):
  def __init__(
    self,
    cfg,
    env,
    *,
    scale_by_dt: bool = True,
    sigma_rew_neg: float = 0.02,
    apply_squash: bool | None = None,
  ) -> None:
    # scale_by_dt is accepted (ManagerBasedRlEnv constructs this class with
    # that keyword explicitly -- a drop-in RewardManager replacement has to
    # accept it) but deliberately IGNORED: compute() below never applies dt
    # scaling regardless of this value. See module docstring.
    del scale_by_dt
    super().__init__(cfg, env, scale_by_dt=False)
    self.sigma_rew_neg = sigma_rew_neg
    # MJLAB_DISABLE_SQUASH=1 -> plain sum (squash off), for the A/B test
    # described in the module docstring. Explicit apply_squash kwarg always
    # wins over the env var, for direct/programmatic use.
    if apply_squash is None:
      apply_squash = os.environ.get("MJLAB_DISABLE_SQUASH", "") != "1"
    self.apply_squash = apply_squash
    self._reward_buf_pos = torch.zeros_like(self._reward_buf)
    self._reward_buf_neg = torch.zeros_like(self._reward_buf)

  def compute(self, dt: float) -> torch.Tensor:
    del dt  # Deliberately unused -- WTW's own reward scaling has no dt factor.
    self._reward_buf_pos[:] = 0.0
    self._reward_buf_neg[:] = 0.0

    for term_idx, (name, term_cfg) in enumerate(
      zip(self._term_names, self._term_cfgs, strict=False)
    ):
      if term_cfg.weight == 0.0:
        self._step_reward[:, term_idx] = 0.0
        continue

      value = term_cfg.func(self._env, **term_cfg.params)
      self._check_term_shape(name, value)
      value = value * term_cfg.weight  # No dt multiplication -- see docstring.
      value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)

      # Classify by the computed value's sign, not the configured weight's
      # sign -- see module docstring.
      if torch.sum(value) >= 0:
        self._reward_buf_pos += value
      else:
        self._reward_buf_neg += value

      self._episode_sums[name] += value
      self._step_reward[:, term_idx] = value

    if self.apply_squash:
      self._reward_buf = self._reward_buf_pos * torch.exp(self._reward_buf_neg / self.sigma_rew_neg)
    else:
      self._reward_buf = self._reward_buf_pos + self._reward_buf_neg
    return self._reward_buf
