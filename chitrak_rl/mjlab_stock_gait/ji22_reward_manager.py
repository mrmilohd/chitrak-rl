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
"""

from __future__ import annotations

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
  ) -> None:
    super().__init__(cfg, env, scale_by_dt=scale_by_dt)
    self.sigma_rew_neg = sigma_rew_neg
    self._reward_buf_pos = torch.zeros_like(self._reward_buf)
    self._reward_buf_neg = torch.zeros_like(self._reward_buf)

  def compute(self, dt: float) -> torch.Tensor:
    self._reward_buf_pos[:] = 0.0
    self._reward_buf_neg[:] = 0.0
    scale = dt if self._scale_by_dt else 1.0

    for term_idx, (name, term_cfg) in enumerate(
      zip(self._term_names, self._term_cfgs, strict=False)
    ):
      if term_cfg.weight == 0.0:
        self._step_reward[:, term_idx] = 0.0
        continue

      value = term_cfg.func(self._env, **term_cfg.params)
      self._check_term_shape(name, value)
      value = value * term_cfg.weight * scale
      value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)

      # Classify by the computed value's sign, not the configured weight's
      # sign -- see module docstring.
      if torch.sum(value) >= 0:
        self._reward_buf_pos += value
      else:
        self._reward_buf_neg += value

      self._episode_sums[name] += value
      self._step_reward[:, term_idx] = value / scale

    self._reward_buf = self._reward_buf_pos * torch.exp(self._reward_buf_neg / self.sigma_rew_neg)
    return self._reward_buf
