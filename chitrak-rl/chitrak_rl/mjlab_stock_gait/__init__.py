"""Importing this package registers the Mjlab-Velocity-Flat-Go1-WTW-Rewards task.

mjlab's `train`/`play` CLIs only auto-discover tasks registered inside the
installed mjlab.tasks package. This task lives outside that package, so it
won't show up in `train <task-id>` automatically -- import this package (or
run launch_train.py) before calling mjlab's training/play entrypoints.
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.config.go1.rl_cfg import unitree_go1_ppo_runner_cfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfg import mjlab_stock_gait_env_cfg

TASK_ID = "Mjlab-Velocity-Flat-Go1-WTW-Rewards"

register_mjlab_task(
  task_id=TASK_ID,
  env_cfg=mjlab_stock_gait_env_cfg(),
  play_env_cfg=mjlab_stock_gait_env_cfg(play=True),
  rl_cfg=unitree_go1_ppo_runner_cfg(),  # reused as-is from the stock task -- no RMA, no gait yet
  runner_cls=VelocityOnPolicyRunner,
)
