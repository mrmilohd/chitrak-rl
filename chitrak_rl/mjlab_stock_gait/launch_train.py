"""Runnable training entrypoint for mjlab_stock_gait.

Bypasses mjlab's `train` CLI on purpose -- that CLI only finds tasks
registered inside the installed mjlab.tasks package, and this task lives
outside it (see __init__.py). Importing `mjlab_stock_gait` here runs its
__init__.py, which calls register_mjlab_task() before we ever touch
mjlab.scripts.train.

Also patches in Ji22RewardManager (ji22_reward_manager.py) in place of
mjlab's default RewardManager, replicating WTW's actual published reward
summation (rew_buf_pos * exp(rew_buf_neg / sigma), not a plain unclipped sum)
-- see ji22_reward_manager.py for why.

IMPORTANT: with --gpu-ids requesting >1 GPU, launch_training() routes through
torchrunx, which spawns separate worker processes and reconstructs
mjlab.scripts.train.run_train via cloudpickle. For a normal importable
function, cloudpickle resolves it BY REFERENCE (re-imports mjlab.scripts.train
fresh in the worker) rather than replaying this script -- so a plain
module-level patch here would never reach the workers that actually train.
Fix: patch run_train itself to a wrapper that applies the RewardManager patch
INSIDE its own function body, so the patch runs wherever the function is
actually called, regardless of how that process reconstructed it. (Moot on
this machine specifically -- no CUDA at all, always single-process -- but
this is the documented gotcha for any future GPU run.)

Usage:
    python -m mjlab_stock_gait.launch_train
    python -m mjlab_stock_gait.launch_train --agent.max-iterations 1000
    python -m mjlab_stock_gait.launch_train --env.scene.num-envs 2048
"""

import sys

import tyro

import mjlab_stock_gait  # noqa: F401  (side effect: registers the task)
import mjlab.envs.manager_based_rl_env as _mbre
import mjlab.scripts.train as _mjlab_train
from mjlab.scripts.train import TrainConfig, launch_training

from .ji22_reward_manager import Ji22RewardManager

TASK_ID = mjlab_stock_gait.TASK_ID

_original_run_train = _mjlab_train.run_train


def _patched_run_train(task_id, cfg, log_dir):
  _mbre.RewardManager = Ji22RewardManager
  return _original_run_train(task_id, cfg, log_dir)


_mjlab_train.run_train = _patched_run_train


def main() -> None:
  args = tyro.cli(
    TrainConfig,
    args=sys.argv[1:],
    default=TrainConfig.from_task(TASK_ID),
    prog=f"{sys.argv[0]} {TASK_ID}",
  )
  launch_training(task_id=TASK_ID, args=args)


if __name__ == "__main__":
  main()
