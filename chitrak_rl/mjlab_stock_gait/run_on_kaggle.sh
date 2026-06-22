#!/usr/bin/env bash
# Kaggle run script for mjlab_stock_gait.
#
# Assumes both the repo is already cloned AND mjlab is already installed in
# this Kaggle kernel -- no pip install step here. Run as a single notebook
# cell:
#
#   !bash chitrak/chitrak_rl/mjlab_stock_gait/run_on_kaggle.sh
#
# (adjust the path to wherever `git clone` put it). Deliberately self
# contained in one script rather than split across notebook cells -- Kaggle/
# Jupyter `!cd ...` does NOT persist to the next cell (only the `%cd` magic
# does), so anything split across cells silently breaks on the second cell.
# One `!bash` invocation sidesteps that entirely.
#
# Notebook setup: New Notebook -> Settings (right sidebar) -> Accelerator ->
# GPU T4 x2, before running this.
#
# Logs to wandb (mjlab's actual default logger -- RslRlBaseRunnerCfg.logger
# is "wandb"; the earlier local CPU smoke-tests on this repo overrode it to
# tensorboard just to avoid needing a wandb login on an offline dev machine).
# Assumes this Kaggle kernel is already wandb-authenticated.
#
# Any extra arguments you pass to this script are forwarded straight to
# launch_train.py, e.g.:
#   !bash .../run_on_kaggle.sh --agent.max-iterations 100   # quick test first
#   !bash .../run_on_kaggle.sh --agent.max-iterations 5000  # real run

set -euo pipefail

# --- 1. cd to the directory mjlab_stock_gait is importable from. ---
# This script lives at <repo>/chitrak_rl/mjlab_stock_gait/run_on_kaggle.sh,
# so its own directory's parent is chitrak_rl/ -- works regardless of what
# you named the cloned folder or what Jupyter's cwd happens to be.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- 2. Train, using both T4s. ---
# --gpu-ids all makes launch_training() detect both GPUs and route through
# torchrunx's multi-process launcher automatically. This is exactly the
# scenario the Ji22RewardManager patch in launch_train.py has to survive --
# torchrunx reconstructs run_train via cloudpickle in each worker process, so
# the patch is applied INSIDE run_train's own body, not as a plain
# module-level assignment (see launch_train.py's docstring) -- already
# handled, nothing extra needed here for that to work correctly.
python -m mjlab_stock_gait.launch_train \
  --gpu-ids all \
  --env.scene.num-envs 2048 \
  --agent.max-iterations 1500 \
  "$@"
