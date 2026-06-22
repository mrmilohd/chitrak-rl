"""Runnable play/inference entrypoint for mjlab_stock_gait.

Same reason as launch_train.py: this task lives outside mjlab's installed
mjlab.tasks package, so mjlab's own `play` CLI won't find it -- importing
`mjlab_stock_gait` here runs its __init__.py (registers the task) before we
call mjlab.scripts.play.run_play() directly.

No Ji22RewardManager patch needed here (unlike launch_train.py) -- play only
runs the saved actor's forward pass for inference; reward computation
doesn't influence which actions get taken, so it doesn't matter that this
uses mjlab's stock unclipped RewardManager instead.

Usage:
    python -m mjlab_stock_gait.play_local
    python -m mjlab_stock_gait.play_local --checkpoint-file path/to/model_N.pt
    python -m mjlab_stock_gait.play_local --num-envs 1
"""

import sys
from pathlib import Path

import tyro

import mjlab_stock_gait  # noqa: F401  (side effect: registers the task)
from mjlab.scripts.play import PlayConfig, run_play

TASK_ID = mjlab_stock_gait.TASK_ID
DEFAULT_CHECKPOINT = str(Path(__file__).parent / "model_650.pt")


def main() -> None:
  args = tyro.cli(
    PlayConfig,
    args=sys.argv[1:],
    default=PlayConfig(
      checkpoint_file=DEFAULT_CHECKPOINT,
      viewer="viser",
      num_envs=1,
    ),
    prog=f"{sys.argv[0]} {TASK_ID}",
  )
  run_play(TASK_ID, args)


if __name__ == "__main__":
  main()
