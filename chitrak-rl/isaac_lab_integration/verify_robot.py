"""
Verification script — spawns Chitrak in Isaac Lab and prints joint/actuator info.

Run inside the container:
    export PYTHONPATH=/workspace/isaaclab/source/isaaclab:/workspace/isaaclab/source/chitrak_integration:$PYTHONPATH
    ./isaaclab.sh -p /workspace/isaaclab/source/chitrak_integration/verify_robot.py
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify Chitrak robot loads correctly")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args(["--headless"])
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from chitrak_asset import CHITRAK_CFG

def main():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.005))
    sim.set_camera_view(eye=[2.0, 2.0, 1.5], target=[0.0, 0.0, 0.3])

    # Ground plane
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())

    # Spawn robot
    robot_cfg = CHITRAK_CFG.replace(prim_path="/World/Chitrak")
    robot = Articulation(robot_cfg)

    sim.reset()

    # ── Joint info ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("JOINT NAMES ({} total):".format(robot.num_joints))
    print("="*60)
    for i, name in enumerate(robot.joint_names):
        print(f"  [{i:2d}] {name}")

    # ── Body info ────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("BODY NAMES ({} total):".format(robot.num_bodies))
    print("="*60)
    for i, name in enumerate(robot.body_names):
        print(f"  [{i:2d}] {name}")

    # ── Actuator info ────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("ACTUATORS:")
    print("="*60)
    for name, actuator in robot.actuators.items():
        print(f"\n  Group: '{name}'")
        print(f"    Type:           {type(actuator).__name__}")
        print(f"    Joints:         {actuator.joint_names}")
        print(f"    Effort limit:   {actuator.effort_limit}")
        print(f"    Velocity limit: {actuator.velocity_limit}")
        print(f"    Stiffness:      {actuator.stiffness}")
        print(f"    Damping:        {actuator.damping}")

    # ── Initial state ────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("INITIAL JOINT POSITIONS:")
    print("="*60)
    joint_pos = robot.data.joint_pos[0]
    for i, name in enumerate(robot.joint_names):
        print(f"  {name:30s}: {joint_pos[i].item():.4f} rad")

    print("\n" + "="*60)
    print("ROOT STATE (pos, quat):")
    print("="*60)
    root = robot.data.root_state_w[0]
    print(f"  Position : {root[:3].tolist()}")
    print(f"  Quaternion: {root[3:7].tolist()}")
    print("="*60)
    print("\n[OK] Chitrak loaded successfully!\n")

    simulation_app.close()


if __name__ == "__main__":
    main()
