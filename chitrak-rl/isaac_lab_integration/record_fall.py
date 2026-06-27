"""
Records Chitrak spawning and falling to the ground as an MP4.

Run inside the container:
    export PYTHONPATH=/workspace/isaaclab/source/isaaclab:/workspace/isaaclab/source/chitrak_integration:$PYTHONPATH
    ./isaaclab.sh -p /workspace/isaaclab/source/chitrak_integration/record_fall.py
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record Chitrak falling")
parser.add_argument("--duration", type=float, default=3.0, help="Simulation duration in seconds")
parser.add_argument("--fps", type=int, default=30, help="Output video FPS")
parser.add_argument("--output", type=str, default="/workspace/isaaclab/source/chitrak_integration/chitrak_fall.mp4")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args(["--headless", "--enable_cameras"])
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import SimulationContext

from chitrak_asset import CHITRAK_CFG

OUTPUT_PATH = "/workspace/isaaclab/source/chitrak_integration/chitrak_fall.mp4"
DURATION_S = 3.0
FPS = 30
SIM_DT = 0.005


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=SIM_DT))
    sim.set_camera_view(eye=[0.8, 0.8, 0.5], target=[0.0, 0.0, 0.1])

    # Ground plane
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())

    # Lighting
    sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8)).func(
        "/World/light", sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    )

    # Robot
    robot_cfg = CHITRAK_CFG.replace(prim_path="/World/Chitrak")
    robot = Articulation(robot_cfg)

    # Camera
    camera_cfg = CameraCfg(
        prim_path="/World/Camera",
        update_period=1.0 / FPS,
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=0.8,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 100.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.8, 0.8, 0.5),
            rot=(0.7071, 0.0, 0.0, -0.7071),
            convention="world",
        ),
    )
    camera = Camera(camera_cfg)

    sim.reset()
    print(f"[INFO] Simulating {DURATION_S}s and recording to {OUTPUT_PATH}")

    frames = []
    total_steps = int(DURATION_S / SIM_DT)
    capture_every = max(1, int(1.0 / (FPS * SIM_DT)))

    for step in range(total_steps):
        sim.step()
        robot.update(SIM_DT)
        camera.update(SIM_DT)

        if step % capture_every == 0:
            rgb = camera.data.output["rgb"]
            if rgb is not None:
                frame = rgb[0, ..., :3].cpu().numpy().astype(np.uint8)
                frames.append(frame)
                if step % (capture_every * 10) == 0:
                    print(f"  [{step}/{total_steps}] captured frame {len(frames)}")

    print(f"[INFO] Saving {len(frames)} frames to {OUTPUT_PATH} ...")
    try:
        import imageio
        imageio.mimwrite(OUTPUT_PATH, frames, fps=FPS, codec="libx264", quality=8)
        print(f"[OK] Saved: {OUTPUT_PATH}")
    except Exception as e:
        print(f"[WARN] imageio failed ({e}), trying cv2 ...")
        try:
            import cv2
            h, w = frames[0].shape[:2]
            out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
            for f in frames:
                out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            out.release()
            print(f"[OK] Saved: {OUTPUT_PATH}")
        except Exception as e2:
            print(f"[ERROR] Could not save video: {e2}")
            npy_path = OUTPUT_PATH.replace(".mp4", "_frames.npy")
            np.save(npy_path, np.stack(frames))
            print(f"[INFO] Saved raw frames to {npy_path} instead")

    simulation_app.close()


if __name__ == "__main__":
    main()
