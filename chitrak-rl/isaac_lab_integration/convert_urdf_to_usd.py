"""
Run this script INSIDE the Isaac Lab container to convert chitrak_fixed.urdf → USD.

Usage (from inside the container):
    cd /workspace/isaaclab
    python /teamspace/studios/this_studio/chitrak-rl/isaac_lab_integration/convert_urdf_to_usd.py
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert Chitrak URDF to USD")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args(["--headless"])
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

URDF_PATH = "/teamspace/studios/this_studio/chitrak-rl/isaac_lab_integration/chitrak_fixed.urdf"
USD_OUTPUT_DIR = "/teamspace/studios/this_studio/chitrak-rl/isaac_lab_integration/usd_output"

cfg = UrdfConverterCfg(
    asset_path=URDF_PATH,
    usd_dir=USD_OUTPUT_DIR,
    usd_file_name="chitrak.usd",
    fix_base=False,
    merge_fixed_joints=True,
    force_usd_conversion=True,
    make_instanceable=True,
    joint_drive=UrdfConverterCfg.JointDriveCfg(
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
            stiffness=25.0,
            damping=0.5,
        )
    ),
)

print(f"[INFO] Converting URDF: {URDF_PATH}")
converter = UrdfConverter(cfg)
print(f"[INFO] USD saved to: {converter.usd_path}")

simulation_app.close()
