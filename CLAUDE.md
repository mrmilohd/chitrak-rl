# Chitrak Isaac Lab Project

## What this is
Integrating the Chitrak quadruped robot (mars-iitr/chitrak) into Isaac Lab for RL locomotion training.

## Repo locations
- `chitrak-rl/` — original ROS + MuJoCo RL code (mars-iitr/chitrak + mrmilohd/chitrak-rl)
- `IsaacLab/` — Isaac Lab v2.3.2 (isaac-sim/IsaacLab)
- `IsaacLab/source/chitrak_integration/` — our integration files

## Integration files
- `chitrak_fixed.urdf` — URDF with corrected mesh paths (container: `/workspace/isaaclab/source/chitrak_integration/`)
- `chitrak_asset.py` — `CHITRAK_CFG` ArticulationCfg
- `chitrak_env_cfg.py` — DirectRLEnvCfg (backup, not used for training)
- `convert_urdf_to_usd.py` — one-time URDF→USD conversion (already done)
- `usd_output/chitrak.usd` — converted USD (already generated)
- `verify_robot.py` — sanity check script (confirmed working)
- `record_fall.py` — video recording (camera gets stuck headless, skip for now)

## Robot specs
- 12 DOF quadruped (3 joints/leg × 4 legs)
- Joint naming: `{fr,fl,br,bl}_{hip_roll,hip_pitch,knee}_joint`
- Left legs have mirrored joint limits vs right legs
- Actuators: effort_limit=2.5 Nm, velocity_limit=8.0 rad/s
- URDF meshes: `chitrak-rl/chitrak_description/meshes/*.STL`

## Container setup
```bash
cd /teamspace/studios/this_studio/IsaacLab/docker
./container.py enter
```

## Always set PYTHONPATH inside container
```bash
export PYTHONPATH=/workspace/isaaclab/source/isaaclab:/workspace/isaaclab/source/chitrak_integration:$PYTHONPATH
```

## Known issues
- `No module named 'isaaclab'` — fix with PYTHONPATH export above
- Camera/viewport hangs headless — skip record_fall.py for now
- Video recording needs a real display or VNC setup

## Next steps
- Register ChitrakEnv as a Gym env
- Set up RSL-RL training config
- Wire up rewards (Walk These Ways style from mjlab_stock_gait)
