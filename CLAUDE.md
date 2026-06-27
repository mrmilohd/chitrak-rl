# Chitrak Isaac Lab Project

## Repo
**github.com/mrmilohd/chitrak-rl** — everything is here, plug and play.

## What was done
1. Cloned `mars-iitr/chitrak` and `mrmilohd/chitrak-rl` into `chitrak-rl/`
2. Cloned `isaac-sim/IsaacLab` at tag `v2.3.2`
3. Fixed URDF mesh paths (were hardcoded to `/home/aaditya/Desktop/...`)
4. Added `chitrak-rl` as a Docker bind mount in `IsaacLab/docker/docker-compose.yaml` → `/workspace/chitrak-rl`
5. Converted `chitrak_fixed.urdf` → USD using `UrdfConverter` inside the container
6. Verified robot loads: 12 joints, correct actuators, correct init state
7. Found and fixed mirrored joint limits on left legs (fl, bl)
8. Pushed everything to GitHub with `setup.sh` for one-command restore

## Fresh session restore
```bash
git clone https://github.com/mrmilohd/chitrak-rl.git /teamspace/studios/this_studio
cd /teamspace/studios/this_studio
bash setup.sh
```

Then inside container:
```bash
export PYTHONPATH=/workspace/isaaclab/source/isaaclab:/workspace/isaaclab/source/chitrak_integration:$PYTHONPATH
./isaaclab.sh -p -m pip install flatdict
./isaaclab.sh -p /workspace/isaaclab/source/chitrak_integration/verify_robot.py
```

## Key paths
| What | Host path | Container path |
|------|-----------|----------------|
| Integration files | `IsaacLab/source/chitrak_integration/` | `/workspace/isaaclab/source/chitrak_integration/` |
| Robot meshes | `chitrak-rl/chitrak_description/meshes/` | `/workspace/chitrak-rl/chitrak_description/meshes/` |
| USD file | `chitrak-rl/isaac_lab_integration/usd_output/chitrak.usd` | `/workspace/isaaclab/source/chitrak_integration/usd_output/chitrak.usd` |
| Isaac Lab docker | `IsaacLab/docker/` | — |

## Robot specs
- 12 DOF quadruped (3 joints/leg × 4 legs: hip_roll, hip_pitch, knee)
- Joint naming: `{fr,fl,br,bl}_{hip_roll,hip_pitch,knee}_joint`
- Actuators: `DCMotorCfg`, effort=2.5 Nm, velocity=8.0 rad/s, stiffness=25.0, damping=0.5
- ⚠️ Left legs (fl, bl) have MIRRORED joint limits vs right legs — init positions must be negated

## Known issues
- `No module named 'isaaclab'` → fix with PYTHONPATH export above
- `No module named 'flatdict'` → `./isaaclab.sh -p -m pip install flatdict`
- Camera/viewport hangs headless → skip `record_fall.py`, use MuJoCo renderer instead
- USD conversion already done — don't re-run `convert_urdf_to_usd.py`

## What's next
- Create `chitrak_rough_env_cfg.py` subclassing `LocomotionVelocityRoughEnvCfg`
- Register env as `Isaac-Velocity-Flat-Chitrak-v0` in isaaclab_tasks
- Train with RSL-RL: `./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-Chitrak-v0 --headless`
- Tune rewards and actuator stiffness/damping based on training behavior
