# Chitrak → Isaac Lab Integration

## Files

| File | Purpose |
|------|---------|
| `chitrak_fixed.urdf` | Original URDF with mesh paths fixed for this server |
| `convert_urdf_to_usd.py` | One-time conversion: URDF → USD (run inside container) |
| `chitrak_asset.py` | Isaac Lab `ArticulationCfg` for Chitrak |
| `chitrak_env_cfg.py` | Isaac Lab `DirectRLEnvCfg` for locomotion training |

## Step-by-step

### 1. Enter the Isaac Lab container
```bash
cd /teamspace/studios/this_studio/IsaacLab/docker
./container.py enter
```

### 2. Convert URDF to USD (one-time, takes ~1 min)
```bash
cd /workspace/isaaclab
python /teamspace/studios/this_studio/chitrak-rl/isaac_lab_integration/convert_urdf_to_usd.py
```
Output: `isaac_lab_integration/usd_output/chitrak.usd`

### 3. Verify the USD loaded correctly (optional sanity check)
```bash
cd /workspace/isaaclab
python scripts/tools/check_urdf.py \
  /teamspace/studios/this_studio/chitrak-rl/isaac_lab_integration/chitrak_fixed.urdf
```

### 4. Train
```bash
cd /workspace/isaaclab
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Chitrak-v0 \
  --headless
```
(You will need to register the env — see below.)

## Registering the environment

Add to `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/__init__.py`:
```python
import gymnasium as gym
from chitrak_env_cfg import ChitrakEnvCfg

gym.register(
    id="Chitrak-v0",
    entry_point="isaaclab.envs:DirectRLEnv",
    kwargs={"cfg": ChitrakEnvCfg()},
)
```

## Joint order (12 DOF)
```
fr_hip_roll_joint   fl_hip_roll_joint   br_hip_roll_joint   bl_hip_roll_joint
fr_hip_pitch_joint  fl_hip_pitch_joint  br_hip_pitch_joint  bl_hip_pitch_joint
fr_knee_joint       fl_knee_joint       br_knee_joint       bl_knee_joint
```
