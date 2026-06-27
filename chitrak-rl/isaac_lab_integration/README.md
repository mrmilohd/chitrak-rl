# Chitrak → Isaac Lab Integration

Everything you need to get Chitrak running in Isaac Lab from a fresh server.
USD is already pre-generated and committed — no conversion needed.

---

## Fresh session setup (run every time)

### Step 1 — Clone this repo
```bash
git clone https://github.com/mrmilohd/chitrak-rl.git /teamspace/studios/this_studio
cd /teamspace/studios/this_studio
```

### Step 2 — Run setup script (clones IsaacLab, patches docker, starts container)
```bash
bash setup.sh
```
This takes ~15 min on first run (downloads Isaac Sim image). Instant on repeat runs.

### Step 3 — Enter the container
```bash
cd /teamspace/studios/this_studio/IsaacLab/docker
./container.py enter
```

### Step 4 — Set PYTHONPATH (every new container session)
```bash
export PYTHONPATH=/workspace/isaaclab/source/isaaclab:/workspace/isaaclab/source/chitrak_integration:$PYTHONPATH
```

### Step 5 — Install missing dep (once per container image)
```bash
./isaaclab.sh -p -m pip install flatdict
```

### Step 6 — Verify robot loads correctly
```bash
./isaaclab.sh -p /workspace/isaaclab/source/chitrak_integration/verify_robot.py
```

Expected output:
```
JOINT NAMES (12 total):
  [ 0] fr_hip_roll_joint
  ...
ACTUATORS:
  Group: 'legs'  |  effort_limit: 2.5  |  velocity_limit: 8.0
[OK] Chitrak loaded successfully!
```

---

## Files reference

| File | Purpose |
|------|---------|
| `chitrak_fixed.urdf` | URDF with corrected mesh paths |
| `chitrak_asset.py` | `CHITRAK_CFG` — ArticulationCfg for Isaac Lab |
| `chitrak_env_cfg.py` | DirectRLEnvCfg (obs/action/reward structure) |
| `verify_robot.py` | Sanity check — prints joints, actuators, init state |
| `record_fall.py` | Video recording (skip — headless camera hangs) |
| `convert_urdf_to_usd.py` | URDF→USD conversion (already done, don't re-run) |
| `usd_output/chitrak.usd` | Pre-generated USD file (ready to use) |

---

## Joint order (12 DOF)

| Index | Joint | Limits |
|-------|-------|--------|
| 0 | fr_hip_roll_joint | -1.57 to 1.57 |
| 1 | fr_hip_pitch_joint | 0 to 3.14 |
| 2 | fr_knee_joint | -3.14 to 0 |
| 3 | fl_hip_roll_joint | -1.57 to 1.57 |
| 4 | fl_hip_pitch_joint | -3.14 to 0 ← mirrored |
| 5 | fl_knee_joint | 0 to 3.14 ← mirrored |
| 6 | br_hip_roll_joint | -1.57 to 1.57 |
| 7 | br_hip_pitch_joint | 0 to 3.14 |
| 8 | br_knee_joint | -3.14 to 0 |
| 9 | bl_hip_roll_joint | -1.57 to 1.57 |
| 10 | bl_hip_pitch_joint | -3.14 to 0 ← mirrored |
| 11 | bl_knee_joint | 0 to 3.14 ← mirrored |

⚠️ Left legs (fl, bl) have mirrored joint limits — initial positions must be negated vs right legs.

---

## Training setup

### Create env config
Create `/workspace/isaaclab/source/chitrak_integration/chitrak_rough_env_cfg.py`:

```python
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from chitrak_asset import CHITRAK_CFG

@configclass
class ChitrakFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = CHITRAK_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None
        self.actions.joint_pos.scale = 0.25
        self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"
        # reward tuning
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 0.25
        self.rewards.dof_torques_l2.weight = -1.0e-5
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.undesired_contacts = None
```

### Register env
Add to `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/__init__.py`:
```python
import gymnasium as gym
from chitrak_rough_env_cfg import ChitrakFlatEnvCfg

gym.register(
    id="Isaac-Velocity-Flat-Chitrak-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"cfg": ChitrakFlatEnvCfg()},
    disable_env_checker=True,
)
```

### Train
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-Chitrak-v0 \
  --num_envs 4096 \
  --headless
```

### Play back trained policy
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Flat-Chitrak-v0 \
  --num_envs 32 \
  --checkpoint logs/rsl_rl/chitrak_flat/*/model_*.pt
```

---

## Actuator tuning

In `chitrak_asset.py`:

| Param | Current | Effect |
|-------|---------|--------|
| `stiffness` | 25.0 | Position gain (Kp) — higher = stiffer, more responsive |
| `damping` | 0.5 | Velocity damping (Kd) — higher = less oscillation |
| `effort_limit` | 2.5 Nm | Max torque (from URDF) |
| `velocity_limit` | 8.0 rad/s | Max joint speed (from URDF) |

- Robot shaking/oscillating → increase `damping`
- Robot floppy/slow → increase `stiffness`
