# Chitrak → Isaac Lab: Full Setup Log

Everything done from scratch, in order, so anyone can replicate it.

---

## Environment

- Platform: Lightning AI Studio (SSH server, Linux)
- GPU: NVIDIA (container uses CUDA)
- Working directory: `/teamspace/studios/this_studio/`

---

## Step 1: Clone repos

```bash
# Chitrak robot (ROS + MuJoCo RL code)
git clone https://github.com/mars-iitr/chitrak chitrak-rl

# Chitrak RL extensions
git clone https://github.com/mrmilohd/chitrak-rl chitrak-rl   # merged into same folder

# Isaac Lab
git clone https://github.com/isaac-sim/IsaacLab.git IsaacLab
cd IsaacLab
git checkout v2.3.2
```

---

## Step 2: Build and start Isaac Lab Docker container

```bash
cd IsaacLab/docker
./container.py start    # answer N to X11 forwarding prompt
./container.py enter    # drops you into the container
```

The first `start` builds a Docker image (~7.5 GB Isaac Sim base + dependencies).  
Takes 10–20 minutes on first run. Subsequent starts are instant.

Inside the container you are at: `/workspace/isaaclab`

---

## Step 3: Fix PYTHONPATH (every new container session)

Isaac Lab's packages aren't on the default Python path inside the container.  
Run this every time you enter:

```bash
export PYTHONPATH=/workspace/isaaclab/source/isaaclab:/workspace/isaaclab/source/chitrak_integration:$PYTHONPATH
```

To make it permanent, add it to `~/.bashrc` inside the container:
```bash
echo 'export PYTHONPATH=/workspace/isaaclab/source/isaaclab:/workspace/isaaclab/source/chitrak_integration:$PYTHONPATH' >> ~/.bashrc
```

---

## Step 4: Create the integration folder

All our files live at:
- Host: `/teamspace/studios/this_studio/IsaacLab/source/chitrak_integration/`
- Container: `/workspace/isaaclab/source/chitrak_integration/`

These are the same directory — `source/` is a bind mount between host and container.

### 4a. Fix mesh paths in the URDF

The original URDF had absolute paths hardcoded to a developer's machine:
```
/home/aaditya/Desktop/chitrak/chitrak_description/meshes/
```

Fixed by replacing with container-visible path:
```
/workspace/chitrak-rl/chitrak_description/meshes/
```

Done via sed:
```bash
sed 's|/home/aaditya/Desktop/chitrak/chitrak_description/meshes/|/workspace/chitrak-rl/chitrak_description/meshes/|g' \
  chitrak-rl/chitrak_rl/chitrak.urdf \
  > IsaacLab/source/chitrak_integration/chitrak_fixed.urdf
```

### 4b. Add chitrak-rl as a Docker bind mount

Added to `IsaacLab/docker/docker-compose.yaml` under `x-default-isaac-lab-volumes`:
```yaml
- type: bind
  source: /teamspace/studios/this_studio/chitrak-rl
  target: /workspace/chitrak-rl
```

Then restarted the container:
```bash
./container.py stop
./container.py start   # N to X11
./container.py enter
```

---

## Step 5: Convert URDF → USD

Isaac Lab uses USD files, not raw URDFs. One-time conversion:

```bash
# Inside container
export PYTHONPATH=/workspace/isaaclab/source/isaaclab:/workspace/isaaclab/source/chitrak_integration:$PYTHONPATH
./isaaclab.sh -p -m pip install flatdict   # missing dep, install once

./isaaclab.sh -p /workspace/isaaclab/source/chitrak_integration/convert_urdf_to_usd.py
```

Output: `source/chitrak_integration/usd_output/chitrak.usd`  
(also generates `chitrak_base.usd`, `chitrak_physics.usd`, `chitrak_robot.usd`, `chitrak_sensor.usd`)

---

## Step 6: Verify robot loaded correctly

```bash
./isaaclab.sh -p /workspace/isaaclab/source/chitrak_integration/verify_robot.py
```

Expected output:
```
JOINT NAMES (12 total):
  [ 0] fr_hip_roll_joint
  [ 1] fr_hip_pitch_joint
  [ 2] fr_knee_joint
  [ 3] fl_hip_roll_joint
  ...

ACTUATORS:
  Group: 'legs'
    Type:           DCMotorCfg
    Joints:         ['fr_hip_roll_joint', ...]
    Effort limit:   2.5
    Velocity limit: 8.0
    Stiffness:      25.0
    Damping:        0.5

[OK] Chitrak loaded successfully!
```

**Important fix found during verification:**  
Left legs (`fl_*`, `bl_*`) have mirrored joint limits in the URDF.  
Initial joint positions must be:
- `fr/br_hip_pitch`: +0.5 rad (limits 0 to 3.14)
- `fl/bl_hip_pitch`: -0.5 rad (limits -3.14 to 0)
- `fr/br_knee`: -1.0 rad (limits -3.14 to 0)
- `fl/bl_knee`: +1.0 rad (limits 0 to 3.14)

---

## Integration files reference

| File | Purpose |
|------|---------|
| `chitrak_fixed.urdf` | Mesh-path-corrected URDF |
| `convert_urdf_to_usd.py` | URDF → USD conversion script |
| `usd_output/chitrak.usd` | Generated USD (ready to use) |
| `chitrak_asset.py` | `CHITRAK_CFG` — ArticulationCfg for the robot |
| `chitrak_env_cfg.py` | Basic DirectRLEnvCfg (obs/action/reward structure) |
| `verify_robot.py` | Sanity check: prints joints, actuators, init state |
| `record_fall.py` | Video recording (headless camera hangs — see note below) |

---

## Known issues

**Camera hangs headless:**  
`record_fall.py` gets stuck at viewport initialization in headless mode.  
Workaround: use MuJoCo renderer instead (works perfectly offscreen):
```bash
# Outside the container, in chitrak-rl/chitrak_rl/
python -c "
import mujoco, numpy as np, imageio
m = mujoco.MjModel.from_xml_path('chitrak_patched.xml')
d = mujoco.MjData(m)
renderer = mujoco.Renderer(m, height=720, width=1280)
frames = []
for _ in range(300):
    mujoco.mj_step(m, d)
    renderer.update_scene(d)
    frames.append(renderer.render())
imageio.mimwrite('chitrak_fall.mp4', frames, fps=30)
"
```

---

## How to proceed: Training

### Option A — Manager-based (recommended, same as Go1)

Create `chitrak_rough_env_cfg.py` in the integration folder:

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
```

Register as a Gym env (add to `isaaclab_tasks/manager_based/locomotion/velocity/config/__init__.py`):
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

Train with RSL-RL:
```bash
export PYTHONPATH=/workspace/isaaclab/source/isaaclab:/workspace/isaaclab/source/chitrak_integration:$PYTHONPATH

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-Chitrak-v0 \
  --num_envs 4096 \
  --headless
```

### Reward tuning

All rewards are in `LocomotionVelocityRoughEnvCfg.RewardsCfg`. Override in `__post_init__`:

```python
# Increase to push harder for velocity tracking
self.rewards.track_lin_vel_xy_exp.weight = 1.5

# Penalize unstable body orientation
self.rewards.flat_orientation_l2.weight = -2.5

# Reward feet leaving the ground (promotes gait)
self.rewards.feet_air_time.weight = 0.25

# Energy penalties
self.rewards.dof_torques_l2.weight = -1.0e-5
self.rewards.dof_acc_l2.weight = -2.5e-7
self.rewards.action_rate_l2.weight = -0.01

# Disable rewards not applicable
self.rewards.undesired_contacts = None
```

### Actuator tuning

In `chitrak_asset.py`, `DCMotorCfg` parameters:

| Param | Value | What it does |
|-------|-------|--------------|
| `effort_limit` | 2.5 Nm | Max torque, from URDF spec |
| `velocity_limit` | 8.0 rad/s | Max joint speed, from URDF spec |
| `stiffness` | 25.0 | Position gain (Kp) — higher = stiffer response |
| `damping` | 0.5 | Velocity damping (Kd) — higher = less oscillation |

If robot shakes/oscillates → increase `damping`.  
If robot is floppy/slow → increase `stiffness`.

### Play back a trained policy

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Flat-Chitrak-v0 \
  --num_envs 32 \
  --checkpoint logs/rsl_rl/chitrak_flat/*/model_*.pt
```

---

## File tree (final state)

```
/teamspace/studios/this_studio/
├── CLAUDE.md                          # project memory for Claude Code
├── chitrak-rl/
│   ├── SETUP_LOG.md                   # this file
│   ├── chitrak_description/meshes/    # 13 STL files for all links
│   └── chitrak_rl/
│       ├── chitrak.urdf               # original URDF (broken mesh paths)
│       ├── chitrak_patched.xml        # MuJoCo scene (works as-is)
│       └── mjlab_stock_gait/          # original MuJoCo Lab RL config
└── IsaacLab/
    ├── docker/docker-compose.yaml     # modified: added chitrak-rl bind mount
    └── source/
        └── chitrak_integration/       # all our files (visible in container too)
            ├── chitrak_fixed.urdf
            ├── chitrak_asset.py
            ├── chitrak_env_cfg.py
            ├── convert_urdf_to_usd.py
            ├── verify_robot.py
            ├── record_fall.py
            └── usd_output/
                ├── chitrak.usd
                ├── chitrak_base.usd
                ├── chitrak_physics.usd
                ├── chitrak_robot.usd
                └── chitrak_sensor.usd
```
