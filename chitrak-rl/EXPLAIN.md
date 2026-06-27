# Isaac Lab — How Everything Works, From Scratch

This document explains how Isaac Lab is structured, how we used it for Chitrak,
and how to add new rewards, observations, sensors, and curriculum from scratch.
Every section says exactly which file to open.

---

## 1. The Big Picture

Isaac Lab is built around a **manager-based** architecture. Instead of writing one big
training loop, you declare *what you want* in config classes and Isaac Lab wires it up.

```
ManagerBasedRLEnvCfg
├── SceneCfg         → what physical objects exist (robot, terrain, sensors, lights)
├── ActionsCfg       → how the policy controls the robot (joint positions, velocities)
├── ObservationsCfg  → what the policy sees (sensors, robot state)
├── RewardsCfg       → what behaviour is rewarded or penalised
├── EventCfg         → randomisation on reset (domain randomisation)
├── TerminationsCfg  → when to end an episode
└── CurriculumCfg    → how difficulty scales over training
```

Each section is just a Python dataclass. You change a number, training changes.

---

## 2. The Two Key Files

### The base class (read-only reference)
```
IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py
```
This defines all 7 managers for Go1/A1/Anymal. You never edit this directly — you subclass it.
Open it to see what defaults exist so you know what you can override.

### Your Chitrak env config (this is where you work)
```
IsaacLab/source/chitrak_integration/chitrak_rough_env_cfg.py   ← CREATE THIS FILE
```
All reward tuning, observation changes, sensor additions, curriculum — everything goes in `__post_init__` here:

```python
# File: IsaacLab/source/chitrak_integration/chitrak_rough_env_cfg.py

from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from chitrak_asset import CHITRAK_CFG

@configclass
class ChitrakFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()            # run parent setup first
        # everything below OVERRIDES the parent defaults
        self.scene.robot = CHITRAK_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
```

### `__post_init__` explained
Python dataclasses set all declared field values in `__init__`, then call `__post_init__`
immediately after. It's where you write override logic because `self` exists there but not
at class-definition time. `super().__post_init__()` runs the parent's setup before yours.

---

## 3. Rewards — How to Add, Remove, Tune

### Where to tune reward weights

**Open:** `IsaacLab/source/chitrak_integration/chitrak_rough_env_cfg.py`

All reward changes go inside `__post_init__`:

```python
def __post_init__(self):
    super().__post_init__()

    # --- TUNE WEIGHTS ---
    self.rewards.track_lin_vel_xy_exp.weight = 1.5        # was 1.0
    self.rewards.feet_air_time.weight = 0.25
    self.rewards.dof_torques_l2.weight = -1.0e-5
    self.rewards.action_rate_l2.weight = -0.01

    # --- TUNE PARAMS (not just weight) ---
    self.rewards.feet_air_time.params["threshold"] = 0.4  # seconds in air

    # --- DISABLE ---
    self.rewards.undesired_contacts = None
    self.rewards.flat_orientation_l2 = None

    # --- ADD a new built-in reward ---
    from isaaclab.managers import RewardTermCfg as RewTerm
    import isaaclab.envs.mdp as mdp
    self.rewards.stand_still = RewTerm(
        func=mdp.stand_still_joint_deviation_l1,
        weight=-0.5,
        params={"command_name": "base_velocity"}
    )
```

### Where the default reward values come from

**Open:** `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`

Look for `class RewardsCfg` — every field there is a reward you can override. Example:

```python
# Inside velocity_env_cfg.py (DO NOT EDIT — just read)
class RewardsCfg:
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_exp, weight=1.0, ...)
    lin_vel_z_l2         = RewTerm(func=mdp.lin_vel_z_l2,         weight=-2.0)
    ang_vel_xy_l2        = RewTerm(func=mdp.ang_vel_xy_l2,        weight=-0.05)
    dof_torques_l2       = RewTerm(func=mdp.joint_torques_l2,     weight=-1.0e-5)
    dof_acc_l2           = RewTerm(func=mdp.joint_acc_l2,         weight=-2.5e-7)
    action_rate_l2       = RewTerm(func=mdp.action_rate_l2,       weight=-0.01)
    feet_air_time        = RewTerm(func=mdp.feet_air_time,        weight=0.125, ...)
    undesired_contacts   = RewTerm(func=mdp.undesired_contacts,   weight=-1.0, ...)
    flat_orientation_l2  = RewTerm(func=mdp.flat_orientation_l2,  weight=0.0)
    dof_pos_limits       = RewTerm(func=mdp.joint_pos_limits,     weight=0.0)
```

### Where the reward function code lives

Two files — both are read-only references:

**Built-in (general):**
```
IsaacLab/source/isaaclab/isaaclab/envs/mdp/rewards.py
```

| Function | What it rewards/penalises |
|----------|--------------------------|
| `track_lin_vel_xy_exp` | Forward/sideways velocity matching command |
| `track_ang_vel_z_exp` | Yaw rate matching command |
| `lin_vel_z_l2` | Vertical bouncing |
| `ang_vel_xy_l2` | Rolling/pitching body |
| `flat_orientation_l2` | Body tilt from upright |
| `base_height_l2` | Deviation from target height |
| `joint_torques_l2` | High torques (energy use) |
| `joint_vel_l2` | High joint speeds |
| `joint_acc_l2` | Jerky motion |
| `joint_deviation_l1` | Deviation from default pose |
| `joint_pos_limits` | Joints near hard limits |
| `action_rate_l2` | Rapid action changes |
| `action_l2` | Large actions |
| `undesired_contacts` | Knee/body hitting ground |
| `desired_contacts` | Feet making contact |
| `contact_forces` | Large impact forces |
| `is_alive` | Surviving (positive reward) |

**Locomotion-specific:**
```
IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py
```

| Function | What it rewards/penalises |
|----------|--------------------------|
| `feet_air_time` | Feet leaving ground (promotes gait) |
| `feet_slide` | Feet sliding while in contact |
| `track_lin_vel_xy_yaw_frame_exp` | Velocity tracking in yaw frame |
| `stand_still_joint_deviation_l1` | Moving when command is zero |

### Writing a custom reward

**Step 1 — Create the function:**

```
IsaacLab/source/chitrak_integration/chitrak_rewards.py   ← CREATE THIS FILE
```

```python
# File: IsaacLab/source/chitrak_integration/chitrak_rewards.py

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

def penalise_hip_roll(env: ManagerBasedRLEnv,
                      asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalise large hip roll angles — keeps the robot from leaning sideways."""
    asset = env.scene[asset_cfg.name]
    hip_roll_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(hip_roll_pos), dim=1)  # shape: (num_envs,)
```

**The signature must always be:** `func(env, **params) -> Tensor shape (num_envs,)`

**Step 2 — Register it in your env cfg:**

```
IsaacLab/source/chitrak_integration/chitrak_rough_env_cfg.py
```

```python
from chitrak_rewards import penalise_hip_roll
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg

# inside __post_init__:
self.rewards.hip_roll_penalty = RewTerm(
    func=penalise_hip_roll,
    weight=-0.1,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint"])}
)
```

---

## 4. Observations — What the Policy Sees

### Where to change observations

**Open:** `IsaacLab/source/chitrak_integration/chitrak_rough_env_cfg.py`

```python
def __post_init__(self):
    super().__post_init__()

    # --- DISABLE an observation ---
    self.observations.policy.height_scan = None   # remove terrain heightmap

    # --- TUNE NOISE on existing observation ---
    from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
    self.observations.policy.joint_pos.noise = Unoise(n_min=-0.05, n_max=0.05)

    # --- ADD a new observation ---
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    import isaaclab.envs.mdp as mdp
    self.observations.policy.base_height = ObsTerm(func=mdp.base_pos_z)
```

### Where the default observations are declared

**Open:** `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`

Look for `class ObservationsCfg → class PolicyCfg`:

```python
# Inside velocity_env_cfg.py (DO NOT EDIT — read only)
class PolicyCfg(ObsGroup):
    base_lin_vel      = ObsTerm(func=mdp.base_lin_vel, ...)     # 3 values
    base_ang_vel      = ObsTerm(func=mdp.base_ang_vel, ...)     # 3 values
    projected_gravity = ObsTerm(func=mdp.projected_gravity, ...)# 3 values
    velocity_commands = ObsTerm(func=mdp.generated_commands, ...)# 3 values
    joint_pos         = ObsTerm(func=mdp.joint_pos_rel, ...)    # 12 values
    joint_vel         = ObsTerm(func=mdp.joint_vel_rel, ...)    # 12 values
    actions           = ObsTerm(func=mdp.last_action)           # 12 values
    height_scan       = ObsTerm(func=mdp.height_scan, ...)      # N values (disable for flat)
    # total (no height_scan): 3+3+3+3+12+12+12 = 48
```

All terms are auto-concatenated into one flat vector fed to the policy.

### Where the observation function code lives

**Open:** `IsaacLab/source/isaaclab/isaaclab/envs/mdp/observations.py`

| Function | Output shape | What it is |
|----------|-------------|------------|
| `base_pos_z` | (1,) | Base height above ground |
| `base_lin_vel` | (3,) | Linear velocity in base frame |
| `base_ang_vel` | (3,) | Angular velocity in base frame |
| `projected_gravity` | (3,) | Gravity vector in base frame (tilt sensor) |
| `root_pos_w` | (3,) | Position in world frame |
| `root_lin_vel_w` | (3,) | Linear velocity in world frame |
| `root_ang_vel_w` | (3,) | Angular velocity in world frame |
| `joint_pos` | (num_joints,) | Absolute joint positions |
| `joint_pos_rel` | (num_joints,) | Joint positions relative to default |
| `joint_vel` | (num_joints,) | Absolute joint velocities |
| `joint_vel_rel` | (num_joints,) | Joint velocities relative to default |
| `joint_effort` | (num_joints,) | Applied joint torques |
| `height_scan` | (N,) | Terrain height map around robot |
| `imu_orientation` | (4,) | IMU quaternion |
| `imu_ang_vel` | (3,) | IMU angular velocity |
| `imu_lin_acc` | (3,) | IMU linear acceleration |
| `generated_commands` | (3,) | Current velocity command (vx, vy, wz) |
| `last_action` | (num_actions,) | Previous action sent to robot |

### Writing a custom observation

**Step 1 — Create the function** (can go in `chitrak_rewards.py` or a new file):

```
IsaacLab/source/chitrak_integration/chitrak_rewards.py
```

```python
# File: IsaacLab/source/chitrak_integration/chitrak_rewards.py

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

def foot_contact_state(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Returns binary 1/0 for each foot: 1 = in contact, 0 = in air. Shape: (num_envs, 4)."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return (sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1) > 1.0).float()
```

**Step 2 — Register in your env cfg:**

```
IsaacLab/source/chitrak_integration/chitrak_rough_env_cfg.py
```

```python
from chitrak_rewards import foot_contact_state
from isaaclab.managers import ObservationTermCfg as ObsTerm, SceneEntityCfg

# inside __post_init__:
self.observations.policy.foot_contacts = ObsTerm(
    func=foot_contact_state,
    params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_calf_link"])}
)
```

> Note: adding 4 foot-contact observations changes `num_observations` from 48 → 52.
> The RSL-RL trainer reads `num_observations` from the env automatically, so no manual update needed.

---

## 5. Sensors — Contact, Height Scanner, IMU

### Where sensors are declared

**Open:** `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`

Look for `class MySceneCfg` — sensors are fields here:

```python
# Inside velocity_env_cfg.py (read only)
class MySceneCfg(InteractiveSceneCfg):
    robot          = ...
    terrain        = ...
    height_scanner = RayCasterCfg(...)    # terrain heightmap
    contact_forces = ContactSensorCfg(...)# foot/body contact detection
    light          = ...
```

### Where sensor class definitions live

```
IsaacLab/source/isaaclab/isaaclab/sensors/contact_sensor.py   ← ContactSensorCfg
IsaacLab/source/isaaclab/isaaclab/sensors/ray_caster/         ← RayCasterCfg
IsaacLab/source/isaaclab/isaaclab/sensors/imu.py              ← ImuCfg
```

### Adding / modifying sensors

**Open:** `IsaacLab/source/chitrak_integration/chitrak_rough_env_cfg.py`

```python
def __post_init__(self):
    super().__post_init__()

    # --- DISABLE height scanner (flat terrain) ---
    self.scene.height_scanner = None
    self.observations.policy.height_scan = None

    # --- CHANGE which body the contact sensor attaches to ---
    # (parent attaches to ".*" = all bodies — keep this for Chitrak)
    # Just change which bodies you USE in rewards:
    self.rewards.feet_air_time.params["sensor_cfg"] = SceneEntityCfg(
        "contact_forces", body_names=[".*_calf_link"]   # Chitrak's foot bodies
    )
    self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
        "contact_forces", body_names=[".*_thigh_link", "torso_link"]
    )
    self.terminations.base_contact.params["sensor_cfg"] = SceneEntityCfg(
        "contact_forces", body_names=["torso_link"]
    )

    # --- ADD an IMU sensor ---
    from isaaclab.sensors import ImuCfg
    self.scene.imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        update_period=0.0,
    )
    # then use it in observations:
    import isaaclab.envs.mdp as mdp
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    self.observations.policy.imu_ang_vel = ObsTerm(
        func=mdp.imu_ang_vel,
        params={"asset_cfg": SceneEntityCfg("imu")}
    )
```

---

## 6. Actions — How the Policy Controls the Robot

### Where actions are defined

**Open:** `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`

```python
# Inside velocity_env_cfg.py (read only)
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],    # all joints
        scale=0.5,             # action is multiplied by this before adding to default pos
        use_default_offset=True
    )
```

### Tuning the action scale

**Open:** `IsaacLab/source/chitrak_integration/chitrak_rough_env_cfg.py`

```python
def __post_init__(self):
    super().__post_init__()
    self.actions.joint_pos.scale = 0.25   # smaller = more conservative movements
```

A scale of 0.25 means the policy outputs values in roughly [-1, 1], which get multiplied
to ±0.25 rad before being added to the default joint position. Go1 flat uses 0.25.

---

## 7. Robot Asset — Joints, Actuators, Init Position

### File to edit

```
IsaacLab/source/chitrak_integration/chitrak_asset.py
```

This is the only file that controls the robot's physical properties in simulation.

```python
# File: IsaacLab/source/chitrak_integration/chitrak_asset.py

CHITRAK_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/workspace/isaaclab/source/chitrak_integration/usd_output/chitrak.usd",
        activate_contact_sensors=True,   # MUST be True for contact rewards to work
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.25),           # spawn height — tune if robot spawns inside ground
        joint_pos={                      # starting pose
            ".*_hip_roll_joint":  0.0,
            "fr_hip_pitch_joint": 0.5,   "br_hip_pitch_joint": 0.5,
            "fl_hip_pitch_joint": -0.5,  "bl_hip_pitch_joint": -0.5,  # mirrored
            "fr_knee_joint": -1.0,       "br_knee_joint": -1.0,
            "fl_knee_joint":  1.0,       "bl_knee_joint":  1.0,        # mirrored
        },
    ),
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=[".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint"],
            effort_limit=2.5,    # Nm — from URDF
            velocity_limit=8.0,  # rad/s — from URDF
            stiffness=25.0,      # Kp — position gain. Tune if robot is floppy or shaky
            damping=0.5,         # Kd — velocity damping. Tune if robot oscillates
        )
    }
)
```

### Reference for Go1 actuator values

**Open:** `IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/unitree.py`

The Go1's DCMotorCfg is what we copied stiffness/damping from. Chitrak's motors are
weaker (2.5 Nm vs Go1's ~23 Nm) so these values likely need reducing.

---

## 8. Domain Randomisation (Events)

### Where default events are defined

**Open:** `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`

Look for `class EventCfg` — it has: physics material, base mass, COM offset, external forces, joint reset, robot push.

### Where to tune/disable events

**Open:** `IsaacLab/source/chitrak_integration/chitrak_rough_env_cfg.py`

```python
def __post_init__(self):
    super().__post_init__()

    # disable random pushes during initial training
    self.events.push_robot = None

    # reduce mass randomisation range for small robot
    self.events.add_base_mass.params["mass_distribution_params"] = (-0.1, 0.1)

    # reduce friction range
    self.events.physics_material.params["static_friction_range"] = (0.8, 1.0)
    self.events.physics_material.params["dynamic_friction_range"] = (0.6, 0.8)
```

### Where event function code lives

```
IsaacLab/source/isaaclab/isaaclab/envs/mdp/events.py
```

Key functions: `randomize_rigid_body_material`, `randomize_rigid_body_mass`,
`push_by_setting_velocity`, `reset_joints_by_scale`, `reset_root_state_uniform`

---

## 9. Terminations — When to End an Episode

### Where defaults are defined

**Open:** `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`

```python
# Inside velocity_env_cfg.py (read only)
class TerminationsCfg:
    time_out     = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(func=mdp.illegal_contact, params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base"]),
        "threshold": 1.0,
    })
```

### Where to change them

**Open:** `IsaacLab/source/chitrak_integration/chitrak_rough_env_cfg.py`

```python
def __post_init__(self):
    super().__post_init__()

    # fix body name — Chitrak's base is "torso_link" not "base"
    self.terminations.base_contact.params["sensor_cfg"] = SceneEntityCfg(
        "contact_forces", body_names=["torso_link"]
    )

    # change episode length
    self.episode_length_s = 15.0
```

### Where termination function code lives

```
IsaacLab/source/isaaclab/isaaclab/envs/mdp/terminations.py
IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/terminations.py
```

---

## 10. Curriculum — Increasing Difficulty

### Where the terrain curriculum lives

```
IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/curriculums.py
```

Function `terrain_levels_vel`: robot walked far → harder tile. Robot walked little → easier tile.

### Where to configure curriculum

**Open:** `IsaacLab/source/chitrak_integration/chitrak_rough_env_cfg.py`

```python
def __post_init__(self):
    super().__post_init__()

    # for flat terrain training — disable terrain curriculum entirely
    self.curriculum.terrain_levels = None
    self.scene.terrain.terrain_type = "plane"
    self.scene.terrain.terrain_generator = None
```

### Custom curriculum (command velocity ramp)

Add this function to `chitrak_rewards.py` or a new `chitrak_curriculum.py`:

```
IsaacLab/source/chitrak_integration/chitrak_rewards.py
```

```python
import torch
def scale_commands_with_progress(env, env_ids, command_name: str):
    """Ramp command speed from 30% → 100% over training."""
    progress = env.common_step_counter / 10_000_000
    scale = min(1.0, 0.3 + 0.7 * progress)
    env.command_manager.get_term(command_name).cfg.ranges.lin_vel_x = (-scale, scale)
    return torch.tensor(scale)
```

Register it in `chitrak_rough_env_cfg.py`:

```python
from chitrak_rewards import scale_commands_with_progress
from isaaclab.managers import CurriculumTermCfg as CurrTerm

self.curriculum.command_scale = CurrTerm(
    func=scale_commands_with_progress,
    params={"command_name": "base_velocity"}
)
```

---

## 11. How We Did It for Chitrak — Step by Step

### Step 1: Robot asset
**File created:** `IsaacLab/source/chitrak_integration/chitrak_asset.py`

Values sourced from:
- `effort_limit`, `velocity_limit` → read directly from `chitrak_fixed.urdf` (`<limit effort="2.5" velocity="8.0"/>`)
- `stiffness=25.0`, `damping=0.5` → copied from `IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/unitree.py` (Go1) — needs tuning
- `pos=(0.0, 0.0, 0.25)` → guessed
- Mirrored init positions for fl/bl → derived from URDF joint limit inspection

### Step 2: URDF → USD conversion (already done, don't re-run)
**File used:** `IsaacLab/source/chitrak_integration/convert_urdf_to_usd.py`
**Output:** `IsaacLab/source/chitrak_integration/usd_output/chitrak.usd`

### Step 3: Env config
**File created:** `IsaacLab/source/chitrak_integration/chitrak_env_cfg.py` (DirectRLEnvCfg version — use manager-based instead for training)
**File to create next:** `IsaacLab/source/chitrak_integration/chitrak_rough_env_cfg.py`

### Step 4: Verify robot
**File used:** `IsaacLab/source/chitrak_integration/verify_robot.py`

---

## 12. Training & Iteration Loop

### Train

```bash
# Inside Isaac Lab container:
export PYTHONPATH=/workspace/isaaclab/source/isaaclab:/workspace/isaaclab/source/chitrak_integration:$PYTHONPATH

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-Chitrak-v0 \
  --num_envs 4096 \
  --headless
```

Logs: `IsaacLab/logs/rsl_rl/Isaac-Velocity-Flat-Chitrak-v0/`

### Watch training

```bash
./isaaclab.sh -p -m tensorboard.main --logdir logs/rsl_rl/
```

### Play back policy

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Flat-Chitrak-v0 \
  --num_envs 32 \
  --checkpoint logs/rsl_rl/Isaac-Velocity-Flat-Chitrak-v0/*/model_*.pt
```

### Typical iteration loop

1. Train ~1000 iterations, open tensorboard
2. `track_lin_vel` not increasing → robot not moving → increase `feet_air_time` weight
3. Robot falls immediately → `stiffness` too low, or `pos` spawns inside ground
4. Robot shakes → increase `damping` in `chitrak_asset.py`, increase `action_rate_l2` penalty
5. Robot drags feet → add `feet_slide` reward (negative weight)
6. Adjust in `chitrak_rough_env_cfg.py` (weights) or `chitrak_asset.py` (actuators), re-train from checkpoint

---

## 13. Complete File Map

```
IsaacLab/source/
│
├── chitrak_integration/                    ← ALL CHITRAK FILES LIVE HERE
│   ├── chitrak_asset.py                    ← robot USD path, joints, actuators, init pos
│   ├── chitrak_rough_env_cfg.py            ← CREATE THIS: your env overrides
│   ├── chitrak_rewards.py                  ← CREATE THIS: custom reward/obs functions
│   ├── chitrak_env_cfg.py                  ← old DirectRLEnvCfg (don't use for training)
│   ├── verify_robot.py                     ← sanity check (run once)
│   ├── convert_urdf_to_usd.py              ← already ran, don't re-run
│   └── usd_output/chitrak.usd             ← pre-generated, ready to use
│
├── isaaclab/isaaclab/envs/mdp/
│   ├── rewards.py                          ← all built-in reward functions
│   ├── observations.py                     ← all built-in observation functions
│   ├── events.py                           ← domain randomisation functions
│   └── terminations.py                     ← termination functions
│
├── isaaclab_assets/isaaclab_assets/robots/
│   ├── unitree.py                          ← Go1/A1 configs (reference for actuators)
│   └── anymal.py                           ← Anymal configs (another reference)
│
└── isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/
    ├── velocity_env_cfg.py                 ← BASE CLASS (read to see defaults)
    ├── mdp/
    │   ├── rewards.py                      ← feet_air_time, feet_slide, etc.
    │   ├── curriculums.py                  ← terrain_levels_vel
    │   └── terminations.py                 ← terrain_out_of_bounds
    └── config/go1/                         ← Go1 flat+rough configs (copy as pattern)
```

---

## 14. Quick Reference

| Task | File to open |
|------|-------------|
| Change a reward weight | `chitrak_integration/chitrak_rough_env_cfg.py` → `__post_init__` |
| See what rewards exist | `velocity_env_cfg.py` → `class RewardsCfg` |
| See reward function code | `isaaclab/envs/mdp/rewards.py` or `locomotion/velocity/mdp/rewards.py` |
| Add a custom reward | create function in `chitrak_rewards.py`, register in `chitrak_rough_env_cfg.py` |
| Add/remove an observation | `chitrak_rough_env_cfg.py` → `self.observations.policy.X = ...` |
| See observation function code | `isaaclab/envs/mdp/observations.py` |
| Change action scale | `chitrak_rough_env_cfg.py` → `self.actions.joint_pos.scale = 0.25` |
| Change actuator stiffness/damping | `chitrak_integration/chitrak_asset.py` → `DCMotorCfg` |
| Change spawn height | `chitrak_integration/chitrak_asset.py` → `init_state.pos` |
| Disable terrain curriculum | `chitrak_rough_env_cfg.py` → `self.curriculum.terrain_levels = None` |
| Disable domain randomisation | `chitrak_rough_env_cfg.py` → `self.events.push_robot = None` |
| Change episode length | `chitrak_rough_env_cfg.py` → `self.episode_length_s = 15.0` |
| Change sim frequency | `chitrak_rough_env_cfg.py` → `self.sim.dt`, `self.decimation` |
