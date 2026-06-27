# Isaac Lab — How Everything Works, From Scratch

This document explains how Isaac Lab is structured, how we used it for Chitrak,
and how to add new rewards, observations, sensors, and curriculum from scratch.

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

## 2. The File That Does Everything: `velocity_env_cfg.py`

**Location:** `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`

This is the base locomotion config that Go1, A1, Anymal etc. all inherit from.
For Chitrak we subclass this and override only what differs.

```python
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

@configclass
class ChitrakFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # override only what you need
        self.scene.robot = CHITRAK_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
```

---

## 3. Rewards — How to Add, Remove, Tune

### How rewards work

Every reward is one line in `RewardsCfg`. It points to a Python function and has a weight:

```python
from isaaclab.managers import RewardTermCfg as RewTerm
import isaaclab.envs.mdp as mdp

class RewardsCfg:
    track_lin_vel = RewTerm(func=mdp.track_lin_vel_xy_exp, weight=1.0,
                            params={"command_name": "base_velocity", "std": 0.5})
    penalise_torques = RewTerm(func=mdp.joint_torques_l2, weight=-1e-5)
```

- **Positive weight** → behaviour is encouraged
- **Negative weight** → behaviour is penalised
- **weight=0** → disabled but still tracked in logs

### All built-in reward functions (ready to use, no code needed)

These live in `IsaacLab/source/isaaclab/isaaclab/envs/mdp/rewards.py`:

| Function | What it rewards |
|----------|----------------|
| `track_lin_vel_xy_exp` | Forward/sideways velocity matching command |
| `track_ang_vel_z_exp` | Yaw rate matching command |
| `lin_vel_z_l2` | Penalise vertical bouncing |
| `ang_vel_xy_l2` | Penalise rolling/pitching |
| `flat_orientation_l2` | Penalise body tilt |
| `base_height_l2` | Penalise deviation from target height |
| `joint_torques_l2` | Penalise high torques (energy) |
| `joint_vel_l2` | Penalise high joint speeds |
| `joint_acc_l2` | Penalise jerky motion |
| `joint_deviation_l1` | Penalise deviation from default pose |
| `joint_pos_limits` | Penalise joints near limits |
| `action_rate_l2` | Penalise rapid action changes |
| `action_l2` | Penalise large actions |
| `undesired_contacts` | Penalise knee/body hitting ground |
| `desired_contacts` | Reward feet making contact |
| `contact_forces` | Penalise large impact forces |
| `is_alive` | Reward surviving |

Also in `mdp/rewards.py` (locomotion-specific):

| Function | What it rewards |
|----------|----------------|
| `feet_air_time` | Reward feet leaving ground (promotes gait) |
| `feet_slide` | Penalise feet sliding on ground |
| `track_lin_vel_xy_yaw_frame_exp` | Velocity tracking in yaw frame |
| `stand_still_joint_deviation_l1` | Penalise moving when command is zero |

### Tuning a reward

In your env cfg `__post_init__`:

```python
# Change weight only
self.rewards.track_lin_vel_xy_exp.weight = 1.5

# Change weight and params
self.rewards.feet_air_time.weight = 0.25
self.rewards.feet_air_time.params["threshold"] = 0.4

# Disable a reward
self.rewards.undesired_contacts = None

# Add a new built-in reward
from isaaclab.managers import RewardTermCfg as RewTerm
import isaaclab.envs.mdp as mdp

self.rewards.stand_still = RewTerm(
    func=mdp.stand_still_joint_deviation_l1,
    weight=-0.5,
    params={"command_name": "base_velocity"}
)
```

### Writing a custom reward from scratch

Create a file `chitrak_rewards.py` in `chitrak_integration/`:

```python
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

def penalise_hip_roll(env: ManagerBasedRLEnv,
                      asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalise large hip roll angles — keeps the robot upright."""
    asset = env.scene[asset_cfg.name]
    # joint_pos shape: (num_envs, num_joints)
    # get indices of all hip_roll joints
    hip_roll_ids = asset_cfg.joint_ids  # set via SceneEntityCfg(joint_names=[".*_hip_roll_joint"])
    hip_roll_pos = asset.data.joint_pos[:, hip_roll_ids]
    return torch.sum(torch.square(hip_roll_pos), dim=1)
```

Then add it to your env cfg:

```python
from chitrak_rewards import penalise_hip_roll
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg

self.rewards.hip_roll_penalty = RewTerm(
    func=penalise_hip_roll,
    weight=-0.1,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint"])}
)
```

**The function signature must always be:** `func(env, **params) -> torch.Tensor` returning shape `(num_envs,)`.

---

## 4. Observations — What the Policy Sees

### How observations work

```python
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel    = ObsTerm(func=mdp.base_lin_vel)           # shape (3,)
        base_ang_vel    = ObsTerm(func=mdp.base_ang_vel)           # shape (3,)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)    # shape (3,)
        velocity_commands = ObsTerm(func=mdp.generated_commands,   # shape (3,)
                                    params={"command_name": "base_velocity"})
        joint_pos       = ObsTerm(func=mdp.joint_pos_rel)          # shape (12,)
        joint_vel       = ObsTerm(func=mdp.joint_vel_rel)          # shape (12,)
        actions         = ObsTerm(func=mdp.last_action)            # shape (12,)
        # total = 3+3+3+3+12+12+12 = 48
```

All terms are concatenated automatically into one flat vector.

### All built-in observation functions

From `IsaacLab/source/isaaclab/isaaclab/envs/mdp/observations.py`:

| Function | Output shape | What it is |
|----------|-------------|------------|
| `base_pos_z` | (1,) | Base height above ground |
| `base_lin_vel` | (3,) | Linear velocity in base frame |
| `base_ang_vel` | (3,) | Angular velocity in base frame |
| `projected_gravity` | (3,) | Gravity vector in base frame |
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
| `generated_commands` | (3,) | Current velocity command |
| `last_action` | (num_actions,) | Previous action |

### Adding noise to an observation

```python
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise

joint_pos = ObsTerm(func=mdp.joint_pos_rel,
                    noise=Unoise(n_min=-0.01, n_max=0.01))   # uniform ±0.01 rad
base_lin_vel = ObsTerm(func=mdp.base_lin_vel,
                       noise=Gnoise(mean=0.0, std=0.1))       # gaussian noise
```

### Adding a new observation

Disable one in `__post_init__`:
```python
self.observations.policy.height_scan = None   # remove height scan
```

Add a custom one:
```python
def foot_contact_state(env, sensor_cfg):
    """Returns 1/0 for each foot in contact."""
    from isaaclab.sensors import ContactSensor
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return (sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1) > 1.0).float()

self.observations.policy.foot_contacts = ObsTerm(
    func=foot_contact_state,
    params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_calf_link"])}
)
```

---

## 5. Sensors — Contact, Height Scanner, IMU

### Contact sensor (already in Go1 config)

In `MySceneCfg`:
```python
from isaaclab.sensors import ContactSensorCfg

contact_forces = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/.*",   # attach to all bodies
    history_length=3,
    track_air_time=True                    # needed for feet_air_time reward
)
```

Use in rewards:
```python
self.rewards.feet_air_time.params["sensor_cfg"] = SceneEntityCfg(
    "contact_forces", body_names=[".*_calf_link"]   # Chitrak foot links
)
self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
    "contact_forces", body_names=[".*_thigh_link", "torso_link"]
)
```

### Height scanner (terrain awareness)

```python
from isaaclab.sensors import RayCasterCfg, patterns

height_scanner = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/torso_link",   # attach to base
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    ray_alignment="yaw",
    pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
    mesh_prim_paths=["/World/ground"],
)
```

Use in observations:
```python
height_scan = ObsTerm(
    func=mdp.height_scan,
    params={"sensor_cfg": SceneEntityCfg("height_scanner")},
    noise=Unoise(n_min=-0.1, n_max=0.1),
    clip=(-1.0, 1.0),
)
```

For Chitrak flat terrain (no hills) — disable it:
```python
self.scene.height_scanner = None
self.observations.policy.height_scan = None
```

### IMU sensor

```python
from isaaclab.sensors import ImuCfg

imu = ImuCfg(
    prim_path="{ENV_REGEX_NS}/Robot/torso_link",
    update_period=0.0,   # update every step
)
```

Use in observations:
```python
imu_ang_vel = ObsTerm(func=mdp.imu_ang_vel,
                      params={"asset_cfg": SceneEntityCfg("imu")})
```

---

## 6. Domain Randomisation (Events)

Events run on reset to randomise the simulation — this is what makes trained policies
transfer to real robots.

### Built-in event functions (from `isaaclab.envs.mdp`)

```python
from isaaclab.managers import EventTermCfg as EventTerm
import isaaclab.envs.mdp as mdp

@configclass
class EventCfg:
    # Randomise physics material (friction) at episode start
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.4, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # Add random mass to base at each reset
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-0.5, 0.5),   # kg added
            "operation": "add",
        },
    )

    # Push robot with random force
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
            }
        },
    )

    # Reset joints to default positions
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.9, 1.1),  # ±10% of default
            "velocity_range": (0.0, 0.0),
        },
    )
```

### Disable an event

```python
self.events.push_robot = None
self.events.base_com = None
```

---

## 7. Terminations — When to End an Episode

```python
from isaaclab.managers import TerminationTermCfg as DoneTerm
import isaaclab.envs.mdp as mdp

@configclass
class TerminationsCfg:
    # Always needed — ends episode after max time
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # End if base hits ground (body collision)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"),
            "threshold": 1.0,
        },
    )
```

For Chitrak — the torso body is `torso_link`:
```python
self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"
```

---

## 8. Curriculum — Increasing Difficulty

### Terrain curriculum (Go1-style)

The built-in terrain curriculum moves robots to harder terrain tiles when they
walk far enough, and easier tiles when they struggle.

It lives in `mdp/curriculums.py` as `terrain_levels_vel`.

```python
from isaaclab.managers import CurriculumTermCfg as CurrTerm
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

@configclass
class CurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
```

The logic: robot walked > half the terrain tile width → move up. Walked < half
of commanded distance → move down.

### Disable curriculum (flat terrain training)

```python
self.curriculum.terrain_levels = None
self.scene.terrain.terrain_type = "plane"
self.scene.terrain.terrain_generator = None
```

### Custom curriculum — command velocity scaling

Start with slow commands, increase as robot improves:

```python
def scale_commands_with_progress(env, env_ids, command_name: str):
    """Increase command velocity range as training progresses."""
    progress = env.common_step_counter / 10_000_000   # fraction of total steps
    scale = min(1.0, 0.3 + 0.7 * progress)           # ramp from 30% to 100%
    env.command_manager.get_term(command_name).cfg.ranges.lin_vel_x = (
        -scale, scale
    )
    return torch.mean(torch.tensor(scale))

self.curriculum.command_scale = CurrTerm(
    func=scale_commands_with_progress,
    params={"command_name": "base_velocity"}
)
```

---

## 9. How We Did It for Chitrak — Step by Step

### Step 1: Robot asset config (`chitrak_asset.py`)

```python
# Source of each value:
CHITRAK_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="...chitrak.usd",          # generated by UrdfConverter
        activate_contact_sensors=True,       # needed for feet_air_time reward
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.25),               # guessed — tune visually
        joint_pos={
            # Right legs: pitch 0→3.14, knee -3.14→0
            "fr_hip_pitch_joint": 0.5,
            "br_hip_pitch_joint": 0.5,
            "fr_knee_joint": -1.0,
            "br_knee_joint": -1.0,
            # Left legs: MIRRORED limits (read from URDF)
            "fl_hip_pitch_joint": -0.5,
            "bl_hip_pitch_joint": -0.5,
            "fl_knee_joint": 1.0,
            "bl_knee_joint": 1.0,
        },
    ),
    actuators={
        "legs": DCMotorCfg(
            effort_limit=2.5,    # from URDF: <limit effort="2.5"/>
            velocity_limit=8.0,  # from URDF: <limit velocity="8.0"/>
            stiffness=25.0,      # copied from Go1 (unitree.py) — TUNE THIS
            damping=0.5,         # copied from Go1 — TUNE THIS
        )
    }
)
```

### Step 2: URDF → USD conversion

```python
# IsaacLab/source/chitrak_integration/convert_urdf_to_usd.py
cfg = UrdfConverterCfg(
    asset_path="chitrak_fixed.urdf",
    fix_base=False,           # free-floating (walking robot)
    merge_fixed_joints=True,  # torso_joint is fixed → merge into base
    make_instanceable=True,   # needed for multi-env vectorisation
)
UrdfConverter(cfg)
# Output: usd_output/chitrak.usd + configuration/*.usd
```

### Step 3: Env config subclasses Go1

```python
# chitrak_rough_env_cfg.py
class ChitrakFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = CHITRAK_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # flat terrain — no height scanner needed
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None
        # Chitrak-specific
        self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"
        self.actions.joint_pos.scale = 0.25   # from Go1 flat config
```

---

## 10. Training & Iteration Loop

### Train
```bash
export PYTHONPATH=/workspace/isaaclab/source/isaaclab:/workspace/isaaclab/source/chitrak_integration:$PYTHONPATH

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-Chitrak-v0 \
  --num_envs 4096 \
  --headless
```

Logs go to: `logs/rsl_rl/Isaac-Velocity-Flat-Chitrak-v0/`

### Watch training (tensorboard)
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

1. Train for ~1000 iterations
2. Check tensorboard: is `track_lin_vel` reward increasing?
3. If robot falls immediately → reduce stiffness or fix init position
4. If robot doesn't walk → increase `feet_air_time` weight
5. If robot shakes → increase `damping`, increase `action_rate_l2` penalty
6. If robot drags feet → increase `feet_slide` penalty
7. Adjust, re-train from checkpoint

---

## 11. Where Everything Lives in the Isaac Lab Repo

```
IsaacLab/source/
├── isaaclab/isaaclab/
│   ├── envs/mdp/
│   │   ├── rewards.py          ← all built-in reward functions
│   │   ├── observations.py     ← all built-in observation functions
│   │   ├── events.py           ← all domain randomisation functions
│   │   └── terminations.py     ← all termination functions
│   ├── assets/
│   │   └── articulation/       ← ArticulationCfg, robot state data
│   ├── sensors/
│   │   ├── contact_sensor.py   ← ContactSensorCfg
│   │   ├── ray_caster/         ← RayCasterCfg (height scanner)
│   │   └── imu.py              ← ImuCfg
│   └── sim/converters/
│       └── urdf_converter.py   ← UrdfConverter (URDF→USD)
│
├── isaaclab_assets/isaaclab_assets/robots/
│   ├── unitree.py              ← Go1/A1/H1 configs (reference for Chitrak)
│   └── anymal.py               ← Anymal configs
│
└── isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/
    ├── velocity_env_cfg.py     ← BASE CLASS we inherit from
    ├── mdp/
    │   ├── rewards.py          ← locomotion-specific rewards (feet_air_time etc)
    │   ├── curriculums.py      ← terrain curriculum
    │   └── terminations.py     ← terrain out-of-bounds termination
    └── config/
        ├── go1/                ← Go1 flat + rough configs (copy pattern for Chitrak)
        └── anymal_b/           ← another reference
```

---

## 12. Quick Reference — Common Patterns

### Add a reward
```python
self.rewards.my_reward = RewTerm(func=mdp.some_function, weight=0.5, params={...})
```

### Remove a reward
```python
self.rewards.undesired_contacts = None
```

### Add an observation
```python
self.observations.policy.foot_contact = ObsTerm(func=my_fn, params={...})
```

### Remove an observation
```python
self.observations.policy.height_scan = None
```

### Restrict to specific joints/bodies
```python
SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint"])  # regex
SceneEntityCfg("robot", body_names=["torso_link", ".*_thigh_link"])
```

### Change num envs
```python
self.scene.num_envs = 2048
```

### Change episode length
```python
self.episode_length_s = 15.0
```

### Change simulation frequency
```python
self.sim.dt = 1/200       # 200 Hz physics
self.decimation = 4        # policy runs at 200/4 = 50 Hz
```
