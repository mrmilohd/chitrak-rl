import gymnasium as gym
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

class ChitrakEnv(gym.Env):
    """
    Gym environment for Chitrak quadruped robot.
    
    Observation space: 42 dimensional
        - 12 joint positions
        - 12 joint velocities  
        - 3 euler angles (roll, pitch, yaw) from IMU
        - 3 angular velocities (wx, wy, wz) from IMU
        - 12 previous actions
    
    Action space: 12 joint target angles (-1.5 to 1.5 rad)
    """

    def __init__(self):
        # load the MuJoCo model from your scene file
        # this reads the XML, meshes, actuators everything
        self.model = mujoco.MjModel.from_xml_path("scene.xml")
        
        # data holds the current simulation state
        # qpos, qvel, ctrl all live here
        # MuJoCo updates this every mj_step() call
        self.data = mujoco.MjData(self.model)
        
        # store previous action for smoothness penalty in reward
        # and to include in observation vector
        self.prev_action = np.zeros(12)
        
        # how many physics steps per policy step
        # policy runs at 50Hz, physics at 500Hz
        # so we step physics 10 times per policy action
        self.physics_steps_per_control = 10
        
        # action space: 12 target joint angles
        # roughly -1.5 to 1.5 rad covers full joint range
        self.action_space = gym.spaces.Box(
            low=-1.5,
            high=1.5,
            shape=(12,),
            dtype=np.float32
        )
        
        # observation space: 42 dimensional vector
        # no specific bounds so we use -inf to inf
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(42,),
            dtype=np.float32
        )

    def get_obs(self):
        """
        Assemble the 42-dim observation vector from MuJoCo state.
        MuJoCo automatically updates all of this every mj_step().
        """
        
        # joint positions: indices 7-18 of qpos
        # first 7 are freejoint (xyz position + quaternion of torso)
        # after that are the 12 joint angles in radians
        joint_pos = self.data.qpos[7:19].copy()
        
        # joint velocities: indices 6-17 of qvel
        # first 6 are freejoint velocities (linear + angular of torso)
        # after that are the 12 joint angular velocities in rad/s
        joint_vel = self.data.qvel[6:18].copy()
        
        # body orientation as quaternion from freejoint
        # qpos[3:7] = [w, x, y, z] quaternion of torso
        # this is what your IMU measures on the real robot
        quat = self.data.qpos[3:7].copy()
        
        # convert quaternion to euler angles (roll, pitch, yaw)
        # scipy expects [x, y, z, w] order, MuJoCo gives [w, x, y, z]
        # so we reorder before passing to scipy
        euler = Rotation.from_quat(
            [quat[1], quat[2], quat[3], quat[0]]
        ).as_euler('xyz')
        
        # body angular velocity from freejoint
        # qvel[3:6] = [wx, wy, wz] rotational velocity of torso
        # this is what your IMU gyroscope measures on real robot
        ang_vel = self.data.qvel[3:6].copy()
        
        # concatenate everything into one flat 42-dim vector
        obs = np.concatenate([
            joint_pos,        # 12 — how bent each joint is
            joint_vel,        # 12 — how fast each joint is moving
            euler,            # 3  — body tilt (roll pitch yaw)
            ang_vel,          # 3  — body rotation rate
            self.prev_action  # 12 — what we commanded last step
        ]).astype(np.float32)
        
        return obs  # shape (42,)

    def get_reward(self):
        """
        Compute reward from current simulation state.
        Simple reward for flat ground walking:
            + forward velocity (main goal)
            - stability penalty (don't tilt)
            + survival bonus (stay alive)
            - smoothness penalty (don't jerk joints)
        """
        
        # forward velocity of torso in x direction
        # qvel[0] is vx — how fast robot is moving forward
        # this is the main thing we want to maximize
        forward_vel = self.data.qvel[0]
        
        # get current euler angles for stability check
        quat = self.data.qpos[3:7].copy()
        euler = Rotation.from_quat(
            [quat[1], quat[2], quat[3], quat[0]]
        ).as_euler('xyz')
        roll  = euler[0]
        pitch = euler[1]
        
        # stability penalty — penalize tilting
        # squaring means small tilts are ok, large tilts are heavily penalized
        stability_penalty = roll**2 + pitch**2
        
        # survival bonus — small constant reward just for staying alive
        # encourages robot to not fall immediately
        survival_bonus = 1.0
        
        # smoothness penalty — penalize large changes in joint angles
        # stops the robot from making jerky violent movements
        # current_action is set in step() before get_reward() is called
        smoothness_penalty = np.sum(
            (self.current_action - self.prev_action)**2
        )
        
        # combine all terms with weights
        reward = (
            + 2.0 * forward_vel        # main goal: move forward
            - 0.5 * stability_penalty  # don't tilt
            + 1.0 * survival_bonus     # stay alive
            - 0.1 * smoothness_penalty # don't be jerky
        )
        
        return reward

    def is_done(self):
        """
        Check if episode should end (robot has fallen).
        """
        
        # get current body height
        # qpos[2] is z coordinate of torso
        height = self.data.qpos[2]
        
        # get current tilt angles
        quat = self.data.qpos[3:7].copy()
        euler = Rotation.from_quat(
            [quat[1], quat[2], quat[3], quat[0]]
        ).as_euler('xyz')
        roll  = abs(euler[0])
        pitch = abs(euler[1])
        
        # terminate if:
        # robot tilts more than ~40 degrees sideways or forward
        if roll > 0.7 or pitch > 0.7:
            return True
        
        # robot body is too close to ground (collapsed)
        if height < 0.08:
            return True
        
        return False

    def step(self, action):
        """
        Apply action, step physics, return obs + reward + done.
        This is called by PPO every timestep.
        """
        
        # store current action for smoothness penalty
        # and for next timestep's prev_action in obs
        self.current_action = action.copy()
        
        # send target joint angles to MuJoCo actuators
        # data.ctrl is the control input — 12 values, one per actuator
        # MuJoCo's position actuators will apply torque to reach these angles
        self.data.ctrl[:] = action
        
        # step physics multiple times per policy action
        # gives physics time to settle between policy decisions
        for _ in range(self.physics_steps_per_control):
            mujoco.mj_step(self.model, self.data)
        
        # read new state and assemble observation
        obs = self.get_obs()
        
        # compute reward from new state
        reward = self.get_reward()
        
        # check if robot has fallen
        done = self.is_done()
        
        # update prev_action for next timestep
        self.prev_action = action.copy()
        
        # gymnasium requires (obs, reward, terminated, truncated, info)
        return obs, reward, done, False, {}

    def reset(self, seed=None):
        """
        Reset simulation to starting state.
        Called at the start of each episode.
        """
        
        # reset MuJoCo to initial state defined in XML
        # all joints back to 0, robot back to spawn position
        mujoco.mj_resetData(self.model, self.data)
        
        # reset previous action tracker
        self.prev_action = np.zeros(12)
        self.current_action = np.zeros(12)
        
        # return initial observation and empty info dict
        return self.get_obs(), {}

    def render(self):
        # rendering handled separately via MuJoCo viewer
        pass