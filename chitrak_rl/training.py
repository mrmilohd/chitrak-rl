from chitrak_env import ChitrakEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
import os

# create logs and models directories
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# create 8 parallel environments
# each runs independently, PPO collects experience from all of them
env = make_vec_env(ChitrakEnv, n_envs=8)

# checkpoint callback — saves model every 100k steps
# so if training crashes you don't lose everything
checkpoint_callback = CheckpointCallback(
    save_freq=100_000,
    save_path="./models/",
    name_prefix="chitrak_ppo"
)

# create PPO model
model = PPO(
    "MlpPolicy",       # simple MLP neural network
    env,
    verbose=1,         # print training progress
    tensorboard_log="./logs/",
    learning_rate=3e-4,
    n_steps=2048,      # steps per environment before update
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
)

# train for 10 million steps
model.learn(
    total_timesteps=10_000_000,
    callback=checkpoint_callback
)

# save final model
model.save("models/chitrak_final")
print("training done!")