import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gym.wrappers import FlattenObservation
from gym.spaces import Dict
import time

# Import your custom environment
from GPT_generated_envs.grid_world_env import KeyLockEnv
from GPT_generated_envs.robot_worm_env import RobotWormEnv
from GPT_generated_envs.driving_car_env import SelfDrivingCarEnv
from GPT_generated_envs.recommender_system import RecommenderSystemEnv


env_name_list = {0:"GridWorld", 1: "RobotWorm", 2: "DrivingCar", 3:"RecommenderSystemEnv"}
env_designer_list = {0:"DeLF"}

# Create an instance of your environment
env_name = env_name_list[2]
env_designer = env_designer_list[0]
env = RecommenderSystemEnv(100, 10)


# Check if the observation space is a Dict type
env = FlattenObservation(env)   
print("*****",env.reset()) 
# Optionally, use make_vec_env for parallel environments
env = make_vec_env(lambda: env, n_envs=1)

# Instantiate and train the PPO agent
log_name = f"PPO_{env_designer}_{env_name}_{time.time()}"
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"./{env_designer}_env_logs")
model.learn(total_timesteps=25000, tb_log_name=log_name)

# Save the model
model.save(f"./{env_designer}_env_models/{log_name}")
