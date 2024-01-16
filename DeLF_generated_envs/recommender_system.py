import gym
from gym import spaces
import numpy as np

class RecommenderSystemEnv(gym.Env):
  
    def __init__(self, n_products, n_features):
        super(RecommenderSystemEnv, self).__init__()

        # Environment parameters
        self.n_products = n_products
        self.n_features = n_features

        # Action and observation spaces
        self.action_space = spaces.Discrete(n_products)
        self.observation_space = spaces.Dict({
            'product_features': spaces.MultiDiscrete([n_features] * 4),
            'user_purchase_history': spaces.MultiBinary(n_products),
            'previous_recommendations': spaces.Discrete(n_products),
            'user_feedback': spaces.Discrete(2),  # Binary feedback (satisfied or dissatisfied)
            'time_step': spaces.Discrete(100)  # Assuming a maximum of 100 time steps
        })

        # Initialize environment state variables
        self.user_purchase_history = np.zeros(n_products, dtype=int)
        self.previous_recommendations = np.zeros(n_products, dtype=int)
        self.time_step = 0

    def _perform_transition(self, action):
        # Generate observation based on the action (product recommended)
        observation = {
            'product_features': np.random.randint(0, self.n_features, size=4),
            'user_purchase_history': self.user_purchase_history,
            'previous_recommendations': self.previous_recommendations,
            'user_feedback': np.random.choice([0, 1]),  # Simulating user feedback
            'time_step': self.time_step
        }

        # Update environment state variables
        self.previous_recommendations[action] = 1
        self.time_step += 1

        return observation

    def _get_reward(self, action, observation, terminated):
        # Calculate reward based on user feedback
        reward = observation['user_feedback'] if not terminated else 0
        return reward

    def _is_terminated(self, observation):
        # Terminate episode after a certain number of time steps (e.g., 100)
        return observation['time_step'] >= 99
        #return False

    def _get_info(self):
        # Additional information (if needed)
        return {}

    def step(self, action):
        observation = self._perform_transition(action)
        terminated = self._is_terminated(observation)
        reward = self._get_reward(action, observation, terminated)
        info = self._get_info()

        return observation, reward, terminated, info

    def reset(self, options=None):

        # Reset environment state variables
        self.user_purchase_history = np.zeros(self.n_products, dtype=int)
        self.previous_recommendations = np.zeros(self.n_products, dtype=int)
        self.time_step = 0

        # Initial observation
        observation = {
            'product_features': np.random.randint(0, self.n_features, size=4),
            'user_purchase_history': self.user_purchase_history,
            'previous_recommendations': self.previous_recommendations,
            'user_feedback': np.random.choice([0, 1]),
            'time_step': self.time_step
        }

        return observation, {}
