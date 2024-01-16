import gymnasium as gym
import numpy as np
from gym.spaces import Discrete, Box

class KeyLockEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size=8):
        super(KeyLockEnv, self).__init__()

        # Grid size
        self.grid_size = grid_size

        # Observation space: Agent's pos (x, y), Key's pos (x, y), Lock's pos (x, y), Key status, Lock status
        self.observation_space = Box(low=0, high=grid_size, shape=(8,), dtype=np.int32)

        # Action space: 0 = Move North, 1 = Move South, 2 = Move East, 3 = Move West, 4 = Pick Up Key, 5 = Open Lock
        self.action_space = Discrete(6)

        # Initialize state
        self.state = None
        self.reset()

    def _perform_transition(self, action):
        # Update the agent's position based on the action
        if action == 0 and self.state[1] > 0:  # Move North
            self.state[1] -= 1
        elif action == 1 and self.state[1] < self.grid_size - 1:  # Move South
            self.state[1] += 1
        elif action == 2 and self.state[0] < self.grid_size - 1:  # Move East
            self.state[0] += 1
        elif action == 3 and self.state[0] > 0:  # Move West
            self.state[0] -= 1
        elif action == 4:  # Pick Up Key
            if self.state[0] == self.state[2] and self.state[1] == self.state[3]:
                self.state[6] = 1  # Key is picked up
        elif action == 5:  # Open Lock
            if self.state[0] == self.state[4] and self.state[1] == self.state[5] and self.state[6] == 1:
                self.state[7] = 1  # Lock is opened

        return self.state

    def _get_reward(self, action, observation, terminated):
        # Simple reward function
        if terminated:
            return 100  # High reward for completing the task
        elif action == 4 and observation[6] == 1:
            return 10  # Reward for picking up the key
        elif action == 5 and observation[7] == 1:
            return 50  # Reward for opening the lock
        return -1  # Small penalty for each step

    def _is_terminated(self):
        # The episode terminates when the lock is opened
        return self.state[7] == 1

    def _get_info(self):
        # Additional info about the environment
        return {}

    def step(self, action):
        observation = self._perform_transition(action)
        terminated = self._is_terminated()
        reward = self._get_reward(action, observation, terminated)
        info = self._get_info()

        return observation, reward, terminated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Randomly place the agent, key, and lock in the grid
        positions = np.random.choice(self.grid_size * self.grid_size, 3, replace=False)
        self.state = np.array([positions[0] % self.grid_size, positions[0] // self.grid_size,  # Agent's position
                               positions[1] % self.grid_size, positions[1] // self.grid_size,  # Key's position
                               positions[2] % self.grid_size, positions[2] // self.grid_size,  # Lock's position
                               0,  # Key's status (not picked up)
                               0])  # Lock's status (not opened)

        return self.state, self._get_info()