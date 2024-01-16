import gymnasium as gym
import numpy as np

class RobotWormEnv(gym.Env):
  
    def __init__(self, args):
        super(RobotWormEnv, self).__init__()

        # Constants for the environment
        self.max_torque = 1.0
        self.max_angle = np.pi  # Max angle for joint
        self.time_limit = 1000  # Time limit for each episode
        self.current_step = 0

        # Define action space (torque on two joints)
        self.action_space = gym.spaces.Box(
            low=-self.max_torque, 
            high=self.max_torque, 
            shape=(2,), 
            dtype=np.float32
        )

        # Define observation space
        # Observations: angles of joints, position (x, y), velocity (x, y), and time step
        self.observation_space = gym.spaces.Box(
            low=np.array([-self.max_angle, -self.max_angle, -np.inf, -np.inf, -np.inf, -np.inf, 0]),
            high=np.array([self.max_angle, self.max_angle, np.inf, np.inf, np.inf, np.inf, self.time_limit]),
            dtype=np.float32
        )

        # Initialize state
        self.state = None

    def _perform_transition(self, action):
        # Update the state based on the action
        # This is a simplified example; you should implement the actual dynamics of your robot worm
        joint_angles = self.state[:2]
        position = self.state[2:4]
        velocity = self.state[4:6]

        # Update joint angles and position based on the action (torque)
        # In a real scenario, this would involve a physics simulation
        new_joint_angles = joint_angles + action * 0.1  # Example update
        new_position = position + velocity * 0.1  # Example update
        new_velocity = velocity + action * 0.1  # Example update
        new_time = self.current_step + 1

        self.state = np.array([*new_joint_angles, *new_position, *new_velocity, new_time])
        return self.state

    def _get_reward(self, action, observation, terminated):
        # Define reward function
        # Example: distance moved in the x-direction
        reward = observation[2] - self.state[2]
        return reward

    def _is_terminated(self):
        # Check if the episode should end
        return self.current_step >= self.time_limit

    def _get_info(self):
        # Provide additional info if necessary (optional)
        return {}

    def step(self, action):
        # Perform a step in the environment
        self.current_step += 1
        observation = self._perform_transition(action)
        terminated = self._is_terminated()
        reward = self._get_reward(action, observation, terminated)
        info = self._get_info()

        return observation, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        # Reset the environment state
        super().reset(seed=seed)
        self.current_step = 0
        self.state = np.zeros(7)  # Reset joint angles, position, velocity, and time step
        return self.state, self._get_info()