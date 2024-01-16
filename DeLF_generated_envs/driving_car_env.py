import gym
import numpy as np

class SelfDrivingCarEnv(gym.Env):
  
    def __init__(self, max_speed, speed_limit, destination_distance, obstacle_detection_distance):
        super(SelfDrivingCarEnv, self).__init__()

        # Define constants
        self.max_speed = max_speed
        self.speed_limit = speed_limit
        self.destination_distance = destination_distance
        self.obstacle_detection_distance = obstacle_detection_distance

        # Define observation space
        self.observation_space = gym.spaces.Box(
            low=np.array([0, -1, 0, 0, 0]), 
            high=np.array([obstacle_detection_distance, 1, max_speed, 1, destination_distance]), 
            dtype=np.float32
        )

        # Define action space
        # Action 0: Acceleration (0 to 1)
        # Action 1: Braking (0 to 1)
        # Action 2: Lane Change (-1 to 1)
        self.action_space = gym.spaces.Box(
            low=np.array([0, 0, -1]), 
            high=np.array([1, 1, 1]), 
            dtype=np.float32
        )

        # Initialize state variables
        self.position = 0
        self.velocity = 0
        self.lane_position = 0
        self.obstacle_position = None
        self.generate_obstacle()

    def generate_obstacle(self):
        self.obstacle_position = np.random.uniform(self.position, self.position + self.obstacle_detection_distance)

    def _perform_transition(self, action):
        acceleration, braking, lane_change = action
        # Ensure that acceleration and braking are not applied simultaneously
        effective_acceleration = acceleration - braking
        self.velocity = max(0, min(self.max_speed, self.velocity + effective_acceleration))
        self.position += self.velocity
        self.lane_position = (self.lane_position + lane_change) % 2

        observation = np.array([
            self.obstacle_position - self.position,
            1 if self.lane_position == self.obstacle_position else -1,
            self.velocity,
            self.lane_position,
            self.destination_distance - self.position
        ])
        return observation
    
    def _get_reward(self, action, observation, terminated):
        distance_to_obstacle, _, _, _, distance_to_destination = observation
        if terminated:
            if distance_to_destination <= 0:
                return 100
            else:
                return -100
        else:
            return 1 - 0.5 * (self.velocity > self.speed_limit)
    
    def _is_terminated(self, observation):
        distance_to_obstacle, _, _, _, distance_to_destination = observation
        return distance_to_obstacle <= 0 or distance_to_destination <= 0
    
    def _get_info(self):
        info = {}
        return info

    def step(self, action):
        observation = self._perform_transition(action)
        terminated = self._is_terminated(observation)
        reward = self._get_reward(action, observation, terminated)
        info = self._get_info()

        return observation, reward, terminated, info

    def reset(self, seed=None, options=None):
        #super().reset(seed)
        self.position = 0
        self.velocity = 0
        self.lane_position = 0
        self.generate_obstacle()

        observation = np.array([
            self.obstacle_position,
            0,
            0,
            0,
            self.destination_distance
        ])

        info = {}
        return observation, info