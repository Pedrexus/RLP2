import math

from gym.envs.classic_control import CartPoleEnv


class CartPoleEnvHard(CartPoleEnv):
    """cart pole env with smaller limits"""

    def __init__(self):
        super().__init__()

        # Angle at which to fail the episode
        self.theta_threshold_radians = math.radians(9)
        self.x_threshold = 1.9
