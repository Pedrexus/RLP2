from gym import logger
from gym.envs.classic_control import CartPoleEnv


class CartPoleEnv_RewardInversion(CartPoleEnv):
    """cart pole env with different reward function"""

    def step(self, action):
        state, _, done, info = super().step(action)

        reward = -1 if done else 0
        return state, reward, done, info
