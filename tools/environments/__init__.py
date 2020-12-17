import gym
from gym.envs.registration import register

from .cartpole import CartPoleEnv_RewardInversion

spec = gym.spec("CartPole-v1")
kwargs_v1 = {
    "max_episode_steps": spec.max_episode_steps,
    "reward_threshold": spec.reward_threshold
}

register(
    id='CartPole-reward-v1',
    entry_point=CartPoleEnv_RewardInversion,
    **kwargs_v1
)
