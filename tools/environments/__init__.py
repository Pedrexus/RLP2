import gym
from gym.envs.registration import register

from .cartpole import RewardInversionCartPoleEnv, FrictionCartPoleEnv

spec = gym.spec("CartPole-v1")
kwargs_v1 = {
    "max_episode_steps": spec.max_episode_steps,
    "reward_threshold": spec.reward_threshold
}

register(
    id='CartPole-reward-v1',
    entry_point=RewardInversionCartPoleEnv,
    **kwargs_v1
)

register(
    id='CartPole-friction-v1',
    entry_point=FrictionCartPoleEnv,
    **kwargs_v1
)
