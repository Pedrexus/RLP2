import gym
from gym.envs.registration import register

from .cartpole import CartPoleEnvV2

spec = gym.spec("CartPole-v1")
kwargs = {
    "max_episode_steps": spec.max_episode_steps,
    "reward_threshold": spec.reward_threshold
}

register(
    id='CartPole-v2',
    entry_point=CartPoleEnvV2,
    **kwargs
)
