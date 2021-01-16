
"""
from collections.abc import Iterable

import gym
from stable_baselines3 import DQN

env = gym.make('Skiing-v0')
# action space = Discrete(3)
# observation = np.array, [a, b], bool, {'ale.lives': int}

# model = DQN('MlpPolicy', env, verbose=1)
# model.learn(total_timesteps=25_000)
# model.save("dqn_skiing")
# del model # remove to demonstrate saving and loading

model = DQN.load("dqn_skiing")

# save, load model

obs = env.reset()
for _ in range(25_000):
    # action = env.action_space.sample()
    action, *_states = model.predict(obs)
    print(action, _states)
    if isinstance(action, Iterable):
        action = action[-1]
    obs, rewards, done, info = env.step(action)
    env.render()

    if done:
        obs = env.reset()
"""

import gym
import torch as th
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
)

env = gym.make('BreakoutNoFrameskip-v4')
# model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
# model.learn(5000)

# model.save("ppo_skiing")
# del model # remove to demonstrate saving and loading
model = PPO.load("ppo_skiing")


obs = env.reset()
for i in range(25000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

env.close()
