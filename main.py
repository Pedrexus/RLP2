import math
from time import time

import gym
from hyperopt import hp

from tools.agents import Sarsa, Q, MonteCarloControl
import matplotlib.pyplot as plt

# Reproducibility
RANDOM_SEED = 1234

env = gym.make("CartPole-v1")

# lookup space
N0 = hp.uniform('N0', 0, 1)
granularity = [
    0,  # hp.randint('cart_position', 5),  # btw 0 and 5 5
    0,  # hp.randint('cart_velocity', 5),
    hp.randint('pole_angle', 16),
    hp.randint('pole_angular_velocity', 16),
]

# hyperparameter tuning
# trials, best = MonteCarloControl.tune(env, space=[N0, granularity], seed=RANDOM_SEED)

N0 = .5  # best.pop('N0')
granularity = (0, 0, 12, 12)  # [0, 0, *best.values()]

agent = MonteCarloControl.routine(env, hyparams=(N0, granularity), seed=RANDOM_SEED)

print(f"best result = {agent.optimality()[0]} in {len(agent.episodes)} episodes with N0 = {N0} and granularity = {granularity}")

agent.plot_colormesh()
agent.plot_surface()


# MonteCarlo:
# 420 steps in {} episodes
# best = {'N0': 0.1301054873136729, 'cart_position': 0, 'cart_velocity': 1, 'pole_angle': 3, 'pole_angular_velocity': 2}
#
# Q:
# 501 steps in {} episodes
# best = {'N0': 0.45, 'cart_position': 1, 'cart_velocity': 0, 'pole_angle': 3, 'pole_angular_velocity': 6}
# N0 = 0.5 and granularity = [0, 0, 7, 6]

