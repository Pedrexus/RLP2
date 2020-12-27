import math
from time import time

import gym
from hyperopt import hp

from tools.agents import Sarsa, Q, MonteCarloControl
import matplotlib.pyplot as plt

# Reproducibility
RANDOM_SEED = 1234

env = gym.make("CartPole-v1")

# slow: ~3min
N0 = hp.uniform('N0', 0, 1)
granularity = [0, 0, hp.randint('pole_angle', 20), hp.randint('pole_angular_velocity', 20)]

trials, best = Q.tune(env, space=[N0, granularity], seed=RANDOM_SEED)

N0, granularity = best.pop('N0'), [0, 0, *best.values()]
agent = Q.routine(env, hyparams=(N0, granularity), seed=RANDOM_SEED)

print(f"best result = {agent.optimality()[0]} in {len(agent.episodes)} episodes with N0 = {agent.N0:.2f} and granularity = {agent.granularity}")
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

