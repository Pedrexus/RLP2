from time import time

import gym
from hyperopt import hp

from tools.agents import MonteCarloControl, Q

start = time()
dt = lambda: int(time() - start)

# Reproducibility
RANDOM_SEED = 5  # good in MC

env = gym.make("CartPole-v1")

# lookup space
N0 = hp.uniform('N0', 0, 1)
granularity = [
    hp.randint('cart_position', 5),  # btw 0 and 5
    hp.randint('cart_velocity', 5),
    hp.randint('pole_angle', 12),
    hp.randint('pole_angular_velocity', 12),
]

trials, best = Q.tune(env, space=[N0, granularity], seed=RANDOM_SEED)

N0 = best.pop('N0')
granularity = list(best.values())

result = Q.routine(env, hyparams=(N0, granularity), seed=RANDOM_SEED)

print(f"best result = {result} with N0 = {N0} and granularity = {granularity}")

# MonteCarlo:
# 196 steps in {} episodes
# best = {'N0': 0.1301054873136729, 'cart_position': 0, 'cart_velocity': 1, 'pole_angle': 3, 'pole_angular_velocity': 2}
#
# Q:
# 501 steps in {} episodes
# best = {'N0': 0.45, 'cart_position': 1, 'cart_velocity': 0, 'pole_angle': 3, 'pole_angular_velocity': 6}

