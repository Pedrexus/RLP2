from time import time

import gym
from hyperopt import hp

from tools.agents import MonteCarloControl

start = time()
dt = lambda: int(time() - start)

# Reproducibility
RANDOM_SEED = 5  # good in MC

env = gym.make("CartPole-v1")
env.seed(RANDOM_SEED)

# lookup space
N0 = hp.uniform('N0', 0, 1)
granularity = [
    hp.randint('cart_position', 5),  # btw 0 and 5
    hp.randint('cart_velocity', 5),
    hp.randint('pole_angle', 12),
    hp.randint('pole_angular_velocity', 12),
]

trials, best = MonteCarloControl.tune(env, space=[N0, granularity], seed=RANDOM_SEED)

N0 = best.pop('N0')
granularity = list(best.values())

result = MonteCarloControl.routine(env, seed=RANDOM_SEED, hyparams=(N0, granularity))

print(f"best result = {result} with N0 = {N0} and granularity = {granularity}")

# MonteCarlo:
# best = {'N0': 0.1301054873136729, 'cart_position': 0, 'cart_velocity': 1, 'pole_angle': 3, 'pole_angular_velocity': 2}

# agent = MonteCarloControl(initial_eps=1, seed=RANDOM_SEED, granularity=(0, 0, 200, 3))  # best granularity = (0, 1, 3, 6)
# # agent = Sarsa(initial_eps=.1, seed=RANDOM_SEED)
# agent = Q(initial_eps=.1, granularity=(0, 0, 20, 10), seed=RANDOM_SEED)
