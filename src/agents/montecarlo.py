import random
from collections import defaultdict

from .abc import Agent

random.seed(1234)

class MonteCarloControl(Agent):
    """problem specifications:

    - value function: v(t = 0) = 0
    - step_size: alpha(t) = 1 / N(s_t, a_t)
    - greedy exploration: eps(t) = N_0 / (N_0 + N(s_t)), N_0 = cte
    - N(s): number of visits of state s
    - N(s, a): number of times action = a(s)
    - N_0: fitting hyperparameter
    - plot: V*(s) = max_a Q*(s, a)
    """
    granularity = 1
    episodes = defaultdict(dict)

    def act(self):
        return random.randint(0, 1)

    def evaluate(self, state, reward):
        """cartpole observation: 
            [Cart Position, Cart Velocity, Pole Angle, Pole Angular Velocity]

        - round the values to 2 decimal places: 
            python.Decimal or round(, 2)
            this reduces the granularity level in the space state - saves memory
        """
        state = tuple(round(x, self.granularity) for x in state)  # hashable

        self.episodes[self.trial].append({
            "state": hash(state),
            "reward": reward
        })

