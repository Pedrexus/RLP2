import random
from collections import defaultdict

from numpy import mean

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

    # hyperparameters - #TODO: put it in constructor
    granularity = 1
    gamma = .8
    
    def __init__(self):
        self._value = defaultdict(lambda: 0)
        self._returns = defaultdict(lambda: 0)
        self._episodes = defaultdict(lambda: defaultdict(list))

    def act(self):
        # random policy until I implement
        # first-visit MC policy evaluation
        return random.randint(0, 1)

    def observe(self, state, reward):
        """cartpole observation: 
            [Cart Position, Cart Velocity, Pole Angle, Pole Angular Velocity]

        - round the values to 2 decimal places: 
            python.Decimal or round(, 2)
            this reduces the granularity level in the space state - saves memory
        """
        state = tuple(round(x, self.granularity) for x in state)

        # state(T) is being observed as well
        self._episodes[self.trial]["states"].append(state)
        self._episodes[self.trial]["rewards"].append(reward)

    def reset(self):
        self.__evaluate()
        super().reset()

    def __evaluate(self):
        episode = self._episodes[self.trial] 
        states, rewards = episode["states"], episode["rewards"]

        T = len(rewards)
        G = 0
        for t in range(T - 2, -1, -1):
            G = self.gamma * G + rewards[t + 1]
            if states[t] not in states[:t]: # first-visit
                self._returns[states[t]] = G
                self._value[states[t]] = mean(list(self._returns.values()))
