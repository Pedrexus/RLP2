import math
import random
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from functools import cached_property
from itertools import product

import matplotlib.pyplot as plt
import numpy as np

from ..utils.tuning import TunerMixin


class Actions:
    """Open AI gym CartPole v1 action space"""
    left = 0
    right = 1

    space = (left, right)

    @classmethod
    def sample(cls) -> int:
        return random.choice([cls.left, cls.right])


class Agent(ABC, TunerMixin):
    """Base agent for Open AI gym CartPole V1"""

    # cart pole specific variables
    # Observation:
    #     Type: Box(4)
    #     Num     Observation               Min                     Max
    #     0       Cart Position             -4.8                    4.8
    #     1       Cart Velocity             -Inf                    Inf
    #     2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
    #     3       Pole Angular Velocity     -Inf                    Inf
    upper_bounds = [+4.8, +0.5, +math.radians(24), math.radians(50)]
    lower_bounds = [-4.8, -0.5, -math.radians(24), -math.radians(50)]

    # online learning agent:
    # online = True => policy evaluation is performed after each step
    # online = False => policy evaluation is performed after each episode
    online = False

    def __init__(self, N0=.1, granularity=(2, 2, 3, 6), initial_value=0, gamma=.8, seed=1):
        random.seed(seed)

        self.episodes = defaultdict(lambda: dict(states=[], actions=[], rewards=[]))
        self.trial = 0  # == episode

        # hyperparameters
        self.gamma = gamma
        self.N0 = N0

        # determines the number of states in the state space
        # len(state space) = prod(granularity)
        # the lower each value, the faster it converges,
        # but too low may never converge
        self.granularity = granularity
        self.state_space = [np.linspace(lb, ub, gran) for lb, ub, gran in zip(self.lower_bounds, self.upper_bounds, self.granularity)]

        # the reward expected from taking action A when in state S
        self.value = defaultdict(lambda: initial_value)
        self.counter = defaultdict(lambda: Counter())

    @property
    def episode(self):
        return self.episodes[self.trial]

    @property
    def states(self):
        return self.episode["states"]

    @property
    def actions(self):
        return self.episode["actions"]

    @property
    def rewards(self):
        return self.episode["rewards"]

    @property
    def step(self):
        """Theoretically, in the last time step, we only collect Reward"""
        return len(self.rewards)

    @property
    def past_state(self):
        try:
            return self.states[-2]
        except IndexError:
            pass

    @property
    def current_state(self):
        try:
            return self.states[-1]
        except IndexError:
            pass

    @property
    def last_action(self):
        return self.actions[-1]

    @property
    def current_reward(self):
        try:
            return self.rewards[-1]
        except IndexError:
            pass

    def count(self):
        self.counter["states"][self.current_state] += 1
        self.counter["state-actions"][self.current_state, self.last_action] += 1

    def N(self, state, action=None):
        """Number of episodes the state-action pair was visted at least once"""
        # if action is None:
        #     return sum(1 for episode in self.episodes.values() if state in episode["states"])
        # else:
        #     return sum(1 for episode in self.episodes.values() if state in episode["states"] and action in episode["actions"])

        """Number of times a state-action pair was visited in the episode"""
        if action is None:
            return self.counter["states"][state]
        else:
            return self.counter["state-actions"][state, action]

    def eps(self, state):
        return self.N0 / (self.N0 + self.N(state))

    def alpha(self, state, action):
        return 1 / self.N(state, action)

    def digitize(self, state):
        """limits the number of different possible states

        cart-pole observation:
            [Cart Position, Cart Velocity, Pole Angle, Pole Angular Velocity]

        each entry is mapped into an integer
        """
        return tuple(np.digitize(obs, space) for space, obs in zip(self.state_space, state))

    window = 100
    victory = 195

    def optimality(self):
        """Solved Requirements:

        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
        """

        # sliding window mean
        lengths = [len(ep["rewards"]) for i, ep in self.episodes.items() if i > self.trial - self.window]

        avg, sigma = np.floor(np.mean(lengths)), round(np.std(lengths), 1)

        return avg, sigma

    def won(self):
        avg, _ = self.optimality()
        return avg > self.victory

    def state_value(self, state):
        return max(
            self.value[state, Actions.left],
            self.value[state, Actions.right],
        )

    def greedy_action(self, state):
        if self.value[state, Actions.left] == self.value[state, Actions.right]:
            # if draw, select randomly
            return Actions.sample()
        return np.argmax([
            self.value[state, Actions.left],
            self.value[state, Actions.right],
        ])

    def act(self):
        """eps-greedy policy"""
        if random.uniform(0, 1) < self.eps(self.current_state):
            action = Actions.sample()
        else:
            # greedy action
            action = self.greedy_action(self.current_state)

        self.actions.append(action)

        self.count()
        return action

    def observe(self, state, reward):
        """the agent observes the environment, storing state and reward"""
        # terminal state is being observed as well
        self.states.append(self.digitize(state))
        self.rewards.append(reward)

        if self.online and self.step > 1:
            self.evaluate()

    def reset(self):
        # S_T was being observed. Now it is removed.
        # This causes an error in self.N(self.current_state, self.last_action) == 0
        # in which current_state = S_T, but last_action = S_T-1
        del self.states[-1]

        if not self.online:
            self.evaluate()

        # this stays at the bottom!
        self.trial += 1

    @abstractmethod
    def evaluate(self):
        """policy evaluation and control

        the agent adapts itself based on previous rewards
        """
        raise NotImplementedError

    @cached_property
    def state_value_array(self):
        shape = [g + 1 for g in self.granularity]
        value = np.zeros(shape)

        state_space_full = [(obs if obs.size > 0 else [0]) for obs in self.state_space]
        for state in product(*state_space_full):
            dig_state = self.digitize(state)
            value[dig_state] = self.state_value(dig_state)

        return value.reshape([x for x in shape if x > 1])

    def plot_colormesh(self, axis=0):
        plt.pcolormesh(self.state_value_array)
        plt.colorbar()
        plt.show()

    def plot_surface(self, axis=0, *args, **kwargs):
        shape = self.state_value_array.shape
        assert len(shape) == 2, f"Unable to make surface of tensor with rank {shape}"

        x, y = np.meshgrid(range(shape[0]), range(shape[1]))
        z = self.state_value_array

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surface = ax.plot_surface(x, y, z, rstride=1, cstride=1, linewidth=0, cmap='viridis', antialiased=False)
        fig.colorbar(surface, shrink=0.5, aspect=5)
        plt.show()


class EpsSoftMixin(ABC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._policy = {}

    def policy(self, state, action):
        """the probability of selecting action A given the state S"""
        try:
            prob = self._policy[state]
        except KeyError:
            return random.uniform(0, 1)
        else:
            return prob[action]

    def act(self):
        """eps-soft policy"""

        if self.current_state is None:  # first action in episode, no 'state' has been seen yet
            return Actions.sample()

        p_left = self.policy(self.current_state, action=Actions.left)

        if random.uniform(0, 1) < p_left:
            action = Actions.left
        else:
            action = Actions.right

        self.actions.append(action)
        return action

    def update_policy(state, greedy_action):
        self._policy[state] = {
            greedy_action: 1 - self.eps(t) + self.eps(t) / 2,
            int(not greedy_action): self.eps(t) / 2
        }
