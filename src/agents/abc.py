import random
from abc import ABC, abstractmethod
from collections import defaultdict

from enum import Enum
from numpy import mean, std, argmax


class Actions:
    """Open AI gym CartPole v1 action space"""
    left = 0
    right = 1

    @classmethod
    def sample(cls) -> int:
        return random.choice([cls.left, cls.right])


class Agent(ABC):
    """Base agent for Open AI gym CartPole V1"""

    action_space = Actions()

    # online learning agent:
    # online = True => evaluation is performed after each step
    # online = False => evaluation is performed after each episode
    online = False

    def __init__(self, initial_value=0, alpha=.15, eps=.1, granularity=1, gamma=1):
        self.episodes = defaultdict(lambda: dict(states=[], actions=[], rewards=[]))
        self.trial = 0

        # the reward expected from taking action A when in state S
        self.value = defaultdict(lambda: initial_value)

        # hyperparameters
        self.alpha = alpha
        # self.gamma = gamma
        # self.eps = eps
        self.granularity = granularity

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
        return len(self.states)

    @property
    def current_state(self):
        try:
            return self.states[-1]
        except IndexError:
            pass

    @property
    def last_action(self):
        return self.actions[-1]

    def optimality(self):
        """Solved Requirements:

        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
        """
        window = 100
        victory = 195

        # sliding window mean
        lengths = [len(ep["rewards"]) for i, ep in self.episodes.items() if i > self.trial - window]

        avg, sigma = int(mean(lengths)), round(std(lengths), 1)

        if avg >= victory:
            print("VICTORY!")

        return avg, sigma

    def greedy_action(self, state):
        return argmax([
                self.value[state, Actions.left],
                self.value[state, Actions.right],
            ])

    def act(self):
        """the agent computes the action based on the current state"""

        # first action in episode, no 'state' has been seen yet
        if self.current_state is None or random.uniform(0, 1) < self.eps(self.step - 1):
            action = Actions.sample()
        else:
            # greedy action
            action = self.greedy_action(self.current_state)

        self.actions.append(action)

        if self.online and self.step > 1:
            self.evaluate()
        return action

    def observe(self, state, reward):
        """the agent observes the environment,
        storing state and reward

        cart-pole observation:
            [Cart Position, Cart Velocity, Pole Angle, Pole Angular Velocity]
        """
        state = tuple(round(x, self.granularity) for x in state)

        # state(T) is being observed as well
        self.states.append(state)
        self.rewards.append(reward)

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
