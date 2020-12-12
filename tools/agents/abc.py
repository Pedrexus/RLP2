import random
import math
from abc import ABC, abstractmethod
from collections import defaultdict

from numpy import mean, std, argmax, interp, digitize, linspace

from ..utils import min_max_scale


class Actions:
    """Open AI gym CartPole v1 action space"""
    left = 0
    right = 1

    @classmethod
    def sample(cls) -> int:
        return random.choice([cls.left, cls.right])
    
    @property
    def space(self):
        return [Actions.left, Actions.right]


class Agent(ABC):
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

    def __init__(self, initial_value=0, initial_eps=.1, gamma=.8, granularity=(3, 3, 3, 6), seed=7):
        self.episodes = defaultdict(lambda: dict(states=[], actions=[], rewards=[]))
        self.trial = 0  # == episode

        # hyperparameters
        self.gamma = gamma
        self.N0 = initial_eps

        # determines the number of states in the state space
        # len(state space) = prod(granularity)
        # the lower each value, the faster it converges,
        # but too low may never converge
        self.granularity = granularity
        self.state_space = []
        for gran, lb, ub in zip(self.granularity, self.lower_bounds, self.upper_bounds):
            self.state_space.append(linspace(lb, ub, gran))

         # the reward expected from taking action A when in state S
        self.value = defaultdict(lambda: initial_value)

        random.seed(seed)

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

    def N(self, state, action=None):
        """Number of times a state-action pair was visited in the episode"""

        if action is None:
            return sum(1 for s in self.states if s == state)
        else:
            return sum(1 for (s, a) in zip(self.states, self.actions) if s == state and a == action)

    def eps(self, t):
        return self.N0 / (self.N0 + self.N(self.states[t]))

    def alpha(self, t):
        return 1 / self.N(self.states[t], self.actions[t])

    def discretize(self, state):
        """limits the number of different possible states

        cart-pole observation:
            [Cart Position, Cart Velocity, Pole Angle, Pole Angular Velocity]

        each entry is mapped to an integer
        """
        mew_state = []
        for space, obs in zip(self.state_space, state):
            mew_state.append(int(digitize(obs, space)))

        # # Deixei a implementação aqui caso queira voltar a usa-la
        
        # for obs, lb, ub, grain in zip(state, self.lower_bounds, self.upper_bounds, self.granularity):
        #     scaled = min_max_scale(obs, lb, ub)
        #     new_obs = int(round(scaled * grain))
        #     mew_state.append(new_obs)
        return tuple(mew_state)

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
            # print("VICTORY!!!")
            pass

        return avg, sigma

    def state_value(self, state):
        return max(
            self.value[state, Actions.left],
            self.value[state, Actions.right],
        )

    def greedy_action(self, state):
        if self.value[state, Actions.left] == self.value[state, Actions.right]:
            # if draw, select randomly
            return Actions.sample()
        return argmax([
                self.value[state, Actions.left],
                self.value[state, Actions.right],
            ])

    def act(self):
        """eps-greedy policy"""
        if random.uniform(0, 1) < self.eps(self.step - 1):
            action = Actions.sample()
        else:
            # greedy action
            action = self.greedy_action(self.current_state)

        self.actions.append(action)
        return action

    def observe(self, state, reward):
        """the agent observes the environment, storing state and reward"""
        # terminal state is being observed as well
        self.states.append(self.discretize(state))
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
