import random
from collections import defaultdict

from numpy import mean, std, argmax

from .abc import Agent

# this can make it much easier or much harder...
# random.seed(1234)

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

    # hyperparameters - TODO: put it in constructor
    granularity = 1 # => 4 * 2 * 10^(granularity + 1) possible states
    gamma = 1
    eps = .1
    
    def __init__(self):
        self._policy = {}
        self._value = {}
        self._returns = defaultdict(list)
        self._episodes = defaultdict(lambda: dict(states=[], actions=[], rewards=[]))

        self.trial = 0 # == episode

    def policy(self, state, action):
        """the probability of selecting action A given the state S"""
        try:
            prob = self._policy[state]
        except KeyError:
            return random.uniform(0, 1)
        else:
            return prob[action]

    def value(self, state, action):
        """the reward expected from taking action A when in state S"""
        key = (*state, action)
        try:
            return self._value[key]
        except KeyError:
            return 0  # default

    def returns(self, state, action):
        key = (*state, action)
        return self._returns[key]

    @property
    def episode(self):
        return self._episodes[self.trial]

    def optimality(self):
        """Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
        """
        window = 100
        victory = 195

        # sliding window mean
        lengths = [len(ep["states"]) for i, ep in self._episodes.items() if i > self.trial - self.window]

        avg, sigma = int(mean(lengths)), round(std(lengths), 1)

        if avg >= victory:
            print("VICTORY!")

        return avg, sigma

    def act(self):
        # Monte Carlos Exploring Starts
        # the first action is random in the episode, but the rest follows the policy
        left, right = 0, 1
        states = self.episode["states"]

        if not states:  # first action in episode, 'states' is empty
            return random.choice((left, right))

        p_left = self.policy(states[-1], action=left)
        if random.uniform(0, 1) < p_left:
            action = left
        else:
            action = right

        self.episode["actions"].append(action)
        return action

    def observe(self, state, reward):
        """cartpole observation: 
            [Cart Position, Cart Velocity, Pole Angle, Pole Angular Velocity]

        - round the values to 2 decimal places: 
            python.Decimal or round(, 2)
            this reduces the granularity level in the space state - saves memory
        """
        state = tuple(round(x, self.granularity) for x in state)

        # state(T) is being observed as well
        self.episode["states"].append(state)
        self.episode["rewards"].append(reward)

    def reset(self):
        self.evaluate()
        self.trial += 1
        
    def evaluate(self):
        left, right = 0, 1

        actions, states, rewards = self.episode["actions"], self.episode["states"], self.episode["rewards"]

        T = len(rewards)
        G = 0
        for t in range(T - 2, -1, -1):
            G = self.gamma * G + rewards[t + 1]
            if (states[t], actions[t]) not in zip(states[:t], actions[:t]): # first-visit
                pair = (*states[t], actions[t])
                self._returns[pair].append(G)
                self._value[pair] = mean(self.returns(states[t], actions[t])) # average among all episodes

                # this is simple to perform here because there are only two actions
                # and they are valued 0 and 1. A loop would be required otherwise.
                greedy_action = argmax([
                    self.value(states[t], left),
                    self.value(states[t], right),
                ])

                self._policy[states[t]] = {
                    greedy_action: 1 - self.eps + self.eps / 2,
                    int(not greedy_action): self.eps / 2
                }

                # or
                # probabilities = {}
                # for a in (left, right):
                #     if a == greedy_action:
                #         probabilities[a] = 1 - self.eps + self.eps / 2  # len(action space) = 2
                #     else:
                #         probabilities[a] = self.eps / 2
