import random
from collections import defaultdict

from numpy import mean

from .abc import Agent


# this can make it much easier or much harder...
# random.seed(1234)

class MonteCarloControl(Agent):
    """problem specifications:

    - value function: v(t = 0) = 0 (ok)
    - step_size: alpha(t) = 1 / N(s_t, a_t)
    - greedy exploration: eps(t) = N_0 / (N_0 + N(s_t)), N_0 = cte
    - N(s): number of visits of state s (ok)
    - N(s, a): number of times action = a(s) (ok)
    - N_0: fitting hyperparameter
    - plot: V*(s) = max_a Q*(s, a)
    """

    online = False

    def __init__(self, initial_eps=.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._policy = {}
        self.returns = defaultdict(list)

        self.N0 = initial_eps

    def N(self, state, action=None):
        """Number of times a state-action pair was visited in the episode"""

        if action is None:
            return sum(1 for s in self.states if s == state)
        else:
            return sum(1 for (s, a) in zip(self.states, self.actions) if s == state and a == action)

    def gamma(self, t):
        """also called alpha_t step-size"""
        return 1 / self.N(self.states[t], self.actions[t])

    def eps(self, t):
        return self.N0 / (self.N0 + self.N(self.states[t]))

    def policy(self, state, action):
        """the probability of selecting action A given the state S"""
        try:
            prob = self._policy[state]
        except KeyError:
            return random.uniform(0, 1)
        else:
            return prob[action]

    def act(self):
        # Monte Carlos Exploring Starts
        # the first action is random in the episode, but the rest follows the policy
        if self.current_state is None:  # first action in episode, no 'state' has been seen yet
            action = self.action_space.sample()
        else:
            prob_left = self.policy(self.current_state, action=self.action_space.left)
            if random.uniform(0, 1) < prob_left:
                action = self.action_space.left
            else:
                action = self.action_space.right

        self.actions.append(action)
        return action

    def evaluate(self):
        S, A, R = self.states, self.actions, self.rewards

        G = 0
        for t in range(self.step - 2, -1, -1):
            G = self.gamma(t) * G + R[t + 1]
            if (S[t], A[t]) not in zip(S[:t], A[:t]):  # first-visit
                self.returns[S[t], A[t]].append(G)
                self.value[S[t], A[t]] = mean(self.returns[S[t], A[t]])  # average among all episodes

                greedy_action = self.greedy_action(S[t])

                self._policy[S[t]] = {
                    greedy_action: 1 - self.eps(t) + self.eps(t) / 2,
                    int(not greedy_action): self.eps(t) / 2
                }
