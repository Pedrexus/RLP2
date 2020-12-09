import random
from collections import defaultdict

from numpy import mean, std, argmax

from .abc import Agent, Actions


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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._policy = {}
        self.returns = defaultdict(list)

    # @property
    # def gamma(self):
    #     """also called alpha_t step-size"""
    #     return 1 / self.N(self.current_state, self.last_action)

    def policy(self, state, action):
        """the probability of selecting action A given the state S"""
        try:
            prob = self._policy[state]
        except KeyError:
            return random.uniform(0, 1)
        else:
            return prob[action]

    def act(self):
        """eps-soft action"""

        # Monte Carlos Exploring Starts
        # the first action is random in the episode, but the rest follows the policy
        if self.current_state is None:  # first action in episode, no 'state' has been seen yet
            return Actions.sample()

        p_left = self.policy(self.current_state, action=Actions.left)
        if random.uniform(0, 1) < p_left:
            action = Actions.left
        else:
            action = Actions.right

        self.actions.append(action)
        return action

    def evaluate(self):
        S, A, R = self.states, self.actions, self.rewards

        G = 0
        for t in range(self.step - 2, -1, -1):
            G = self.gamma * G + R[t + 1]
            if (S[t], A[t]) not in zip(S[:t], A[:t]):  # first-visit
                self.returns[S[t], A[t]].append(G)
                self.value[S[t], A[t]] = mean(self.returns[S[t], A[t]])  # average among all episodes

                greedy_action = self.greedy_action(S[t])

                self._policy[S[t]] = {
                    greedy_action: 1 - self.eps(t) + self.eps(t) / 2,
                    int(not greedy_action): self.eps(t) / 2
                }
