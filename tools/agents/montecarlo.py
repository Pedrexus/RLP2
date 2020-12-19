import random
from collections import defaultdict

from numpy import mean, std, argmax

from .abc import Agent, Actions


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
        self.returns = defaultdict(list)

    def evaluate(self):
        S, A, R = self.states, self.actions, self.rewards

        G = 0
        for t in range(self.step - 2, -1, -1):
            G = self.gamma * G + R[t + 1]
            if (S[t], A[t]) not in zip(S[:t], A[:t]):  # first-visit
                self.returns[S[t], A[t]].append(G)
                self.value[S[t], A[t]] = mean(self.returns[S[t], A[t]])  # average among all episodes
                
                # a professora pediu assim, mas dai piora muito...
                # se usar mean(returns) melhora um pouco
                # self.value[S[t], A[t]] += self.alpha(t) * (G - self.value[S[t], A[t]])