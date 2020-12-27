import random
from collections import defaultdict

from numpy import argmax, array

from .abc import Agent, Actions


class Q(Agent):
    """problem specification
    - action-value function: v(t = 0) = 0 (ok)
    - select action randomly when draw
    - step-size/exploration = Monte Carlo Control
    """

    online = True

    def evaluate(self):
        t = self.step - 2
        S, A, R = self.states, self.actions, self.rewards
        # assert len(S) == len(A) + 1 == len(R), f"S: {len(S)} A: {len(A)} R: {len(R)}"

        if self.VFA:
            features = array([self.x(S[t], A[t])])
            self.w[features] += self.static_alpha * (
                    R[t + 1]
                    + self.gamma * self.state_value(S[t + 1])
                    - self.q_hat(S[t], A[t])
            )
        else:
            self.value[S[t], A[t]] += self.alpha(S[t], A[t]) * (
                    R[t + 1]
                    + self.gamma * self.state_value(S[t + 1])
                    - self.value[S[t], A[t]]
            )
