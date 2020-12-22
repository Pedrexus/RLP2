import random
from collections import defaultdict

from numpy import argmax

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
            self.w += self.alpha(S[t], A[t]) * (
                    R[t + 1]
                    + self.gamma * self.state_value(S[t + 1])
                    - self.q_hat(S[t], A[t])
            ) * self.x(S[t],  A[t])
        else:
            self.value[S[t], A[t]] += self.alpha(S[t], A[t]) * (
                    R[t + 1]
                    + self.gamma * self.state_value(S[t + 1])
                    - self.value[S[t], A[t]]
            )


# deprecated
class DoubleQ(Q):
    
    def __init__(self, *args, initial_value=0, **kwargs):
        try:
            super().__init__(initial_value, *args, **kwargs)
        except AttributeError:
            pass
        self.initial_value = initial_value
        self.value_list = [defaultdict(lambda: initial_value)] * 2

    @property
    def value(self):
        data = defaultdict(lambda: self.initial_value)
        # unoptimized
        q1, q2 = self.value_list[0], self.value_list[1]
        data.update({k: q1[k] + q2[k] for k in q1.keys()})

        return data
        

    def evaluate(self):
        t = self.step - 2
        S, A, R = self.states, self.actions, self.rewards
        assert len(S) == len(A) + 1 == len(R), f"S: {len(S)} A: {len(A)} R: {len(R)}"

        Q1, Q2 = self.value_list
        if random.random() > .5:
            # update Q[0]
            A1 = argmax([Q1[S[t + 1], Actions.left], Q1[S[t + 1], Actions.right]])
            Q1[S[t], A[t]] += self.alpha(t) * (R[t + 1] + self.gamma * Q2[S[t + 1], A1] - Q1[S[t], A[t]])
        else:
            # update Q[1]
            A2 = argmax([Q2[S[t + 1], Actions.left], Q2[S[t + 1], Actions.right]])
            Q2[S[t], A[t]] += self.alpha(t) * (R[t + 1] + self.gamma * Q1[S[t + 1], A2] - Q2[S[t], A[t]])