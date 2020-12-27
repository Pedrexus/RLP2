import random
import numpy as np
from collections import defaultdict

from .abc import Agent, Actions


class Sarsa(Agent):
    """problem specification
        - action-value function: v(t = 0) = 0 (ok)
        - select action randomly when draw
        - step-size/exploration = Monte Carlo Control
        - same number of episodes as Q-learning
        - lambda in {0, .2, ..., 1}
        """

    online = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.VFA:
            self.eligibility = defaultdict(lambda: np.zeros_like(self.w))
        else:
            self.eligibility = defaultdict(lambda: 0)
        self.lambd = .2
        self.action = None
    
    def reset(self):
        super().reset()
        if self.VFA:
            self.eligibility = defaultdict(lambda: np.zeros_like(self.w))
        else:
            self.eligibility = defaultdict(lambda: 0)
        self.action = None
    
    def seed(self, seed):
        super().seed(seed)
        random.seed(seed)

    def act(self):
        """eps-greedy policy"""
        if self.action == None:
            if random.uniform(0, 1) < self.eps(self.current_state):
                self.action = Actions.sample()
            else:
                # greedy action
                self.action = self.greedy_action(self.current_state)

        self.actions.append(self.action)

        self.count()
        return self.action

    def evaluate(self):
        t = self.step - 2
        S, A, R = self.states, self.actions, self.rewards

        # Choose A' from S' using policy derived from Q(e.g. eps-greedy)
        if random.uniform(0, 1) < self.eps(S[t + 1]):
            action_prime = Actions.sample()
        else:
            action_prime = self.greedy_action(S[t + 1])
            
        # delta <- R + gamma Q(S', A') - Q(S, A)
        if self.VFA:
            delta = R[t + 1] + self.gamma * self.q_hat(S[t + 1], action_prime) - self.q_hat(S[t], A[t])
        else:
            delta = R[t + 1] + self.gamma * self.value[S[t + 1], action_prime] - self.value[S[t], A[t]]

        # E(S, A) <- E(S, A) + 1
        if self.VFA:
            features = np.array([self.x(S[t], A[t])])
            self.eligibility[S[t], A[t]][features] += 1
        else:
            self.eligibility[S[t], A[t]] += 1

        # for all s in S, a in A(s)
        # iterating through self.eligibility keys saves time
        # since its keys have self.eligibility > 0
        for s, a in self.eligibility.keys():
            # Q(s, a) <- Q(s, a) + alpha * delta * E(s, a)
            if self.VFA:
                self.w += self.static_alpha * delta * self.eligibility[s, a]
            else:
                self.value[s, a] += self.alpha(s, a) * delta * self.eligibility[s, a]
            # E(s, a) <- gamma * lambda * E(s, a)
            self.eligibility[s, a] = self.gamma * self.lambd * self.eligibility[s, a]

        # A <- A'
        self.action = action_prime