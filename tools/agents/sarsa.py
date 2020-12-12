import random
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

    def __init__(self, lambd=.2, seed=7, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._policy = {}
        self.eligibility = defaultdict(lambda: 0)
        self.lambd = lambd
        self.action = None

        random.seed(seed)
    
    def reset(self):
        super().reset()
        self.eligibility = defaultdict(lambda: 0)
        self.action = None

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

        # the first action is random in the episode, but the rest follows the policy
        if self.current_state is None:  # first action in episode, no 'state' has been seen yet
            return Actions.sample()

        if self.action == None:
            p_left = self.policy(self.current_state, action=Actions.left)
            if random.uniform(0, 1) < p_left:
                self.action = Actions.left
            else:
                self.action = Actions.right

        self.actions.append(self.action)
        return self.action

    def evaluate(self):
        # Choose A' from S' using policy derived from Q(e.g. eps-greedy)
        p_left = self.policy(self.current_state, action=Actions.left)
        if random.uniform(0, 1) < p_left:
            action_prime = Actions.left
        else:
            action_prime = Actions.right
            
        # delta <- R + gamma Q(S', A') - Q(S, A)
        delta = self.current_reward + self.gamma * self.value[self.current_state, action_prime] - self.value[self.past_state, self.last_action]

        # E(S, A) <- E(S, A) + 1
        self.eligibility[self.past_state, self.last_action] += 1

        # for all s in S, a in A(s)
        # iterating through self.eligibility keys saves time
        # since its keys have self.eligibility > 0
        for s, a in self.eligibility.keys():
            # Q(s, a) <- Q(s, a) + alpha * delta * E(s, a)
            alpha = 1 / self.N(s, a)
            self.value[s, a] += alpha * delta * self.eligibility[s, a]
            # E(s, a) <- gamma * lambda * E(s, a)
            self.eligibility[s, a] = self.gamma * self.lambd * self.eligibility[s, a]

            # update the eps-greedy policy
            greedy_action = self.greedy_action(s)
            eps = self.N0 / (self.N0 + self.N(s))
            self._policy[s] = {
                greedy_action: 1 - eps + eps / 2,
                int(not greedy_action): eps / 2
            }

        # A <- A'
        self.action = action_prime