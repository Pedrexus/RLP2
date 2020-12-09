from .abc import Agent


class Sarsa(Agent):
    """problem specification
        - action-value function: v(t = 0) = 0 (ok)
        - select action randomly when draw
        - step-size/exploration = Monte Carlo Control
        - same number of episodes as Q-learning
        - lambda in {0, .2, ..., 1}
        """

    online = True

    def evaluate(self):
        t = self.step - 2
        S, A, R = self.states, self.actions[1:], self.rewards  # there is an extra action A_0, prior to S_0
        assert len(S) == len(A) == len(R), f"S: {len(S)} A: {len(A)} R: {len(R)}"

        self.value[S[t], A[t]] += self.alpha(t) * (
                R[t + 1]
                + self.gamma * self.value[S[t + 1], A[t + 1]]
                - self.value[S[t], A[t]]
        )
