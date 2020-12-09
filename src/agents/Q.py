from .abc import Agent


class Q(Agent):
    """problem specification
    - action-value function: v(t = 0) = 0 (ok)
    - select action randomly when draw
    - step-size/exploration = Monte Carlo Control
    """

    online = True

    def evaluate(self):
        t = self.step - 2
        S, A, R = self.states, self.actions, self.rewards  # there is an extra action A_0, prior to S_0
        assert len(S) == len(A) + 1 == len(R), f"S: {len(S)} A: {len(A)} R: {len(R)}"

        self.value[S[t], A[t]] += self.alpha(t) * (
                R[t + 1]
                + self.gamma * self.state_value(S[t + 1])
                - self.value[S[t], A[t]]
        )
