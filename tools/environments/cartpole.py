import math

import numpy as np
from gym import logger
from gym.envs.classic_control import CartPoleEnv


class RewardInversionCartPoleEnv(CartPoleEnv):
    """cart pole env with different reward function"""

    def step(self, action):
        state, _, done, info = super().step(action)

        reward = -1 if done else 0
        return state, reward, done, info


class FrictionCartPoleEnv(CartPoleEnv):
    """cart pole env with friction force"""

    def __init__(self):
        super().__init__()
        self.mu_cart = .1  # cart friction coefficient
        self.mu_pole = .1  # pole friction coefficient

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        try:
            x, x_dot, theta, theta_dot, thetaacc = self.state
        except ValueError:
            # after reset(), state is 4-dim vector
            x, x_dot, theta, theta_dot = self.state
            thetaacc = self.np_random.uniform(low=-0.05, high=0.05)

        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        Nc = self.total_mass * self.gravity - self.polemass_length * (thetaacc * sintheta + theta_dot ** 2 * costheta)
        friction = self.mu_cart * np.sign(Nc * x_dot)

        temp = (force + self.polemass_length * theta_dot ** 2 * (sintheta + friction * costheta)) / self.total_mass - self.gravity * friction
        thetaacc = (self.gravity * sintheta - costheta * temp - self.mu_pole * x_dot / self.polemass_length) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass * (costheta - friction)))
        xacc = (force + self.polemass_length * (theta_dot ** 2 * sintheta - thetaacc * costheta) - Nc * friction) / self.total_mass

        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot, thetaacc)

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state[:-1]), reward, done, {}
