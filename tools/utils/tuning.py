import time

import numpy
from hyperopt import tpe, STATUS_OK, Trials, fmin


class TunerMixin:
    """Hyperparameter tuning"""

    epochs = 20_000

    @classmethod
    def routine(cls, env, hyparams, seed, VFA=False):
        """reinforcement learning routine"""

        # startup agent
        agent = cls(*hyparams, VFA=VFA)

        env.seed(seed)
        agent.seed(seed)

        state = env.reset()
        agent.observe(state, None)  # S[t = 0]

        for epoch in range(cls.epochs):
            action = agent.act()
            state, reward, done, _ = env.step(action)
            agent.observe(state, reward)

            if done:

                # avoid wasting time improving it
                if agent.won():
                    break

                state = env.reset()
                agent.reset()
                agent.observe(state, None)  # S[t = 0]

        env.close()
        return agent

    @classmethod
    def tune(cls, env, space, max_evals=100, seed=1, VFA=False):

        def objective(hyparams):
            """the function to be minimized"""
            start = time.time()

            agent = cls.routine(env, hyparams, seed, VFA)

            avg, _ = agent.optimality()
            return {'loss': - avg, 'status': STATUS_OK, 'eval_time': time.time() - start}

        trials = Trials()

        rstate = numpy.random.RandomState(seed)
        best = fmin(objective, space, algo=tpe.suggest, max_evals=max_evals, trials=trials, rstate=rstate)

        return trials, best
