import time

from hyperopt import tpe, hp, STATUS_OK, Trials, fmin


class TunerMixin:
    """Hyperparameter tuning"""

    epochs = 10_000

    @classmethod
    def routine(cls, env, seed, hyparams):
        """reinforcement learning routine"""

        # startup agent
        agent = cls(*hyparams, seed=seed)

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

        mean, _ = agent.optimality()
        return mean  # this is what we maximize

    @classmethod
    def tune(cls, env, space, max_evals=100, seed=1):

        def objective(hyparams):
            """the function to be minimized"""
            start = time.time()

            result = cls.routine(env, seed, hyparams)
            perf_time = time.time() - start

            # use relative loss
            loss = - result / perf_time
            return {'loss': loss, 'status': STATUS_OK, 'eval_time': perf_time}

        trials = Trials()

        best = fmin(objective, space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

        return trials, best
