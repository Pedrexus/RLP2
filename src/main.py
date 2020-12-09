from time import time

import gym
from agents import MonteCarloControl, Sarsa

start = time()
dt = lambda: int(time() - start)

env = gym.make("CartPole-v1")
agent = MonteCarloControl(initial_eps=.3)

observation = env.reset()
for i in range(100_000):
    # this makes everything 100x slower
    # env.render()

    action = agent.act()
    state, reward, done, _ = env.step(action)

    agent.observe(state, reward)

    if done:
        mean, std = agent.optimality()
        print(f"optimality: {mean} +- {std} (time: {dt()}s)", end="\r", flush=True)

        observation = env.reset()
        agent.reset()

mean, std = agent.optimality()
print(f"loop finished in {dt()}s.")
print(f"optimality: {mean} +- {std} in {len(agent.episodes)} episodes")


env.close()
