from time import time

import gym
from tools.agents import MonteCarloControl, Sarsa, Q, DoubleQ

start = time()
dt = lambda: int(time() - start)

env = gym.make("CartPole-v1")

agent = MonteCarloControl(initial_eps=.1)  # 160 - 180
# agent = Q()  # 137 - 156
# agent = DoubleQ()  #

state = env.reset()
agent.observe(state, None)  # S[t = 0]

for _ in range(100_000):
    # render makes the loop 100x slower
    # env.render()

    action = agent.act()
    state, reward, done, _ = env.step(action)

    agent.observe(state, reward)

    if done:
        mean, std = agent.optimality()
        print(f"optimality: {mean} +- {std} (time: {dt()}s)", end="\r", flush=True)

        state = env.reset()

        agent.reset()
        agent.observe(state, None)  # S[t = 0]

mean, std = agent.optimality()
print(f"optimality: {mean} +- {std} in {len(agent.episodes)} episodes (time: {dt()}s)")

env.close()