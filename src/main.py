from time import time

import gym

from agents.montecarlo import MonteCarloControl

start = time()
dt = lambda: int(time() - start)

env = gym.make("CartPole-v1")
agent = MonteCarloControl()


observation = env.reset()
for _ in range(100_000):
  env.render()

  action = agent.act()
  state, reward, done, _ = env.step(action)

  agent.observe(state, reward)

  if done:    
    mean, std = agent.optimality()
    print(f"optimality: {mean} +- {std} (time: {dt()}s)", end="\r", flush=True)

    observation = env.reset()
    agent.reset()

env.close()