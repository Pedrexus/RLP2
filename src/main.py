import gym

from agents.montecarlo import MonteCarloControl

env = gym.make("CartPole-v1")
agent = MonteCarloControl()


observation = env.reset()
for _ in range(100):
  env.render()

  action = agent.act()
  state, reward, done, _ = env.step(action)

  agent.observe(state, reward)

  if done:
    observation = env.reset()
    agent.reset()

env.close()