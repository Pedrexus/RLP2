import gym

from agents.montecarlo import MonteCarloControl

env = gym.make("CartPole-v1")
agent = MonteCarloControl()


observation = env.reset()
for _ in range(100):
  env.render()
  action = agent.act()
  observation, reward, done, info = env.step(action)
  agent.evaluate(observation, reward)

  if done:
    observation = env.reset()
    agent.reset()

env.close()