import gym
import time

env = gym.make("Skiing-v0")

observation = env.reset()
for _ in range(3000):
  env.render()
  time.sleep(.01)
  action = env.action_space.sample()  # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)  # np.array, [a, b], bool, {'ale.lives': int}

  if done:
    observation = env.reset()
env.close()