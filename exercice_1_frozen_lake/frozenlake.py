import gym
import time

env = gym.make('FrozenLake-v0')
env.render()
for i in range(0,1000):
     env.reset()
     for i in range (0,10):
        action = env.action_space.sample()
        env.step(action)
     env.render()
     time.sleep(0.5)