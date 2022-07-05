import gym
import numpy as np
import matplotlib.pyplot as plt
from agent import Agent

env = gym.make('FrozenLake-v0')
scores = []
win_pct = []
agent  = Agent()
for i in range(0,500000):
     obs = env.reset()
     done = 0
     score = 0
     while not done:
        action = agent.choose_action(reward, obs, action)
        obs, reward, done, info = env.step(action)
        agent.learn(reward, obs, action)
        agent.decrease_epsilon()
        score += reward
     print("Game done")
     env.render()
     scores.append(score)

     if i % 10 == 0 :
         average = np.mean(scores[-10:])
         win_pct.append(average)
plt.plot(win_pct)
plt.show()