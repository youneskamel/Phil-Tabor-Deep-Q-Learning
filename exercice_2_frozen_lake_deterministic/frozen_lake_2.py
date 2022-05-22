import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')
scores = []
win_pct = []
for i in range(0,1000):
     env.reset()
     done = 0
     score = 0
     while not done:
        env.step(2)
        obs, reward, done, info = env.step(1)
        score += reward
     print("Game done")
     env.render()
     scores.append(score)

     if i % 10 == 0 :
         average = np.mean(scores[-10:])
         win_pct.append(average)
plt.plot(win_pct)
plt.show()
