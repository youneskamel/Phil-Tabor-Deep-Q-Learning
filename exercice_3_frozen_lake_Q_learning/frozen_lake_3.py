import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')
scores = []
win_pct = []
policy = {0:1, 1:2, 2:1, 3:0, 4:1, 6:1, 8:2, 9:2, 10:1, 13:2, 14:2}
for i in range(0,1000):
     obs = env.reset()
     done = 0
     score = 0
     while not done:
        action = policy[obs]
        obs, reward, done, info = env.step(action)
        score += reward
     print("Game done")
     env.render()
     scores.append(score)

     if i % 10 == 0 :
         average = np.mean(scores[-10:])
         win_pct.append(average)
plt.plot(win_pct)
plt.show()
