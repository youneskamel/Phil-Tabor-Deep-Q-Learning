import numpy as np
import matplotlib.pyplot as plt
from preprocessing import build_env

env = build_env("Enduro-v4")

scores = []
win_pct = []
for i in range(0,1000):
     env.reset()
     done = 0
     score = 0
     while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        score += reward
     env.render()
     scores.append(score)

     if i % 10 == 0 :
         average = np.mean(scores[-10:])
         win_pct.append(average)
plt.plot(win_pct)
plt.show()
