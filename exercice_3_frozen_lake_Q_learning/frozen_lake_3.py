import gym
import numpy as np
import matplotlib.pyplot as plt
from agent import Agent

env = gym.make('FrozenLake-v0')
scores = []
win_pct_list = []
agent  = Agent()
for i in range(0,500000):
     obs = env.reset()
     done = 0
     score = 0
     while not done:
        action = agent.action()
        obs, reward, done, info = env.step(action)
        agent.update_Q(reward, obs, action)
        agent.update_state(obs)
        score += reward
     print("episode :", i)
     print("epsilon:", agent.epsilon)
     scores.append(score)

     if i % 100 == 0:
            win_pct = np.mean(scores[-100:])
            win_pct_list.append(win_pct)
            if i % 1000 == 0:
                print('episode ', i, 'win pct %.2f' % win_pct,
                      'epsilon %.2f' % agent.epsilon)
plt.plot(win_pct_list)
plt.show()
