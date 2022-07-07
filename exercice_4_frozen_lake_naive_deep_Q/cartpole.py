import gym
import numpy as np
from agent import Agent
from util import plot_learning_curve


env = gym.make('CartPole-v1')
n_games = 10000
scores = []
eps_history = []

agent = Agent(lr=0.0001, input_dims=env.observation_space.shape,
            n_actions=env.action_space.n)

for i in range(n_games):
   score = 0
   done = False
   obs = env.reset()

   while not done:
      action = agent.choose_action(obs)
      obs_, reward, done, info = env.step(action)
      score += reward
      agent.learn(reward, obs, obs_, action)
      obs = obs_
   scores.append(score)
   eps_history.append(agent.epsilon)

   if i % 100 == 0:
      avg_score = np.mean(scores[-100:])
      print('episode ', i, 'score %.1f avg score %.1f epsilon %.2f' %
            (score, avg_score, agent.epsilon))
filename = 'cartpole_naive_dqn.png'
x = [i+1 for i in range(n_games)]
plot_learning_curve(x, scores, eps_history, filename)
