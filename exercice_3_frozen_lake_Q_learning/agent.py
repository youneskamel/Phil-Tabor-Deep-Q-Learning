import random
import numpy as np


class Agent :
    # Initialization function of the agent class.
    def __init__(self) :
        # Values of the dictionary are the states s, values of the sub-dictionaries are the actions a
        self.Q = {} 
        for action in range(16):
            self.Q[action] = {0:0, 1:0, 2:0, 3:0}
        # current state
        self.s = 0        
        # learning rate
        self.alpha = 0.001
        # discount rate
        self.gamma = 0.9
        # exploit-explore ratio
        self.epsilon = 1
        self.epsilon_min = 0.01

    # Update function for Q.
    # Arguments : Reward is the reward received from the chosen action
    # s_prime is the new state after the action 
    #probleme icic ?
    def update_Q(self, reward, s_prime, action):
        current_q = self.Q[self.s][action]
        # Choose the highest reward action knowing we're in s_prime
        max_next_q = max(self.Q[s_prime].values())
        self.Q[self.s][action] = current_q + self.alpha*(reward + self.gamma*max_next_q - current_q)

    # Function used to update the state in the agent class
    def update_state(self, s_prime):
        self.s = s_prime

    # Function used by the agent to choose next action
    def action(self) :
        # define an optimal action and a random action
        optimal_action = max(self.Q[self.s], key=self.Q[self.s].get)
        random_action = np.random.choice(3,1)[0]
        # choose between the optimal and random actions, weighed by epsilon
        action = 0
        if np.random.random() < self.epsilon:
            action = random_action
        else:
            action = optimal_action
        if self.epsilon > self.epsilon_min :
            self.epsilon = self.epsilon*0.9999995
        return action
