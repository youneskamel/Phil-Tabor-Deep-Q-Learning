class Agent :
    # Initialization function of the agent class.
    def __init__(self) :

        # Values of the dictionary are the states s, values of the sub-dictionaries are the actions a
        self.Q = {0: {0:0, 1:0, 2:0, 3:0} ,1: {0:0, 1:0, 2:0, 3:0} ,2: {0:0, 1:0, 2:0, 3:0}, 
        3: {0:0, 1:0, 2:0, 3:0},4: {0:0, 1:0, 2:0, 3:0}, 5: {0:0, 1:0, 2:0, 3:0}, 6: {0:0, 1:0, 2:0, 3:0},
        7:{0:0, 1:0, 2:0, 3:0}, 8:{0:0, 1:0, 2:0, 3:0}, 9:{0:0, 1:0, 2:0, 3:0},10:{0:0, 1:0, 2:0, 3:0}, 11:{0:0, 1:0, 2:0, 3:0},
        12:{0:0, 1:0, 2:0, 3:0}, 13:{0:0, 1:0, 2:0, 3:0}, 14:{0:0, 1:0, 2:0, 3:0}, 15:{0:0, 1:0, 2:0, 3:0}}

        # current state
        self.s = 0        
        # learning rate
        self.alpha = 0.001
        # discount rate
        self.gamma = 0.9
        # exploit-explore ratio
        self.epsilon_max = 1
        self.epsilon_min = 0.01

    # Update function for Q.
    # Arguments : Reward is the reward received from the chosen action
    # s_prime is the new state after the action 
    def update_Q(self, reward, s_prime, action):
        current_q = self.Q[self.s][action]
        # Choose the highest reward actiona knowing we're in s_prime
        max_next_q = max(self.Q[s_prime], key=lambda a: self.Q[s_prime][a])
        current_q += self.alpha*(reward + self.gamma*(max_next_q - current_q))


    # Function used by the agent to choose next action
    def action(self) :
        eps = self.epsilon_min
        optimal_action = max(self.Q[self.s], key=lambda a : self.Q[self.s][a])
        action = eps*optimal_action
        return action