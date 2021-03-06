from q_network import Qnetwork
import numpy as np
import torch as T

# This is the class we will import in the environement to let it learn
#Arguments : 
class Agent():
    def __init__(self, lr, n_actions, input_dims, gamma=0.99, epsilon=1.0, eps_min=0.01, eps_dec=1e-5):
        self.lr = lr
        self.gamma = gamma
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.input_dims = input_dims
        self.action_space = [i for i in range(self.n_actions)]

        self.q_network = Qnetwork(self.lr, self.n_actions, self.input_dims)

    # Function that the agent uses to choose the next action
    # Argument : state is the current state of the environement 
    def choose_action(self, state):
        state = T.tensor(state, dtype=T.float).to(self.q_network.device)
        optimal_action = T.argmax(self.q_network.forward(state)).item()
        random_action = np.random.choice(self.action_space)
        if np.random.random() > self.epsilon:
            next_action = optimal_action
        else:
            next_action = random_action
        return next_action
    
    # Decrease epsilon to make the algorithm more greedy
    def decrease_epsilon(self):  
        if self.epsilon > self.eps_min :          
            self.epsilon = self.epsilon - self.eps_dec
        else : 
            self.epsilon =  self.eps_min

    def learn(self, reward, state, next_state, action):
        # Reset the gradients because we will compute them again with backward()
        self.q_network.optimizer.zero_grad()

        # Transform arguments into tensors for PyTorch
        state = T.tensor(state).to(self.q_network.device)
        next_state = T.tensor(next_state).to(self.q_network.device)
        action = T.tensor(action).to(self.q_network.device)
        reward = T.tensor(reward).to(self.q_network.device)
        
        # You need the Q value predicted for the action you just took
        # So you pass the action as the index to the returned tensor from forward()
        q_pred = self.q_network.forward(state)[action]

        q_next = self.q_network.forward(next_state).max()

        # Why don't we subtract q_pred here like in the Bellman equation ?
        # I think it's because we are not incrementing Q but updating the weights of the network to predict it
        # The Bellman equation q <- q + alpha*(reward + gamma*q_next -q) is an iterative incrementation equation
        # Here we don't want to increment q but to precit it's maximum value which is teh reward + the max value of q_next 
        q_bootstrap = reward + self.gamma*q_next

        loss = self.q_network.loss(q_bootstrap, q_pred).to(self.q_network.device)
        loss.backward()
        self.q_network.optimizer.step()
        self.decrease_epsilon()

    