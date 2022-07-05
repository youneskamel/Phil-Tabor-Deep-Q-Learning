import neural_net from NeuralNet
class Agent():
    def __init__(self, lr, gamma, n_actions, n_states, eps_start, eps_end, eps_dec):
        self.lr = lr
        self.gamma = gamma
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = eps_start
        self.eps_min = eps_end
        self.eps_dec = eps_dec

    
    def choose_action(self, data):
        optimal_action = neural_net.forward(data)
        random_action = np.random.choice(3,1)[0]
        action = 0
        if np.random.random() < self.epsilon:
            action = random_action
        else:
            action = optimal_action
        if self.epsilon > self.epsilon_min :
            self.epsilon = self.epsilon*0.9999995
        return action
         
    
    def learn(self, data):
        neural_net.learn(data, labels)
    
    
    def decrease_epsilon(self):
        self.epsilon = self.epsilon*self.eps_dec if self.epsilon>self.eps_min\
                       else self.eps_min