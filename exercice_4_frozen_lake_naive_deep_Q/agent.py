from q_network import Qnetwork
class Agent():
    def __init__(self, lr, gamma, n_actions, n_states, eps_start, eps_end, eps_dec):
        self.lr = lr
        self.gamma = gamma
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = eps_start
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.neural_net = NeuralNet(0.001,self.n_actions,)

    #roughly right
    def choose_action(self, state):
        state = state #convert to pytorch tensor
        optimal_action = self.neural_net.forward(state)
        random_action = np.random.choice(3,1)[0]
        next_action = 0
        if np.random.random() < self.epsilon:
            next_action = random_action
        else:
            next_action = optimal_action

        return next_action
         
    def decrease_epsilon(self):
        self.epsilon = self.epsilon*self.eps_dec if self.epsilon>self.eps_min\
                       else self.eps_min
    
    def learn(self, reward, obs, action):
        data = [reward, obs, action]
        self.neural_net.learn(data, labels)
    
    