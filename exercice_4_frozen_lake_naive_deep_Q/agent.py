from neural_net import NeuralNet
class Agent():
    def __init__(self, lr, gamma, n_actions, n_states, eps_start, eps_end, eps_dec):
        self.lr = lr
        self.gamma = gamma
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = eps_start
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.neural_net = NeuralNet()

    
    def choose_action(self, reward, obs, action):
        data = [reward, obs, action]
        optimal_action = self.neural_net.forward(data)
        random_action = np.random.choice(3,1)[0]
        next_action = 0
        if np.random.random() < self.epsilon:
            next_action = random_action
        else:
            next_action = optimal_action
        if self.epsilon > self.epsilon_min :
            self.epsilon = self.epsilon*0.9999995
        return next_action
         
    
    def learn(self, reward, obs, action):
        self.neural_net.learn(data, labels)
    
    
    def decrease_epsilon(self):
        self.epsilon = self.epsilon*self.eps_dec if self.epsilon>self.eps_min\
                       else self.eps_min