import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T


# The neural network that replaces our Q-value table
# It takes the state as input data and output the state-action value (Q)
# Arguments : the learning rate, the number of possible actions, the dimmensions of the input
class Qnetwork(nn.Module):
    def __init__(self, lr, n_classes, input_dims):
        super(Qnetwork, self).__init__()

        # Not sure why input dims has a *, maybe unpacking a variadic argument ?
        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, n_classes)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
		# use GPU if available, else CPU
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

	# forward propagation
    def forward(self, state):
        layer1 = F.relu(self.fc1(state))
        layer2 = self.fc2(layer1)

        return layer2

  