from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256): # __init__ Defines the layers
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim) # output layer

    def forward(self, x):
        x = F.relu(self.fc1(x)) # activation function
        return self.fc2(x)
