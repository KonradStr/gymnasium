import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
    # __init__ Defines the layers
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim) # output layer

    # x is state (12 information values) -> sending state through layer 1
    def forward(self, x):
        x = F.relu(self.fc1(x)) # activation function?
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# class DQN(nn.Module):
#     # __init__ Defines the layers
#     def __init__(self, state_dim, action_dim, hidden_dim=256):
#         super(DQN, self).__init__()
#         self.fc1 = nn.Linear(state_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, action_dim)  # output layer
#
#     # x is state (12 information values) -> sending state through layer 1
#     def forward(self, x):
#         x = F.relu(self.fc1(x))  # activation function?
#         return self.fc2(x)
