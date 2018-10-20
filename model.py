import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, n_units=64, n_layers=2):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            n_units (int): Number of nodes in hidden layers
            n_layers (int): Number of hidden layers
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.n_units = n_units # Number of nodes in hidden layers
        self.n_layers = n_layers # Number of hidden layers
        
        self.first_fc = nn.Linear(state_size, self.n_units)
        self.middle_fc = nn.Linear(self.n_units, self.n_units)
        self.last_fc = nn.Linear(self.n_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.first_fc(state))
        for i in range(0, self.n_layers-1):
            x = F.relu(self.middle_fc(x))
        return self.last_fc(x)
