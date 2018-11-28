import torch
import torch.nn as nn
import torch.nn.functional as F

        
class QNetwork(nn.Module):
  def __init__(self, state_size, action_size, seed, nodes= [128, 256]):
    
    super(QNetwork, self).__init__()
        
    self.seed = torch.manual_seed(seed)
        
    self.model= nn.Sequential(
      nn.Linear(state_size, nodes[0]),
      nn.ReLU(),
      nn.Linear(nodes[0], nodes[1]),
      nn.ReLU(),
      nn.Linear(nodes[1], action_size)
    )

  def forward(self, state):
    return self.model(state)
        
        
class Dueling_QNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Dueling_QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.action_size= action_size
        
        
        self.fc1_adv = nn.Linear(state_size, fc1_units)
        self.fc2_adv = nn.Linear(fc1_units, fc2_units)
        self.fc3_adv = nn.Linear(fc2_units, action_size)
        
        self.fc1_val = nn.Linear(state_size, fc1_units)
        self.fc2_val = nn.Linear(fc1_units, fc2_units)
        self.fc3_val = nn.Linear(fc2_units, 1)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        adv= F.relu(self.fc1_adv(state))
        adv= F.relu(self.fc2_adv(adv))
        adv= self.fc3_adv(adv)
        
        val= F.relu(self.fc1_val(state))
        val= F.relu(self.fc2_val(val))
        val = self.fc3_val(val).expand(state.size(0), self.action_size)
        
        return val + adv - adv.mean(1).unsqueeze(1).expand(state.size(0), self.action_size)
        
