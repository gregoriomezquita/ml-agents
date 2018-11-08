"""
Actor-Critic definitions.
"""

import torch 
import torch.nn as nn
import numpy as np
from noisy import NoisyLinear

"""
    Actor
"""
class Actor(nn.Module):

  def __init__(self, state_size, action_size, nodes= [32, 32], seed= 0.0, param_noise= False):
    super(Actor, self).__init__()
    
    self.seed= torch.manual_seed(seed)   #  Already done in agent initialization
    
    if param_noise: 
      FCL= NoisyLinear
    else:
      FCL= nn.Linear
    
    self.model= nn.Sequential(
      nn.BatchNorm1d(state_size),
      FCL(state_size, nodes[0]),
      nn.ReLU(),
      nn.BatchNorm1d(nodes[0]),
      FCL(nodes[0], nodes[1]),
      nn.ReLU(),
      nn.BatchNorm1d(nodes[1]),
      FCL(nodes[1], action_size),
      nn.Tanh()
    ) 
    
    self.model.apply(self.init_weights)
    
  def forward(self, state):  
    return self.model(state)

  def init_weights(self, m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.1)
        
  def resample(self):
    for i in range(len(self.model)):
      if 'NoisyLinear' == self.model[i].__class__.__name__:
        self.model[i].resample()
    
"""
    Critic
"""
class Critic(nn.Module):
    
  def __init__(self, state_size, action_size, nodes= [64, 64], seed= 0.0):
    super(Critic, self).__init__()
    
    self.seed= torch.manual_seed(seed)    #  Already done in agent initialization
    
    self.model_input= nn.Sequential(
      nn.Linear(state_size, nodes[0]),
      nn.ReLU(),
      nn.BatchNorm1d(nodes[0]),
      
    ) 
    self.model_output= nn.Sequential(
      nn.Linear(nodes[0] + action_size, nodes[1]),
      nn.ReLU(),
      nn.Linear(nodes[1], 1),
    ) 
    
    self.model_input.apply(self.init_weights)
    self.model_output.apply(self.init_weights)
    
  def forward(self, state, action):  
    i= torch.cat([self.model_input(state), action], dim=1)
    return self.model_output(i)
    
  def init_weights(self, m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.1)
     
