"""
Actor-Critic definitions.
"""

import torch 
import torch.nn as nn
import numpy as np

"""
    Actor
"""
class Actor(nn.Module):

  def __init__(self, state_size, action_size, nodes= [64, 64], seed= 0.0):
    super(Actor, self).__init__()
    
    self.seed= torch.manual_seed(seed)   #  Already done in agent initialization
    
    self.model= nn.Sequential(
      nn.BatchNorm1d(state_size),
      nn.Linear(state_size, nodes[0]),
      #nn.LayerNorm(nodes[0]),
      nn.ReLU(),
      nn.BatchNorm1d(nodes[0]),
      nn.Linear(nodes[0], nodes[1]),
      #nn.LayerNorm(nodes[1]),
      nn.ReLU(),
      nn.BatchNorm1d(nodes[1]),
      nn.Linear(nodes[1], action_size),
      nn.Tanh()
    ) 
    
    self.model.apply(self.init_weights)
    
  def forward(self, state):  
    return self.model(state)

  def init_weights(self, m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.1)

"""
    Critic
"""
class Critic(nn.Module):
    
  def __init__(self, state_size, action_size, nodes= [64, 64], seed= 0.0):
    super(Critic, self).__init__()
    
    self.seed= torch.manual_seed(seed)    #  Already done in agent initialization
    
    self.model_input= nn.Sequential(
      nn.Linear(state_size, nodes[0]),
      #nn.LayerNorm(nodes[0]),
      nn.ReLU(),
      nn.BatchNorm1d(nodes[0]),
      
    ) 
    self.model_output= nn.Sequential(
      nn.Linear(nodes[0] + action_size, nodes[1]),
      #nn.LayerNorm(nodes[1]),
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

