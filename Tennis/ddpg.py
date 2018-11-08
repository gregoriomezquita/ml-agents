"""
Deep Deterministic Policy Gradient agent
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim


import numpy as np
import random
import copy
from collections import deque
from datetime import datetime

from model import Actor, Critic
from noise import NoNoise, NormalActionNoise, OUNoise

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDPG:
  def __init__(self, actor_state_size, actor_action_size, critic_state_size, critic_action_size, **kwargs):
    
    if 'filename' in kwargs.keys(): 
      data= torch.load(kwargs['filename'])
      self.config= data["config"]
      self.scores= data["scores"]
    elif 'config' in kwargs.keys():
      self.config= kwargs['config']
      data= {}
      self.scores= []
    else:
      raise OSError('DDPG: no configuration parameter in class init')
      
        
    self.actor_state_size = actor_state_size
    self.actor_action_size = actor_action_size
    self.critic_state_size = critic_state_size
    self.critic_action_size = critic_action_size
    memory_size = self.config.get("memory_size", 100000)
    actor_lr = self.config.get("actor_lr", 1e-3)
    critic_lr = self.config.get("critic_lr", 1e-3)
    self.batch_size = self.config.get("batch_size", 256)
    self.discount = self.config.get("discount", 0.9)
    sigma = self.config.get("sigma", 0.2)
    self.tau= self.config.get("tau", 0.001)
    self.seed = self.config.get("seed", 0)
    self.action_noise= self.config.get("action_noise", "No")
    self.critic_l2_reg= self.config.get("critic_l2_reg", 0.0)
    random.seed(self.seed)
    torch.manual_seed(self.seed)
    
    param_noise= False
    if self.action_noise== "Param": param_noise= True
    
    self.actor = Actor(actor_state_size, actor_action_size, nodes= self.config["actor_nodes"], seed= self.seed, param_noise= param_noise).to(device)
    self.critic = Critic(critic_state_size, critic_action_size, nodes= self.config["critic_nodes"], seed= self.seed).to(device)
    self.targetActor = Actor(actor_state_size, actor_action_size, nodes= self.config["actor_nodes"], seed= self.seed, param_noise= param_noise).to(device)
    self.targetCritic = Critic(critic_state_size, critic_action_size, nodes= self.config["critic_nodes"], seed= self.seed).to(device)
    # Initialize parameters
    self.hard_update(self.actor, self.targetActor)
    self.hard_update(self.critic, self.targetCritic)
        
    self.actor_optimizer = optim.Adam(self.actor.parameters(), lr= actor_lr)
    self.critic_optimizer = optim.Adam(self.critic.parameters(), lr= critic_lr, weight_decay= self.critic_l2_reg)
    self.criticLoss = nn.MSELoss()  #nn.SmoothL1Loss()
    #self.criticLoss = nn.SmoothL1Loss()
    
    #self.noise= None
    self.noise = NoNoise()
    if self.action_noise== "OU":
      self.noise = OUNoise(np.zeros(actor_action_size), sigma= sigma)
    elif self.action_noise== "No":
      self.noise = NoNoise()
    elif self.action_noise== "Normal":
      self.noise = NormalActionNoise(np.zeros(actor_action_size), sigma= sigma)
      
    self.memory = Memory(memory_size, self.batch_size, self.seed)
    
    
  def hard_update(self, source, target):
      for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
    
  def soft_update(self, local_model, target_model, tau):
    """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
      target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)  
    
  def act(self, state, add_noise= True):
    """Returns actions for given state as per current policy."""
    self.actor.resample()
    #state = torch.from_numpy(state).float().to(device)
    #state= torch.FloatTensor(state).view(1, -1).to(device)
    #state= torch.FloatTensor(state).unsqueeze(0).to(device)
    state= torch.FloatTensor(state).to(device)
    if len(state.size())== 1:
      state= state.unsqueeze(0)
    
    self.actor.eval()
    with torch.no_grad():
      action = self.actor(state).cpu().data.numpy()
    self.actor.train()
    
    if add_noise and self.noise:
        action += self.noise()
    return np.clip(action, -1, 1)
    
  def step(self, state, action, reward, next_state, done):
    """Save experience in replay memory, and use random sample from buffer to learn."""
    self.memory.add((state, action, reward, next_state, done))
    if len(self.memory) >= self.batch_size:
      self.learn()
      
  def learn_critic(self, states, actions, rewards, next_states, dones, actions_next):
    # ---------------------------- update critic ---------------------------- #
    # Get predicted next-state actions and Q values from target models
    #actions_next = self.targetActor(next_states)
    Q_targets_next = self.targetCritic(next_states, actions_next)
    # Compute Q targets for current states (y_i)
    Q_targets = rewards + (self.discount * Q_targets_next * (1 - dones))
    Q_targets = Variable(Q_targets.data, requires_grad=False)
    # Compute critic loss
    Q_expected = self.critic(states, actions)
    critic_loss = self.criticLoss(Q_expected, Q_targets)
    # Minimize the loss
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()
    # ----------------------- update target networks ----------------------- #
    self.soft_update(self.critic, self.targetCritic, self.tau)
    
  #def learn_actor(self, states, actions, rewards, next_states, dones, actions_pred):
  def learn_actor(self, states, actions_pred):
    # ---------------------------- update actor ---------------------------- #
    # Compute actor loss
    #actions_pred = self.actor(states)
    actor_loss = -self.critic(states, actions_pred).mean()
    # Minimize the loss
    self.actor_optimizer.zero_grad()
    actor_loss.backward()
    self.actor_optimizer.step()
    # ----------------------- update target networks ----------------------- #
    self.soft_update(self.actor, self.targetActor, self.tau) 
      
  def learn(self):
    states, actions, rewards, next_states, dones = self.memory.sample()
    
    self.learn_critic(states, actions, rewards, next_states, dones, self.targetActor(next_states))
    self.learn_actor( states, self.actor(states))
    
  def reset(self):
    self.noise.reset()
      
  def update(self, score= None):
    if score: self.scores.append(score)
      
  def save(self, filename= None):
    data= {"config": self.config, "actor": self.actor.state_dict(), "scores": self.scores,}
    if not filename:
      filename= self.__class__.__name__+ '_'+ datetime.now().strftime("%Y-%m-%d_%H:%M:%S")+ '.data'
    torch.save(data, filename)
    torch.save(self.actor.state_dict(), "last_actor.pth")
    
class Memory:
    def __init__(self, buffer_size, batch_size, seed):
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.seed = random.seed(seed)
    # Add experience tuple or list
    def add(self, experience):
        self.memory.append(experience)
    # Return random sample of experience torch tensors
    def sample(self):
        experiences = random.sample(self.memory, k= self.batch_size)
        num_vars= np.shape(experiences)[1]
        res= (torch.from_numpy(np.vstack([e[i] for e in experiences])).float().to(device) for i in range(num_vars)) 
        return res
    # Get last 'size' experiences torch tensors
    def get(self, size):
        experiences = list(self.memory)[-size:]
        num_vars= np.shape(experiences)[1]
        res= (torch.from_numpy(np.vstack([e[i] for e in experiences])).float().to(device) for i in range(num_vars)) 
        return res
    # Return the current size of internal memory.    
    def __len__(self):
        return len(self.memory)
        
