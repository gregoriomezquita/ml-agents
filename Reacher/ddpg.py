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
from collections import namedtuple, deque
from datetime import datetime

from model import Actor, Critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDPG:
  def __init__(self, **kwargs):
    
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
      
        
    self.state_size = self.config["state_size"]
    self.action_size = self.config["action_size"]
    memory_size = self.config["memory_size"]
    actor_lr = self.config["actor_lr"]
    critic_lr = self.config["critic_lr"]
    self.batch_size = self.config["batch_size"]
    self.discount = self.config["discount"]
    sigma = self.config["sigma"]
    self.tau= self.config["tau"]
    self.seed = self.config["seed"]
    random.seed(self.seed)
    torch.manual_seed(self.seed)
    
    self.actor = Actor(self.state_size, self.action_size, nodes= self.config["actor_nodes"], seed= self.seed).to(device)
    if 'actor' in data.keys(): self.actor.load_state_dict(data['actor'])
    self.critic = Critic(self.state_size, self.action_size, nodes= self.config["critic_nodes"], seed= self.seed).to(device)
    self.targetActor = Actor(self.state_size, self.action_size, nodes= self.config["actor_nodes"], seed= self.seed).to(device)
    self.targetCritic = Critic(self.state_size, self.action_size, nodes= self.config["critic_nodes"], seed= self.seed).to(device)
    self.hard_copy(self.targetActor, self.actor)
    self.hard_copy(self.targetCritic, self.critic)
        
    self.actor_optimizer = optim.Adam(self.actor.parameters(), lr= actor_lr)
    self.critic_optimizer = optim.Adam(self.critic.parameters(), lr= critic_lr)
    self.criticLoss = nn.MSELoss()
    self.noise = OUNoise(self.action_size, self.seed, sigma= sigma)
    self.memory = ReplayBuffer(self.action_size, memory_size, self.batch_size, self.seed)
    
  def hard_copy(self, target, source):
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
    
    state= torch.FloatTensor(state).view(1, -1).to(device)
    self.actor.eval()
    with torch.no_grad():
      action = self.actor(state).cpu().data.numpy()
    self.actor.train()
    if add_noise:
        action += self.noise.sample()
    return np.clip(action, -1, 1)
    
  def step(self, state, action, reward, next_state, done):
    """Save experience in replay memory, and use random sample from buffer to learn."""
    self.memory.add(state, action, reward, next_state, done)
    if len(self.memory) >= self.batch_size:
      self.learn()
    
  def learn(self):
    states, actions, rewards, next_states, dones = self.memory.sample()
    
    # ---------------------------- update critic ---------------------------- #
    # Get predicted next-state actions and Q values from target models
    actions_next = self.targetActor(next_states)
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

    # ---------------------------- update actor ---------------------------- #
    # Compute actor loss
    actions_pred = self.actor(states)
    actor_loss = -self.critic(states, actions_pred).mean()
    # Minimize the loss
    self.actor_optimizer.zero_grad()
    actor_loss.backward()
    self.actor_optimizer.step()

    # ----------------------- update target networks ----------------------- #
    self.soft_update(self.critic, self.targetCritic, self.tau)
    self.soft_update(self.actor, self.targetActor, self.tau) 
    
  def reset(self):
    self.noise.reset()
    
  def save(self, filename= None):
    data= {"config": self.config, "actor": self.actor.state_dict(), "scores": self.scores,}
    if not filename:
      filename= self.__class__.__name__+ '_'+ datetime.now().strftime("%Y-%m-%d_%H:%M:%S")+ '.data'
    torch.save(data, filename)
    torch.save(self.actor.state_dict(), "last_actor.pth")
   

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

    
