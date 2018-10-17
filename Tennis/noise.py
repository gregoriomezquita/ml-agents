"""
  Following noise definitions are base on:
    https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
""" 
import numpy as np
import random
import copy
from math import sqrt
import torch
import torch.nn as nn

class ActionNoise:
    def reset(self):
        pass
        
class NoNoise(ActionNoise):
  def __init__(self):
    pass
    
  def __call__(self):
    return 0.0
           
class OUNoise(ActionNoise):
    """Ornstein-Uhlenbeck process."""

    def __init__(self, mu, sigma=0.2, theta=0.15):
        """Initialize parameters and noise process."""
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def __call__(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class NormalActionNoise(ActionNoise):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        
    def __call__(self):
        return np.random.normal(self.mu, self.sigma)

class AdaptiveParamNoiseSpec:
    def __init__(self, initial_stddev=0.1, desired_action_stddev=0.1, adaptation_coefficient=1.01):
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adaptation_coefficient = adaptation_coefficient

        self.current_stddev = initial_stddev

    def adapt(self, distance):
        if distance > self.desired_action_stddev:
            # Decrease stddev.
            self.current_stddev /= self.adaptation_coefficient
        else:
            # Increase stddev.
            self.current_stddev *= self.adaptation_coefficient
            
    def distance(self, actions1, actions2):
      return ddpg_distance_metric(actions1, actions2) 

def ddpg_distance_metric(actions1, actions2):
    """
    Compute "distance" between actions taken by two policies at the same states
    Expects numpy arrays
    """
    """
    actions1= torch.FloatTensor(actions1)
    actions2= torch.FloatTensor(actions2)
    return nn.functional.kl_div(nn.functional.log_softmax(actions2, dim=1), nn.functional.softmax(actions1, dim=1))
    """
    diff = actions1-actions2
    mean_diff = np.mean(np.square(diff), axis=0)
    dist = sqrt(np.mean(mean_diff))
    
    return dist        
        
