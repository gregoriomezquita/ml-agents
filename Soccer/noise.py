"""
  Following noise definitions are base on:
    https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
""" 
import numpy as np
import random
import copy

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
    def __init__(self, initial_stddev=0.1, desired_action_stddev=0.1, adoption_coefficient=1.01):
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adoption_coefficient = adoption_coefficient

        self.current_stddev = initial_stddev

    def adapt(self, distance):
        if distance > self.desired_action_stddev:
            # Decrease stddev.
            self.current_stddev /= self.adoption_coefficient
        else:
            # Increase stddev.
            self.current_stddev *= self.adoption_coefficient

def ddpg_distance_metric(actions1, actions2):
    """
    Compute "distance" between actions taken by two policies at the same states
    Expects numpy arrays
    """
    diff = actions1-actions2
    mean_diff = np.mean(np.square(diff), axis=0)
    dist = sqrt(np.mean(mean_diff))
    return dist            
