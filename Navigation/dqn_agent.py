import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork, Dueling_QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from prioritized_memory import Memory

SIMPLE_DQN= True        # Simple DQN. If not Double DQN
PER = True             # Prioritized Experience Replay
DUELING_DQN= False      # dueling DQN

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 256         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 1e-4               # learning rate 
UPDATE_EVERY = 3        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        if DUELING_DQN:
          self.qnetwork_local = Dueling_QNetwork(state_size, action_size, seed).to(device)
          self.qnetwork_target = Dueling_QNetwork(state_size, action_size, seed).to(device)
        else:
          self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
          self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        #self.optimizer = optim.RMSprop(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        if PER:
          self.memory = Memory(BUFFER_SIZE)
        else:
          self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
    
        
    def add_sample(self, state, action, reward, next_state, done):
        if PER== False:
          self.memory.add(state, action, reward, next_state, done)
        else:
          target = self.qnetwork_local(Variable(torch.FloatTensor(state))).data
          old_val = target[action]
          target_val = self.qnetwork_target(Variable(torch.FloatTensor(next_state))).data
          if done:
              target[action] = reward
          else:
              target[action] = reward + GAMMA * torch.max(target_val)

          error = abs(old_val - target[action])

          self.memory.add(error, (state, action, reward, next_state, done))
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.add_sample(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
              if PER:
                mini_batch, idxs, is_weights = self.memory.sample(BATCH_SIZE)
                mini_batch = np.array(mini_batch).transpose()
                states = torch.from_numpy(np.vstack(mini_batch[0])).float().to(device)
                actions = torch.from_numpy(np.vstack(mini_batch[1])).long().to(device)
                rewards = torch.from_numpy(np.vstack(mini_batch[2])).float().to(device)
                next_states = torch.from_numpy(np.vstack(mini_batch[3])).float().to(device)
                dones = torch.from_numpy(np.vstack(mini_batch[4]).astype(np.uint8)).float().to(device)
                #dones = mini_batch[4]
                # bool to binary
                #dones = dones.astype(int)
                ########################3
                experiences= (states, actions, rewards, next_states, dones, idxs)
              else:
                experiences = self.memory.sample()
                
              self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()    #set to eval mode
        with torch.no_grad():
            action_values = self.qnetwork_local.forward(state)
        self.qnetwork_local.train()   # set to training mode

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
            
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        if PER:
          states, actions, rewards, next_states, dones, idxs = experiences
        else:
          states, actions, rewards, next_states, dones = experiences
        
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        if SIMPLE_DQN:
          # Get max predicted Q values (for next states) from target model
          Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        else:     # Double DQN
          _, Q_targets_next = self.qnetwork_local(next_states).detach().max(1)    # Get argmax
          Q_targets_next= self.qnetwork_target(next_states).detach().gather(1, Q_targets_next.unsqueeze(1))
        
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        error= abs(Q_expected - Q_targets)
        
        

        # update priority
        #errors = error.data.numpy()
        #for i in range(BATCH_SIZE):
        #    idx = idxs[i]
        #    self.memory.update(idx, errors[i])

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        #loss = F.smooth_l1_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
          target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


    def save(self, filename):
        torch.save(self.qnetwork_local.state_dict(), filename)
        
    def load(self, filename):
        self.qnetwork_local.load_state_dict(torch.load(filename))


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
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
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
