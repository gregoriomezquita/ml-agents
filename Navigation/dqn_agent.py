import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork, Dueling_QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from prioritized_memory import Memory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQN():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, config):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.config= config
        self.state_size = state_size
        self.action_size = action_size
        nodes= self.config.get("nodes", [128, 64])
        self.seed = self.config.get("seed", 0)
        lr = self.config.get("lr", 1e-4)
        memory_size = self.config.get("memory_size", 100000)
        self.batch_size = self.config.get("batch_size", 256)
        self.discount = self.config.get("discount", 0.9)
        self.tau= self.config.get("tau", 0.001)
        self.epsilon= self.config.get("epsilon", 0.1)
        self.epsilon_end= self.config.get("epsilon_end", 0.0001)
        self.epsilon_decay= self.config.get("epsilon_decay", 0.995)
        self.learn_every= self.config.get("learn_every", 4)
        self.dqn= self.config.get("dqn", "simple")
        self.per= self.config.get("per", False)
        
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        # Q-Network
        if self.dqn == "dueling":
            self.qnetwork_local = Dueling_QNetwork(state_size, action_size, self.seed).to(device)
            self.qnetwork_target = Dueling_QNetwork(state_size, action_size, self.seed).to(device)
        else:
            self.qnetwork_local = QNetwork(state_size, action_size, self.seed, nodes= nodes).to(device)
            self.qnetwork_target = QNetwork(state_size, action_size, self.seed, nodes= nodes).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr= lr)
        #self.optimizer = optim.RMSprop(self.qnetwork_local.parameters(), lr= lr)

        # Replay memory
        if self.per:
            self.memory = Memory(memory_size)
        else:
            self.memory = ReplayBuffer(memory_size, self.batch_size, self.seed)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.scores= []
        
    def add_sample(self, state, action, reward, next_state, done):
        if self.per== False:
            self.memory.add((state, action, reward, next_state, 1*done))
        else:
            
            target = self.qnetwork_local(Variable(torch.FloatTensor(state)).to(device)).data
            old_val = target[action]
            target_val = self.qnetwork_target(Variable(torch.FloatTensor(next_state)).to(device)).data
            if done:
                target[action] = reward
            else:
                target[action] = reward + self.discount * torch.max(target_val)

            error = abs(old_val - target[action])

            self.memory.add(error, (state, action, reward, next_state, 1*done))
    

    def act(self, state, add_noise= True):
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
        if add_noise== False: eps= 0.0
        else: eps= self.epsilon
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
            
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.add_sample(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.learn_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                self.learn()
           
    
    def learn(self):  
        if self.per:
            mini_batch, idxs, is_weights = self.memory.sample(self.batch_size)
            mini_batch = np.array(mini_batch).transpose()
            states = torch.from_numpy(np.vstack(mini_batch[0])).float().to(device)
            actions = torch.from_numpy(np.vstack(mini_batch[1])).long().to(device)
            rewards = torch.from_numpy(np.vstack(mini_batch[2])).float().to(device)
            next_states = torch.from_numpy(np.vstack(mini_batch[3])).float().to(device)
            dones = torch.from_numpy(np.vstack(mini_batch[4]).astype(np.uint8)).float().to(device)
            
        else:
            states, actions, rewards, next_states, dones = self.memory.sample()  
        
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions.long())
        if self.dqn == "simple":
            # Get max predicted Q values (for next states) from target model
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        elif self.dqn == "double":     # Double DQN
            _, Q_targets_next = self.qnetwork_local(next_states).detach().max(1)    # Get argmax
            Q_targets_next= self.qnetwork_target(next_states).detach().gather(1, Q_targets_next.unsqueeze(1))
        elif self.dqn== "dueling":     # Dueling
            _, Q_targets_next = self.qnetwork_local(next_states).detach().max(1)    # Get argmax
            Q_targets_next= self.qnetwork_target(next_states).detach().gather(1, Q_targets_next.unsqueeze(1))
        else:
            raise OSError('Error in DQN: {}. Options: simple, double, dueling.'.format(self.dqn))
            
        
        # Compute Q targets for current states
        Q_targets = rewards + (self.discount * Q_targets_next * (1 - dones))
        """
        # update priority
        if self.per:
            error= abs(Q_expected - Q_targets)
            errors = error.data.cpu().numpy()
            for i in range(len(idxs)):
                idx = idxs[i]
                self.memory.update(idx, errors[i])
        """
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        #loss = F.smooth_l1_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- 
        self.soft_update(self.qnetwork_local, self.qnetwork_target)                 

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def reset(self):
        pass
        
    def update(self, score):
        self.scores.append(score)
        self.epsilon= max(self.epsilon_end, self.epsilon_decay * self.epsilon) # decrease epsilon

    def save(self, filename):
        torch.save(self.qnetwork_local.state_dict(), filename)
        
    def load(self, filename):
        self.qnetwork_local.load_state_dict(torch.load(filename))


class ReplayBuffer:
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
        
