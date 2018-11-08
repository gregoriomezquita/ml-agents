import torch
import numpy as np
from glob import glob
import re

from ddpg import DDPG, device

class Agents:
    def __init__(self, state_size, action_size, num_agents=1, config= None):
        if not config: 
            raise OSError('DDPG: no configuration parameter in class init')
        self.config= config
        self.num_agents= num_agents
        self.agents= []
        self.scores= []
        self.best_actors= []
        actor_state_size, actor_action_size= state_size, action_size 
        critic_state_size, critic_action_size= state_size, action_size
        if self.config["multi-agent"]:
            critic_state_size, critic_action_size= (state_size*num_agents), action_size*num_agents
            
        for i in range(num_agents):
            self.agents.append( DDPG(actor_state_size, actor_action_size, 
                                     critic_state_size, critic_action_size,
                                     config= config) )
            self.best_actors.append(self.agents[i].actor.state_dict())

    def reset(self):
        for agent in self.agents: 
            agent.reset()
            
    def act(self, state, noise= True):
        actions= np.hstack([agent.act( state[i], noise ) for i, agent in enumerate(self.agents)])[0]
        return actions
    
    def get_target_action(self, full_states, state_size):
        fs= torch.split(full_states.cpu(), state_size, dim= 1)
        return torch.cat([agent.targetActor(fs[i].to(device)) for i, agent in enumerate(self.agents)], 1)
    
    def get_action(self, full_states, state_size):
        fs= torch.split(full_states.cpu(), state_size, dim= 1)
        return torch.cat([agent.actor(fs[i].to(device)) for i, agent in enumerate(self.agents)], 1)
        
    def step(self, state, action, reward, next_state, done):
        
        if self.config["multi-agent"]:   # Multi-agent experience replay
            full_state, full_next_state= np.hstack(state), np.hstack(next_state)
            if self.config["experience"]== "self":
                for i, agent in enumerate(self.agents):  # Experience replay only from self
                    agent.memory.add((full_state, action, reward[i], full_next_state, 1*done[i]))
            elif self.config["experience"]== "all":
                for agent in self.agents:  # Experience replay from all agents
                    for i in range(self.num_agents):
                        agent.memory.add((full_state, action, reward[i], full_next_state, 1*done[i]))
            for agent in self.agents:
                if len(agent.memory) >= agent.batch_size:
                    full_states, actions, rewards, full_next_states, dones= agent.memory.sample()
                    target= self.get_target_action(full_next_states, agent.actor_state_size)
                    actor=  self.get_action(full_states, agent.actor_state_size)
                    agent.learn_critic(full_states, actions, rewards, full_next_states, dones, target)
                    agent.learn_actor(full_states, actor)
        else:    
            if self.config["experience"]== "self":   # Self experience replay
                for i, agent in zip(range(self.num_agents), self.agents):
                    s= slice(agent.actor_action_size * i, (agent.actor_action_size * i) + agent.actor_action_size)
                    agent.step(state[i], action[s], reward[i], next_state[i], 1*done[i])
            else:    # All experience replay
                for agent in self.agents:
                    for i in range(self.num_agents):
                        s= slice(agent.actor_action_size*i, (agent.actor_action_size*i)+ agent.actor_action_size)
                        agent.memory.add((state[i], action[s], reward[i], next_state[i], 1*done[i]))
                    if len(agent.memory) >= agent.batch_size:
                        agent.learn()
                             
            
    def update(self, score):
        # Save actors for best score
        if len(self.scores) and score > np.max(self.scores): 
            for i, agent in enumerate(self.agents):
                self.best_actors[i]= agent.actor.state_dict()
        self.scores.append(score)
                                      
    def save(self, solved= 0):
        data= {"config": self.config, "scores": self.scores, "actors": self.best_actors, "solved": solved}
        fname= "./{}_".format(self.__class__.__name__)
        last_index= 1
        files= sorted(glob("{}*.data".format(fname)))
        if files:
            last= files[-1]
            if last: 
                last_index= int(re.findall('[0-9]+', last, flags=re.IGNORECASE)[0])
                last_index+= 1
        filename= "{}{}.data".format(fname, last_index)
        torch.save(data, filename)
        
        for i in range(len(self.best_actors)):
            torch.save(self.best_actors[i], "last_actor_{}.pth".format(i+ 1))
        
    def load(self):
        for i, agent in enumerate(self.agents):
            model= torch.load("last_actor_{}.pth".format(i+ 1), map_location=lambda storage, loc: storage)
            agent.actor.load_state_dict(model)
        
    def eval(self):
        for agent in self.agents:
            agent.actor.eval() 

