# Reacher project report.

## First steps
I started out with a vanilla Deep Deterministic Policy Gradient (DDPG) agent from [Udacity Deep Learning Nanodegree repository](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-bipedal) for OpenAI Gym's BipedalWalker environment. 

To follow this project you can execute the python notebook [Continuous_Control.ipynb](Continuous_Control.ipynb). The agent is implemented in [ddpg.py](ddpg.py) which in turn needs [model.py](model.py) to define the network.
At the end of the notebook there is a cell to execute the last agent saved in the **last_actor.pth** file as well as a cell to compare diferent configurations in the same plot.

I have reduced the number of fully connected layers from 3 to 2 in the definition of the critic to reduce complexity and gain a bit in process speed. 
The following hyper parameters are the starting point:
```
config= {
    "actor_lr": 0.001,
    "critic_lr": 0.001,
    "actor_nodes": [256, 256],
    "critic_nodes": [256, 256],
    "batch_size": 128,
    "memory_size": 100000,
    "discount": 0.9,
    "sigma": 0.2, # OUNoise
    "tau": 0.001,
}
```

It is considered that the agent has learned when it gets a +30 reward for 100 episodes.


### Number of nodes comparison
![](images/DDPG-Vanilla-Nodes.png)
### Introducing Batch Normalization
![](images/DPG-BatchNorm-Nodes.png)
### Diferent types of initialization
![](images/DDPG-init.png)
### Epsilon + noise
```
config= {
    "label": "Epsilon=1.0 + noise",
    "state_size": len(state),
    "action_size": brain.vector_action_space_size,
    "seed": seed,
    "actor_lr": 0.001,
    "critic_lr": 0.001,
    "actor_nodes": [256, 256],
    "critic_nodes": [256, 256],
    "batch_size": 256,
    "memory_size": 100000,
    "discount": 0.9,
    "sigma": 0.2, # OUNoise
    "tau": 0.001,
    "epsilon": 1.0,
    "epsilon_decay": 1e-6,
}
```
![](images/DDPG-epsilon.png)
----
I let the agent continue its training after solving the task for a total of **500 or 1000** episodes or it reaches an average score of **15** per 100 episodes.

After playing a bit with the hyperparameters, you can get the agent to learn the task in up to **172 episodes** with:
**2 hidden layers of 128 and 256 nodes, batch size= 256, learning rate= 1e-4, discount factor= 0.99, epsilon start= 0.1, epsilon end= 0.0001, target updates= 3 steps**


![](images/172.png)

There are some improvements in the literature to try to overcome the algorithm of deep q-learning. This is Prioritized Experiece Replay, Double Q-Learning and Dueling DQN. All of them have been implemented and will be compared, fixing as hyper parameters the last ones that have given the best results in terms of learning speed (172 episodes).
In the file [dqn_agent.py](dqn_agent.py) you can choose these improvements by modifying the following constants:
```
SIMPLE_DQN= True        # Simple DQN. If not Double DQN
PER = False             # Prioritized Experience Replay
DUELING_DQN= False      # dueling DQN
```

## Prioritized experience replay
DQN with PER. Thanks to https://github.com/rlcode/per for the implemetation of PER. It has been modified to suit my needs.
Default parameters:
```
e = 0.01
a = 0.6
beta = 0.4
beta_increment_per_sampling = 0.001
```

![](images/DQN_PER.png)

## Double DQN

![](images/DDQN.png)

## Double DQN with Prioritized Experience Replay

![](images/DDQN_PER_209.png)

## Dueling DQN

Implementado en [model.py](model.py).

![](images/Dueling_DQN.png)

## Conclusions
+ No fundamental advantage has been found with DDQN or with PER or Dueling over DQN.
With certain DQN hyperparameters, it turns out to be better in terms of learning speed, reaching **172 episodes at best**.
+ In all cases, learning is very unstable, with the reward graph being very noisy, although there is generally a steady progress up to 13 and then it stabilizes.
+ The agent behaves surprisingly well once it has learned the task although sometimes its movement is a little abrupt. Also sometimes it gets stuck making an oscillating movement without being able to decide which way to go.
+ It's amazing how quickly an agent can learn by itself with deep q-learning in a complex task.

## Improvements
+ To soften the movement of the agent, the reward could be modified depending on whether the lateral movements are very abrupt. In such a case, the reward could be reduced proportionally to lateral movement. The same could also be done in the case of the advance.
