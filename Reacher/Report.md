# Reacher project report.
PPO and DDPG stand out among the different algorithms that we can choose to face a control problem with a continuous space.
DDPG has been chosen because it generally presents a better behavior and stability although at the expense of greater computing resources.
To follow this project you can execute the python notebook [Continuous_Control.ipynb](Continuous_Control.ipynb). The agent is implemented in [ddpg.py](ddpg.py) which in turn needs [model.py](model.py) to define the network.
At the end of the notebook there is a cell to execute the last agent saved in the **last_actor.pth** file as well as a cell to compare diferent configurations in the same plot.
It is considered that the agent has learned when it gets a +30 reward for 100 episodes.
## First steps
I started out with a Deep Deterministic Policy Gradient (DDPG) agent from [Udacity Deep Learning Nanodegree repository](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-bipedal) for OpenAI Gym's BipedalWalker environment. 
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
## Hyperparameters selection
Choosing hyperparameters is not an easy task since the number of possibilities is very high. There is no systematic method and it is one of the biggest challenges of the current DRL. Frequently choosing a suitable parameter can mean the difference between the agent learning or not at all.
However, we can make a comparison of different values of each parameter to try to face the question.

### Number of nodes comparison
![](images/DDPG-Vanilla-Nodes.png)
### Introducing Batch Normalization
![](images/DPG-BatchNorm-Nodes.png)
### Diferent types of initialization
![](images/DDPG-init.png)
### Diferent batch sizes
![](images/DDPG-batches.png)
### Epsilon + noise
```
config= {
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

## Conclusions
+ No fundamental advantage has been found with DDQN or with PER or Dueling over DQN.
With certain DQN hyperparameters, it turns out to be better in terms of learning speed, reaching **172 episodes at best**.
+ In all cases, learning is very unstable, with the reward graph being very noisy, although there is generally a steady progress up to 13 and then it stabilizes.
+ The agent behaves surprisingly well once it has learned the task although sometimes its movement is a little abrupt. Also sometimes it gets stuck making an oscillating movement without being able to decide which way to go.
+ It's amazing how quickly an agent can learn by itself with deep q-learning in a complex task.

## Improvements
+ To soften the movement of the agent, the reward could be modified depending on whether the lateral movements are very abrupt. In such a case, the reward could be reduced proportionally to lateral movement. The same could also be done in the case of the advance.
