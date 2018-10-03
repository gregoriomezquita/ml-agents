# Reacher project report.
In this paper we are going to do an analysis of the DDPG algorithm in the Reacher environment of ml-agents.
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
To make it faster, the comparison will be made in the first 300 episodes.

### Number of nodes comparison
![](images/DDPG-Vanilla-Nodes.png)
### Introducing Batch Normalization
![](images/DPG-BatchNorm-Nodes.png)
### Diferent types of initialization
![](images/DDPG-init.png)
### Diferent batch sizes
![](images/DDPG-batches.png)
### Noise: standard deviation (sigma)
![](images/DDPG-sigma.png)
### Learning rate
Vamos a introducir los mejores resultados anteriores para comenzar a ver el efecto del learning rate:
```
config= {
    "critic_lr": 0.001,
    "actor_nodes": [256, 256],
    "critic_nodes": [256, 256],
    "batch_size": 512,
    "memory_size": 100000,
    "discount": 0.9,
    "sigma": 0.1, # OUNoise
    "tau": 0.001,
}
```
![](images/DDPG-actor_lr.png)
----

## Conclusions
+ Network definitions, initializer and batch normalization have been key elements for the agent to start learning.
+ Revisar sistematicamente lo hyperparametros ayuda a mejorar el aprendizaje del agente aunque no asegura encontrar los valores optimos.
+ El agente es sorprendemente sensible al ruido introducido para randomizar la exploracion de las acciones. Cuanto mas reducimos la desviacion tipica, es decir, noise variance, mas rapidamente aprende el agente.
Llevado a limite, el agente es capaz de aprender completamente la tarea eliminando completamete el ruido exploratorio en la fase de aprendizaje. Es justo lo contrario de lo que uno espera.
Esto podria explicarse porque este entorno no ofrece muchas variaciones de movimientos del objetivo.
Su comportamiento es bastante bueno:

![](images/DDPG-no-noise.gif?style=centerme)

De alguna forma la red es capaz de aprender todas las posibilidades que ofrece el entorno.
Que pasaria si redujeramos los nodos del actor para ver hasta donde es capaz de seguir aprendiendo.
+ It's amazing how quickly an agent can learn by itself with deep q-learning in a complex task.

## Improvements
+ To soften the movement of the agent, the reward could be modified depending on whether the lateral movements are very abrupt. In such a case, the reward could be reduced proportionally to lateral movement. The same could also be done in the case of the advance.
