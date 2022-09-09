# ZRayRL

This project is a RL toolbox with a basic/concise and consistent implementation.

## I. Installation and Run
### I.1. Main Dependencies
- python
- pytorch
- OpenAI gym
- numpy

### I.2. Configure the Environment with Anaconda

1. Locate the root directory
1. Create a new conda environment, ```conda env create -f zrayrl.yaml```
1. Activate the new conda environment, ```conda activate zrayrl```

### I.3 How to Run
```python run.py [alg] [act_type]```, e.g., ```python run.py DQN discrete```.

## II. Algorithm List and Related Papers

| Algorithm | Action | Paper |
| :---: | :--- | :--- |
| Q-Learning | [Discrete](https://github.com/ZhangRui111/ZRayRL/tree/master/algorithms/discrete/QLearning) | \ |
| DQN | [Discrete](https://github.com/ZhangRui111/ZRayRL/tree/master/algorithms/discrete/DQN) | [Human-Level Control through Deep Reinforcement Learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) |
| DoubleDQN | [Discrete](https://github.com/ZhangRui111/ZRayRL/tree/master/algorithms/discrete/DoubleDQN) | [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/pdf/1509.06461.pdf) |
| DuelingDQN | [Discrete](https://github.com/ZhangRui111/ZRayRL/tree/master/algorithms/discrete/DuelingDQN) | [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/pdf/1511.06581.pdf) |
| PERDQN | [Discrete](https://github.com/ZhangRui111/ZRayRL/tree/master/algorithms/discrete/PERDQN) | [Prioritized Experience Replay](https://arxiv.org/pdf/1511.05952.pdf) |
| D3QN | [Discrete](https://github.com/ZhangRui111/ZRayRL/tree/master/algorithms/discrete/D3QN) | Double Dueling DQN |
| DDPG | [Continuous](https://github.com/ZhangRui111/ZRayRL/tree/master/algorithms/continuous/DDPG) | [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/pdf/1509.02971.pdf) |
| TD3 | [Continuous](https://github.com/ZhangRui111/ZRayRL/tree/master/algorithms/continuous/TD3) | [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/pdf/1802.09477.pdf) |
| REINFORCE | [Discrete](https://github.com/ZhangRui111/ZRayRL/tree/master/algorithms/discrete/REINFORCE) / [Continuous](https://github.com/ZhangRui111/ZRayRL/tree/master/algorithms/continuous/REINFORCE) | [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://people.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf) |
| PPO | [Discrete](https://github.com/ZhangRui111/ZRayRL/tree/master/algorithms/discrete/PPO) / [Continuous](https://github.com/ZhangRui111/ZRayRL/tree/master/algorithms/continuous/PPO) | [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) |
| A2C | [Discrete](https://github.com/ZhangRui111/ZRayRL/tree/master/algorithms/discrete/A2C) / [Continuous](https://github.com/ZhangRui111/ZRayRL/tree/master/algorithms/continuous/A2C) | [Asynchronous Methods for Deep Reinforcement Learning](http://proceedings.mlr.press/v48/mniha16.pdf) |
| A3C (TODO) | Discrete/Continuous | [Asynchronous Methods for Deep Reinforcement Learning](http://proceedings.mlr.press/v48/mniha16.pdf) |
| SAC | [Continuous](https://github.com/ZhangRui111/ZRayRL/tree/master/algorithms/continuous/SAC) | [1. Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/pdf/1801.01290.pdf) <br> [2. Soft Actor-Critic Algorithms and Applications](https://arxiv.org/pdf/1812.05905.pdf) |
| DiscreteSAC (TODO) | Discrete | [Soft Actor-Critic for Discrete Action Settings](https://arxiv.org/pdf/1910.07207.pdf) |

TODO: An illustration on relationships among all algorithms

### III. Practical Tips
#### III.1. Tips

1. Algorithm selection
   - For discrete tasks: **D3QN**.
   - For continuous tasks: **PPO**, **TD3**.
1. **NO Batch Normalization**
1. Try a small learning rate from the beginning, e.g., 1e-4.
   
   PPO can handle larger learning rate? Maybe.
   
1. Reward function: global reward + factor * local reward
    - global reward cares for the final target, while local reward aims to provide some prior knowledge.
    - the positive factor should be lower than 1, i.e., \[0, 1\].
    - tune the factor carefully.
1. The **learning rate** and the **reward function** are among the top factors in RL training.
1. When applying RL to deal with CV problems, do not involve the image feature extractor into the policy/value network. 
   
   Usually, we employ a separate feature extractor decoupled from the policy/value network.
   We can load ImageNet-pretrained weights to the feature extractor while build the policy/value network with multiple FC layers.
   This can help reduce the number of trainable parameters and speed up the training.
   
1. For algorithm that has actor and critic, the learning rate for the critic usually is larger than that for the actor.
1. Be careful with the activation (type) following the output layer.
1. Normalization on the observation/state. 
1. Larger policy/value network is not necessarily better. For some simple tasks/states, smaller policy/value network can save lots of training time.

#### III.2. Optional Tips

1. Gradient Clipping on the loss/error term
1. Change the random seed (emmm)

#### III.3. More Tips

1. Occam's Razor: Entities should not be multiplied unnecessarily.
1. SOTA algorithm does not necessarily perform best in practical applications. 
   
   Or, tricks that worked in the paper do not necessarily work in practical applications.

#### III.3. References

## Acknowledgement
- [Rainbow is all you need!](https://github.com/Curt-Park/rainbow-is-all-you-need)
- [PG is all you need!](https://github.com/MrSyee/pg-is-all-you-need)
- [ElegantRL](https://github.com/AI4Finance-Foundation/ElegantRL)
