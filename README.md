# ZRayRL

This project is a RL toolbox with a basic/concise implementation.

---

## I. Dependencies and Install
### I.1. Main Dependencies
- python
- pytorch
- OpenAI gym
- numpy

### I.2. Install

## II. Algorithm List and Related Papers

| Algorithm | Action | Paper |
| :---: | :--- | :--- |
| DQN | Discrete | [Human-level control through deep reinforcement learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) |
| Double DQN | Discrete | [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/pdf/1509.06461.pdf) |
| Dueling DQN | Discrete | [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/pdf/1511.06581.pdf) |
| D3QN | Discrete | Double Dueling DQN |
| DDPG | Continuous | [Continuous control with deep reinforcement learning](https://arxiv.org/pdf/1509.02971.pdf) |
| PPO | Discrete/Continuous | |

### III. Practical Tips
#### III.1. Tips

1. Algorithm selection
   - For discrete tasks: D3QN.
   - For continuous tasks: PPO, TD3.
1. NO Batch Normalization
1. Try a small learning rate from the beginning, e.g., 1e-4.
1. Reward function: global reward + factor * local reward
    - global reward cares for the final target, while local reward aims to provide some prior knowledge.
    - the positive factor should be lower than 1, i.e., \[0, 1\].
    - tune the factor carefully.
1. The learning rate and the reward function are among the top factors in RL training.
1. When applying RL to deal with CV problems, do not involve the image feature extractor into the policy/value network. 
   
   Usually, we employ a separate feature extractor decoupled from the policy/value network.
   We can load ImageNet-pretrained weights to the feature extractor while build the policy/value network with multiple FC layers.
   This can help reduce the number of trainable parameters and speed up the training.
   
1. For algorithm that has actor and critic, the learning rate for the critic usually is larger than that for the actor.
1. Be careful with the activation (type) following the output layer.

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
