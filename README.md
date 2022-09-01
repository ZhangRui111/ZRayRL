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
| PPO | Discrete/Continuous | |

### III. Practical Tips
#### III.1. Tips

1. Algorithm selection
   - For discrete tasks: D3QN
   - For continuous tasks: PPO, TD3.
1. NO Batch Normalization
1. Reward function: global reward + factor * local reward
    - global reward cares for the final target, while local reward aims to provide some prior knowledge.
    - the positive factor should be lower than 1, i.e., \[0, 1\].
    - tune the factor carefully.
1. SOTA algorithm does not necessarily perform best in practical applications.

#### III.2. Optional Tips

1. Gradient Clipping on the loss/error term
1. Change the random seed (emmm)

#### III.3. References

## Acknowledgement
- [Rainbow is all you need!](https://github.com/Curt-Park/rainbow-is-all-you-need)
- [PG is all you need!](https://github.com/MrSyee/pg-is-all-you-need)
