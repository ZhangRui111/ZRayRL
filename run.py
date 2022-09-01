import argparse
# import importlib
import gym
import numpy as np
import os
import random
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("alg", type=str,  default="",
                        choices=["DQN", "doubleDQN", "DuelingDQN", "D3QN",
                                 "DDPG", "TD3",
                                 "PPO"],
                        help="the DRL algorithm name.")
    parser.add_argument("act_type", type=str, default="",
                        choices=["discrete", "continuous"],
                        help="discrete/continuous action space.")
    opt = parser.parse_args()

    # set the random seed
    seed = 777
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # the root path for logging
    log_root = "logs"
    os.makedirs(log_root, exist_ok=True)

    # import corresponding modules
    # ---------- solution 1 ----------
    # algorithm_module_path = "algorithms.{}.{}".format(opt.act_type, opt.alg)
    # algorithm_module = importlib.import_module(algorithm_module_path)
    # ---------- solution 2 ----------
    if opt.alg == "DQN":
        from algorithms.discrete.DQN.agent import DQNAgent
    elif opt.alg == "D3QN":
        pass
    else:
        raise NotImplementedError("{} is not implemented".format(opt.alg))

    # initialize the environment
    if opt.act_type == "discrete":
        env_id = "CartPole-v1"
        env = gym.make(env_id, new_step_api=True)
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
    elif opt.act_type == "continuous":
        pass
    else:
        raise Exception()

    # initialize the agent and launch the training
    if opt.alg == "DQN":
        # hyper-parameters
        num_frames = 10000
        lr = 1e-3
        memory_size = 1000
        batch_size = 32
        target_update = 100
        epsilon_decay = 1 / 2000  # it takes 2000 frames to reach the min_epsilon
        agent = DQNAgent(env, obs_dim, action_dim, lr, memory_size,
                         batch_size, target_update, epsilon_decay)
    elif opt.alg == "D3QN":
        pass
    else:
        raise NotImplementedError("{} is not implemented".format(opt.alg))

    agent.train(num_frames)


if __name__ == '__main__':
    main()
