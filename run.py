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
                        choices=["DQN", "DoubleDQN", "DuelingDQN", "D3QN",
                                 "DDPG", "TD3",
                                 "PPO"],
                        help="the DRL algorithm name.")
    parser.add_argument("act_type", type=str, default="",
                        choices=["discrete", "continuous"],
                        help="discrete/continuous action space.")
    opt = parser.parse_args()

    # set the random seed (reproductivity)
    seed = 777
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

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
    elif opt.alg == "DoubleDQN":
        from algorithms.discrete.DoubleDQN.agent import DQNAgent
    elif opt.alg == "DuelingDQN":
        from algorithms.discrete.DuelingDQN.agent import DQNAgent
    elif opt.alg == "D3QN":
        from algorithms.discrete.D3QN.agent import DQNAgent
    elif opt.alg == "DDPG":
        from algorithms.continuous.DDPG.agent import DDPGAgent
    elif opt.alg == "TD3":
        from algorithms.continuous.TD3.agent import TD3Agent
    elif opt.alg == "PPO":
        if opt.act_type == "discrete":
            from algorithms.discrete.PPO.agent import PPOAgent
        else:
            from algorithms.continuous.PPO.agent import PPOAgent
    else:
        raise NotImplementedError("{} is not implemented".format(opt.alg))

    # initialize the environment
    if opt.act_type == "discrete":
        env_id = "CartPole-v1"
        env = gym.make(env_id, new_step_api=True)
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
    elif opt.act_type == "continuous":
        env_id = "Pendulum-v1"
        env = gym.make(env_id, new_step_api=True)
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_low = env.action_space.low
        action_high = env.action_space.high
    else:
        raise Exception()

    # initialize the agent and launch the training
    if opt.alg == "DQN":
        # hyper-parameters
        num_frames = 20000
        lr = 1e-3
        memory_size = 1000
        batch_size = 32
        target_update = 100
        epsilon_decay = 1 / 2000  # it takes 2000 frames to reach the min_epsilon
        agent = DQNAgent(env, obs_dim, action_dim, lr, memory_size,
                         batch_size, target_update, epsilon_decay)
    elif opt.alg == "DoubleDQN":
        # hyper-parameters
        num_frames = 20000
        lr = 1e-3
        memory_size = 1000
        batch_size = 32
        target_update = 100
        epsilon_decay = 1 / 2000  # it takes 2000 frames to reach the min_epsilon
        agent = DQNAgent(env, obs_dim, action_dim, lr, memory_size,
                         batch_size, target_update, epsilon_decay)
    elif opt.alg == "DuelingDQN":
        # hyper-parameters
        num_frames = 20000
        lr = 1e-3
        memory_size = 1000
        batch_size = 32
        target_update = 100
        epsilon_decay = 1 / 2000  # it takes 2000 frames to reach the min_epsilon
        agent = DQNAgent(env, obs_dim, action_dim, lr, memory_size,
                         batch_size, target_update, epsilon_decay)
    elif opt.alg == "D3QN":
        # hyper-parameters
        num_frames = 20000
        lr = 1e-3
        memory_size = 1000
        batch_size = 32
        target_update = 100
        epsilon_decay = 1 / 2000  # it takes 2000 frames to reach the min_epsilon
        agent = DQNAgent(env, obs_dim, action_dim, lr, memory_size,
                         batch_size, target_update, epsilon_decay)
    elif opt.alg == "DDPG":
        # hyper-parameters
        num_frames = 50000
        lr_actor = 3e-4
        lr_critic = 1e-3
        memory_size = 100000
        batch_size = 128
        ou_noise_theta = 1.0
        ou_noise_sigma = 0.1
        initial_random_steps = 10000
        agent = DDPGAgent(env, obs_dim, action_dim, action_low, action_high,
                          lr_actor, lr_critic, memory_size,
                          batch_size, ou_noise_theta, ou_noise_sigma,
                          initial_random_steps=initial_random_steps)
    elif opt.alg == "TD3":
        # hyper-parameters
        num_frames = 50000
        lr_actor = 3e-4
        lr_critic = 1e-3
        memory_size = 100000
        batch_size = 128
        initial_random_steps = 10000
        agent = TD3Agent(env, obs_dim, action_dim, action_low, action_high,
                         lr_actor, lr_critic, memory_size,
                         batch_size, initial_random_steps=initial_random_steps)
    elif opt.alg == "PPO":
        if opt.act_type == "discrete":
            # hyper-parameters
            num_frames = 50000
            lr_actor = 2e-3
            lr_critic = 5e-3
            batch_size = 128
            agent = PPOAgent(env, obs_dim, action_dim,
                             lr_actor, lr_critic, batch_size, gamma=0.9, tau=0.8,
                             epsilon=0.2, epoch=32, rollout_len=1024, entropy_weight=0.005)
        else:
            # hyper-parameters
            num_frames = 50000
            lr_actor = 2e-3
            lr_critic = 5e-3
            batch_size = 128
            agent = PPOAgent(env, obs_dim, action_dim, action_low, action_high,
                             lr_actor, lr_critic, batch_size, gamma=0.9, tau=0.8,
                             epsilon=0.2, epoch=32, rollout_len=1024, entropy_weight=0.005)
    else:
        raise NotImplementedError("{} is not implemented".format(opt.alg))

    agent.train(num_frames)


if __name__ == '__main__':
    main()
