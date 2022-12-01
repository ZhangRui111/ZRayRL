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
                                 "PERDQN", "DDPG", "TD3", "REINFORCE", "PPO",
                                 "A2C", "A3C", "SAC", "DiscreteSAC"],
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
        assert opt.act_type == "discrete", \
            "DQN does not support continuous action space"
        from algorithms.discrete.DQN.agent import DQNAgent
    elif opt.alg == "DoubleDQN":
        assert opt.act_type == "discrete", \
            "DoubleDQN does not support continuous action space"
        from algorithms.discrete.DoubleDQN.agent import DQNAgent
    elif opt.alg == "DuelingDQN":
        assert opt.act_type == "discrete", \
            "DuelingDQN does not support continuous action space"
        from algorithms.discrete.DuelingDQN.agent import DQNAgent
    elif opt.alg == "PERDQN":
        assert opt.act_type == "discrete", \
            "PERDQN does not support continuous action space"
        from algorithms.discrete.PERDQN.agent import DQNAgent
    elif opt.alg == "D3QN":
        assert opt.act_type == "discrete", \
            "D3QN does not support continuous action space"
        from algorithms.discrete.D3QN.agent import DQNAgent
    elif opt.alg == "DDPG":
        assert opt.act_type == "continuous", \
            "DDPG does not support discrete action space"
        from algorithms.continuous.DDPG.agent import DDPGAgent
    elif opt.alg == "TD3":
        assert opt.act_type == "continuous", \
            "TD3 does not support discrete action space"
        from algorithms.continuous.TD3.agent import TD3Agent
    elif opt.alg == "REINFORCE":
        if opt.act_type == "discrete":
            from algorithms.discrete.REINFORCE.agent import REINFORCEAgent
        else:
            from algorithms.continuous.REINFORCE.agent import REINFORCEAgent
    elif opt.alg == "PPO":
        if opt.act_type == "discrete":
            from algorithms.discrete.PPO.agent import PPOAgent
        else:
            from algorithms.continuous.PPO.agent import PPOAgent
    elif opt.alg == "A2C":
        if opt.act_type == "discrete":
            from algorithms.discrete.A2C.agent import A2CAgent
        else:
            from algorithms.continuous.A2C.agent import A2CAgent
    elif opt.alg == "SAC":
        assert opt.act_type == "continuous", "SAC does not support discrete action space"
        from algorithms.continuous.SAC.agent import SACAgent
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
        args = {
            'obs_dim': obs_dim,
            'action_dim': action_dim,
            'lr': 1e-3,
            'memory_size': 1000,
            'batch_size': 32,
            'target_update': 100,
            # it takes 4000 frames to reach the min_epsilon
            'epsilon_decay': 1 / 2000,
        }
        agent = DQNAgent(env, **args)
    elif opt.alg == "DoubleDQN":
        # hyper-parameters
        num_frames = 20000
        args = {
            'obs_dim': obs_dim,
            'action_dim': action_dim,
            'lr': 1e-3,
            'memory_size': 1000,
            'batch_size': 32,
            'target_update': 100,
            'epsilon_decay': 1 / 2000,
        }
        agent = DQNAgent(env, **args)
    elif opt.alg == "DuelingDQN":
        # hyper-parameters
        num_frames = 20000
        args = {
            'obs_dim': obs_dim,
            'action_dim': action_dim,
            'lr': 1e-3,
            'memory_size': 1000,
            'batch_size': 32,
            'target_update': 100,
            'epsilon_decay': 1 / 2000,
        }
        agent = DQNAgent(env, **args)
    elif opt.alg == "PERDQN":
        # hyper-parameters
        num_frames = 20000
        args = {
            'obs_dim': obs_dim,
            'action_dim': action_dim,
            'lr': 1e-3,
            'memory_size': 1000,
            'batch_size': 32,
            'target_update': 100,
            'epsilon_decay': 1 / 2000,
            'alpha': 0.2,
            'beta': 0.6,
            'prior_eps': 1e-6,
        }
        agent = DQNAgent(env, **args)
    elif opt.alg == "D3QN":
        # hyper-parameters
        num_frames = 20000
        args = {
            'obs_dim': obs_dim,
            'action_dim': action_dim,
            'lr': 1e-3,
            'memory_size': 1000,
            'batch_size': 32,
            'target_update': 100,
            'epsilon_decay': 1 / 2000,
        }
        agent = DQNAgent(env, **args)
    elif opt.alg == "DDPG":
        # hyper-parameters
        num_frames = 40000
        args = {
            'obs_dim': obs_dim,
            'action_dim': action_dim,
            'action_low': action_low,
            'action_high': action_high,
            'lr_actor': 3e-4,
            'lr_critic': 1e-3,
            'memory_size': 50000,
            'batch_size': 128,
            'initial_random_steps': 10000,
            'ou_noise_theta': 1.0,
            'ou_noise_sigma': 0.1,
        }
        agent = DDPGAgent(env, **args)
    elif opt.alg == "TD3":
        # hyper-parameters
        num_frames = 40000
        args = {
            'obs_dim': obs_dim,
            'action_dim': action_dim,
            'action_low': action_low,
            'action_high': action_high,
            'lr_actor': 3e-4,
            'lr_critic': 1e-3,
            'memory_size': 50000,
            'batch_size': 128,
            'initial_random_steps': 10000,
        }
        agent = TD3Agent(env, **args)
    elif opt.alg == "REINFORCE":
        if opt.act_type == "discrete":
            # hyper-parameters
            num_frames = 50000
            lr = 5e-3
            agent = REINFORCEAgent(env, obs_dim, action_dim,
                                   lr, gamma=0.9, entropy_weight=0.01)
        else:
            # hyper-parameters
            num_frames = 50000
            lr = 5e-3
            agent = REINFORCEAgent(env, obs_dim, action_dim, action_low, action_high,
                                   lr, gamma=0.9, entropy_weight=0.01)
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
    elif opt.alg == "A2C":
        if opt.act_type == "discrete":
            # hyper-parameters
            num_frames = 50000
            lr_actor = 1e-3
            lr_critic = 2e-3
            agent = A2CAgent(env, obs_dim, action_dim,
                             lr_actor, lr_critic, gamma=0.9, entropy_weight=0.01)
        else:
            # hyper-parameters
            num_frames = 50000
            lr_actor = 2e-3
            lr_critic = 5e-3
            agent = A2CAgent(env, obs_dim, action_dim, action_low, action_high,
                             lr_actor, lr_critic, gamma=0.9, entropy_weight=0.01)
    elif opt.alg == "SAC":
        # hyper-parameters
        num_frames = 50000
        lr_actor = 1e-3
        lr_critic_q = 3e-3
        lr_critic_v = 3e-3
        memory_size = 100000
        batch_size = 128
        initial_random_steps = 10000
        agent = SACAgent(env, obs_dim, action_dim, action_low, action_high,
                         lr_actor, lr_critic_q, lr_critic_v, memory_size,
                         batch_size, initial_random_steps=initial_random_steps)
    else:
        raise NotImplementedError("{} is not implemented".format(opt.alg))

    # train
    agent.train(num_frames)

    # test
    agent.test()


if __name__ == '__main__':
    main()
