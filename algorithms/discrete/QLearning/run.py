from algorithms.discrete.QLearning.env import Env
from algorithms.discrete.QLearning.brain import QLearningTable

MAX_EPISODES = 500


def train(env, agent):
    print("start")
    for episode in range(MAX_EPISODES):
        s = env.reset()
        while True:
            a = agent.choose_action(str(s))
            s_, r, done, _ = env.step(a)
            agent.learn(str(s), a, r, str(s_))
            # print("state {} action {} reward {} next state {}".format(s, a, r, s_))
            # print(agent.q_table)
            s = s_

            if done:
                break

    print("game over")
    print(agent.q_table)


def main():
    env = Env()
    agent = QLearningTable(
        states=[str(item) for item in list(range(env.n_states))],
        actions=list(range(env.n_actions)),
        learning_rate=0.1,
        discount_factor=0.9,
        e_greedy=0.9,
    )
    train(env, agent)


if __name__ == '__main__':
    main()
