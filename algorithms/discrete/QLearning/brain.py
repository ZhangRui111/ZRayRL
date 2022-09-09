import numpy as np
import pandas as pd


class QLearningTable:
    def __init__(self, states, actions, learning_rate=0.05, discount_factor=0.9, e_greedy=0.9):
        self.states = states
        self.actions = actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions)
        self.init_q_table()

    def choose_action(self, observation):
        """ Select an action, following epsilon-greedy policy. """
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        """ Update the Q table by the Bellman equation. """
        q_predict = self.q_table.loc[s, a]

        # if s_ != 'terminal':
        #     q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        # else:
        #     q_target = r  # next state is terminal
        q_target = r + self.gamma * self.q_table.loc[s_, :].max()

        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

    def init_q_table(self):
        """ Initialize the Q table. """
        for state in self.states:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),  # initialize all values to 0.
                    index=self.q_table.columns,
                    name=state,
                )
            )
