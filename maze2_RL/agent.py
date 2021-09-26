#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/19 下午7:14
# @Site    : 
# @File    : agent.py
# @Software: PyCharm
import numpy as np
import pandas as pd


class RL(object):
    # learning_rate学习率  reward_decay折扣率 epsilon e贪婪系数
    def __init__(self, actions, learning_rate=0.9, reward_decay=0.9, epsilon=0.01):
        self.lr = learning_rate
        self.gamma = reward_decay
        self.actions = actions
        self.epsilon = epsilon
        # Q-table分数表
        self.q_table = pd.DataFrame(columns=self.actions, dtype=float)

    def learn(self, *args):
        pass

    def action_select(self, observation):
        self.check_in_q_table(observation)
        # e贪婪策略，大于e取最高奖励
        if np.random.uniform() > self.epsilon:
            state_action = self.q_table.loc[observation, :]
            # np.max(state_action) 为下一个state行中最大的一列，即最大分数的action
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # 命中随机策略
            action = np.random.choice(self.actions)

        # print("selected action:", ['up', 'down', 'right', 'left','zs','zx','ys','yx'][action])
        return action

    def check_in_q_table(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    name=state,
                    index=self.q_table.columns
                )
            )


class QLearningTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, epsilon=0.01):
        super(QLearningTable, self).__init__(actions, learning_rate, reward_decay, epsilon)

    def learn(self, sig, a, r, sig_):
        self.check_in_q_table(sig_)
        # 获取当前q-table值
        q_value = self.q_table.loc[sig, a]
        if sig_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[sig_, :].max()
        else:
            q_target = r
        # 更新q-table
        self.q_table.loc[sig, a] += self.lr * (q_target - q_value)


class SarsaTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, epsilon=0.01):
        super(SarsaTable, self).__init__(actions, learning_rate, reward_decay, epsilon)

    def learn(self, sig, a, r, sig_, a_):
        self.check_in_q_table(sig_)
        q_value = self.q_table.loc[sig, a]
        if sig_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[sig_, a_]
        else:
            q_target = r
        self.q_table.loc[sig, a] += self.lr * (q_target - q_value)


class SarsaLambdaTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, epsilon=0.01, trace_decay=0.9):
        super(SarsaLambdaTable, self).__init__(actions, learning_rate, reward_decay, epsilon)

        # lambda-：（0，1）
        self.lambda_ = trace_decay
        self.eligibility_trace = self.q_table.copy()

    def check_in_q_table(self, state):
        if state not in self.q_table.index:
            to_be_append = pd.Series(
                [0] * len(self.actions),
                name=state,
                index=self.q_table.columns
            )

            self.q_table = self.q_table.append(to_be_append)
            self.eligibility_trace = self.eligibility_trace.append(to_be_append)

    def learn(self, sig, a, r, sig_, a_):
        self.check_in_q_table(sig_)
        q_value = self.q_table.loc[sig, a]
        if sig_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[sig_, a_]
        else:
            q_target = r
        error = q_target - q_value

        self.eligibility_trace.loc[sig, :] *= 0
        self.eligibility_trace.loc[sig, a] = 1

        self.q_table += self.lr * self.eligibility_trace * error

        self.eligibility_trace *= self.gamma * self.lambda_
