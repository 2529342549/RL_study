#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/24 下午7:18
# @Site    : 
# @File    : maze_agent.py
# @Software: PyCharm
import numpy as np


class Qlearning(object):
    def __init__(self, observation_n, action_n, learning_rate=0.01, gamma=0.9, e_greed=0.1):
        self.action_n = action_n  # 动作空间
        self.lr = learning_rate  # 学习效率
        self.gamma = gamma  # reward衰减率
        self.epsilon = e_greed  # 按一定概率随机选取动作
        self.Q = np.zeros((observation_n, action_n))

    def sample(self, observation):
        if np.random.uniform(0, 1) < (1 - self.epsilon):
            # 根据Q—table选取Q值
            action = self.predict(observation)
        else:
            action = np.random.choice(self.action_n)
        return action

    # 根据输入的观测值预测输出的动作，带探索
    def predict(self, observation):
        Q_list = self.Q[observation, :]
        maxQ = np.max(Q_list)
        action_list = np.where(Q_list == maxQ)[0]
        action = np.random.choice(action_list)
        return action

    def learn(self, observation, action, reward, next_observation, done):
        predict_Q = self.Q[observation, action]
        if done:
            target_Q = reward
        else:
            target_Q = reward + self.gamma * np.max(self.Q[next_observation, :])
        # 更新Q
        self.Q[observation, action] += self.lr * (target_Q - predict_Q)

    def save(self):
        Q_file = './q_table.npy'
        np.save(Q_file, self.Q)
        print(Q_file + 'saved.')

    # 从文件中读取数据到Q—table
    def restore(self, Q_file='./q_table.npy'):
        self.Q = np.load(Q_file)
        print(Q_file + 'loaded.')
