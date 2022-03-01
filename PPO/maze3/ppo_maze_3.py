#!/usr/bin/env python
# encoding: utf-8
"""
@author: HHD
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 2529342549@qq.com
@software: pycharm
@file: ppo.py
@time: 2022/2/9 下午8:22
@desc:
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Hyperparameters
learning_rate = 0.001  # 学习率
gamma = 0.95  #
lmbda = 0.95
eps_clip = 0.2
K_epoch = 20



# 定义PPO架构
class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []  # 用来存储交互数据

        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_pi = nn.Linear(64, 4)  # policy函数（输出action）的全连接层
        self.fc_v = nn.Linear(64, 1)  # value函数（输出v）的全连接层
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)  # 优化器

    # policy函数
    # 输入观测值x
    # 输出动作空间概率，从而选择最优action
    def pi(self, x, softmax_dim=0):
        # print(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    # value函数
    # 输入观测值x
    # 输出x状态下value的预测值（reward）,提供给policy函数作为参考值
    def v(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        v = self.fc_v(x)
        return v

    # 把交互数据存入buffer
    def put_data(self, transition):
        self.data.append(transition)

    # 把数据形成batch，训练模型时需要一个一个batch输入模型
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        state = torch.tensor(np.array(s_lst), dtype=torch.float)
        action = torch.tensor(a_lst)
        reward = torch.tensor(r_lst)
        state_prime = torch.tensor(np.array(s_prime_lst), dtype=torch.float)
        done_mask = torch.tensor(done_lst, dtype=torch.float)
        prob_a = torch.tensor(prob_a_lst)
        self.data = []
        return state, action, reward, state_prime, done_mask, prob_a

    # 训练模型
    def train_net(self):
        # make batch 数据，喂给模型
        state, action, reward, state_prime, done_mask, prob_a = self.make_batch()
        for i in range(K_epoch):  # K_epoch：训练多少个epoch
            # 计算td_error 误差，value模型的优化目标就是尽量减少td_error
            td_target = reward + gamma * self.v(state) * done_mask
            delta = td_target - self.v(state)
            delta = delta.detach().numpy()

            # 计算advantage:
            # 即当前策略比一般策略（baseline）要好多少
            # policy的优化目标就是让当前策略比baseline尽量好，但是每次更新时又不能偏离太多，所以后面会有个clip
            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            # 计算ratio 防止单次更新偏离太多
            pi = self.pi(state, softmax_dim=1)
            pi_a = pi.gather(1, action)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            # 通过clip 保证ratio在（1-eps_clip, 1+eps_clip）范围内
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
            # 这里简化ppo，把policy loss和value loss放在一起计算
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(state), td_target.detach())
            # print(loss)
            # 梯度优化
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
