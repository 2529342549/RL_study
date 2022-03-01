#!/usr/bin/env python
# encoding: utf-8
"""
@author: HHD
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited. 
@contact: 2529342549@qq.com
@software: pycharm
@file: ppo_maze2.py
@time: 2022/2/11 上午10:47
@desc:
"""
import numpy as np

# import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import namedtuple
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from torch.utils.tensorboard import SummaryWriter

# Parameters

gamma = 0.99
render = False
seed = 1

# env = gym.make('CartPole-v0').unwrapped
# action个数为19，observation为115。
num_state = 2
num_action = 4

torch.manual_seed(seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
# env.seed(seed)
Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, 16)
        self.fc2 = nn.Linear(16, 8)
        self.action_head = nn.Linear(8, num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state, 16)
        self.fc2 = nn.Linear(16, 8)
        self.state_value = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.state_value(x)
        return value


class PPO(object):
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 50
    buffer_capacity = 100000
    batch_size = 32

    def __init__(self):
        super(PPO, self).__init__()
        self.actor_net = Actor()
        self.critic_net = Critic()
        self.writer = SummaryWriter('../data/output')
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.load_models = False
        # self.load_ep = 210
        # SummaryWriter 类提供了一个高级 API，用于在给定目录中创建事件文件并向其中添加摘要和事件。该类异步更新文件内容。这允许训练程序直接从训练循环调用方法将数据添加到文件中，而不会减慢训练速度。
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 1e-4)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), 3e-4)
        # Adam(Adaptive Moment Estimation)本质上是带有动量项的RMSprop，它利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率。它的优点主要在于经过偏置校正后，每一次迭代学习率都有个确定范围，使得参数比较平稳。
        if self.load_models:
            load_model1 = torch.load("/home/hhd/PycharmProjects/RL_study_/PPO/maze2/data/output/actor.pkl")
            load_model2 = torch.load("/home/hhd/PycharmProjects/RL_study_/PPO/maze2/data/output/criter.pkl")
            self.actor_net.load_state_dict(load_model1)
            self.critic_net.load_state_dict(load_model2)
            # print("load model:", str(self.load_ep))

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_prob = self.actor_net(state)
        c = Categorical(action_prob)
        # c = Categorical(action_prob)
        action = c.sample()
        # print(action_prob, action)
        return action.item(), action_prob[:, action.item()].item()

    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        # print value.item()
        return value.item()

    def save_param(self, e):
        state = {'actor_net': self.actor_net.state_dict(), 'critic_net': self.critic_net.state_dict(),
                 'actor_optimizer': self.actor_optimizer.state_dict(), 'critic_optimizer': self.critic_net_optimizer,
                 'epoch': e}
        # torch.save(state, "/home/hhd/PycharmProjects/RL_study_/PPO/search_goal/model/" + str(e) + "a.pt")
        torch.save(self.actor_net.state_dict(),
                   "/home/hhd/PycharmProjects/RL_study_/PPO/maze2/data/output/" + "actor.pkl")
        torch.save(self.critic_net.state_dict(),
                   "/home/hhd/PycharmProjects/RL_study_/PPO/maze2/data/output/" + "criter.pkl")

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1

    def update(self, i_ep):
        # Transition(state, action, action_prob, reward, next_state)
        states = [t.state for t in self.buffer]
        state = torch.tensor(np.array(states), dtype=torch.float32)

        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1)
        reward = [t.reward for t in self.buffer]
        # print reward
        # update: don't need next_state
        # reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).view(-1, 1)
        # print reward
        # next_state = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float)
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1)
        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + gamma * R
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float)
        # print("The agent is updateing....")
        # print len(self.buffer)
        for i in range(self.ppo_update_time):
            # SubsetRandomSampler:从给定的指数列表中随机采样，不可以重复采样。
            # BatchSampler:Sampler采样得到的索引值进行合并，当数量等于一个batch大小后就将这一批的索引值返回
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
                # if self.training_step % 1000 == 0:
                #     print('I_ep {} ，train {} times'.format(i_ep, self.training_step))
                # with torch.no_grad():
                # view(-1, 1):-1表示一个不确定的数
                Gt_index = Gt[index].view(-1, 1)
                V = self.critic_net(state[index])
                # print   Gt, V
                delta = Gt_index - V
                advantage = delta.detach()
                # epoch iteration, PPO core!!!
                action_prob = self.actor_net(state[index]).gather(1, action[index])  # new policy
                # print(action_prob)
                # ratio = (action_prob / old_action_log_prob[index])
                ratio = (action_prob - old_action_log_prob[index]).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage
                # surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
                # print(action_loss)
                self.writer.add_scalar('loss/action_loss', action_loss, global_step=self.training_step)
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                # 梯度剪裁
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # update critic network
                value_loss = F.mse_loss(Gt_index, V)
                self.writer.add_scalar('loss/value_loss', value_loss, global_step=self.training_step)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.training_step += 1

        del self.buffer[:]  # clear experience
