#!/usr/bin/env python
# encoding: utf-8
"""
@author: HHD
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited. 
@contact: 2529342549@qq.com
@software: pycharm
@file: maze_run.py
@time: 2022/2/11 上午10:47
@desc:
"""
import argparse
import pickle
from collections import namedtuple
from itertools import count

import os, time, sys
import numpy as np

# import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
# from tensorboardX import SummaryWriter
import math, random
import time

# Parameters
from logger import Log
from environment_maze import Env

logging = Log(__name__).getlog()
gamma = 0.95
render = False
seed = 1
log_interval = 10

# env = gym.make('CartPole-v0').unwrapped
# action个数为19，observation为115。
num_state = 2
num_action = 4
env = Env()
torch.manual_seed(seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
# env.seed(seed)
Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, 256)
        self.fc1.weight.data.normal_(0, 0.1)  # 权重初始化 (均值为0，方差为0.1的正态分布)
        self.fc2 = nn.Linear(256, 256)
        self.fc2.weight.data.normal_(0, 0.1)
        self.action_head = nn.Linear(256, num_action)
        self.action_head.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state, 256)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(256, 256)
        self.fc2.weight.data.normal_(0, 0.1)
        self.state_value = nn.Linear(256, 1)
        self.state_value.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.state_value(x)
        return value


class PPO(object):
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 30
    buffer_capacity = 10000
    batch_size = 64

    def __init__(self):
        super(PPO, self).__init__()
        self.actor_net = Actor()
        self.critic_net = Critic()
        # self.writer = SummaryWriter('../exp')
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.load_models = False
        # self.load_ep = 210
        # self.writer = SummaryWriter('../exp')
        # SummaryWriter 类提供了一个高级 API，用于在给定目录中创建事件文件并向其中添加摘要和事件。该类异步更新文件内容。这允许训练程序直接从训练循环调用方法将数据添加到文件中，而不会减慢训练速度。
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 1e-3)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), 3e-3)
        # Adam(Adaptive Moment Estimation)本质上是带有动量项的RMSprop，它利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率。它的优点主要在于经过偏置校正后，每一次迭代学习率都有个确定范围，使得参数比较平稳。
        if self.load_models:
            load_model1 = torch.load("/home/hhd/PycharmProjects/RL_study_/PPO/maze2/data/actor.pkl")
            load_model2 = torch.load("/home/hhd/PycharmProjects/RL_study_/PPO/maze2/data/criter.pkl")
            self.actor_net.load_state_dict(load_model1)
            self.critic_net.load_state_dict(load_model2)
            # print("load model:", str(self.load_ep))

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_prob = self.actor_net(state)
        c = Categorical(action_prob)
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
                 'actor_optimizer': self.actor_optimizer.state_dict(), 'critic_optimizer': self.critic_net_optimizer, 'epoch': e}
        # torch.save(state, "/home/hhd/PycharmProjects/RL_study_/PPO/search_goal/model/" + str(e) + "a.pt")
        torch.save(self.actor_net.state_dict(), "/home/hhd/PycharmProjects/RL_study_/PPO/maze2/data/" + "actor.pkl")
        torch.save(self.critic_net.state_dict(), "/home/hhd/PycharmProjects/RL_study_/PPO/maze2/data/" + "criter.pkl")

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
                if self.training_step % 1000 == 0:
                    print('I_ep {} ，train {} times'.format(i_ep, self.training_step))
                # with torch.no_grad():
                # view(-1, 1):-1表示一个不确定的数
                Gt_index = Gt[index].view(-1, 1)
                V = self.critic_net(state[index])
                # print   Gt, V
                delta = Gt_index - V
                advantage = delta.detach()
                # epoch iteration, PPO core!!!
                action_prob = self.actor_net(state[index]).gather(1, action[index])  # new policy

                # ratio = (action_prob / old_action_log_prob[index])
                ratio = (action_prob - old_action_log_prob[index]).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
                # self.writer.add_scalar('loss/action_loss', action_loss, global_step=self.training_step)
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                # 梯度剪裁
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # update critic network
                value_loss = F.mse_loss(Gt_index, V)
                # self.writer.add_scalar('loss/value_loss', value_loss, global_step=self.training_step)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.training_step += 1

        del self.buffer[:]  # clear experience


def main():
    agent = PPO()
    running_reward = 10
    start_time = time.time()
    # env=Env()
    rewards = []
    for e in range(10000):
        state = env.reset()  # env.reset()函数用于重置环境

        # print(type(state), len(state))  # type:'numpy.ndarray' # len:364
        # print state
        # if render: env.render()#env.render()函数用于渲染出当前的智能体以及环境的状态
        episode_reward_sum = 0  # 初始化该循环对应的episode的总奖励
        episode_step = 6000
        # state = env.get_state()
        # print(state)
        for t in range(episode_step):
            env.render()
            action, action_prob = agent.select_action(state)
            # print action_prob, action
            # action_prob,action: tensor([[0.4578, 0.2056, 0.1278, 0.0028, 0.2060]]) tensor([1])
            next_state, reward, done = env.step(action)
            trans = Transition(state, action, action_prob, reward, next_state)
            # if render: env.render()
            agent.store_transition(trans)
            # print next_state
            state = next_state
            episode_reward_sum += reward
            if e % 50 == 0:
                agent.save_param(e)
            if t >= 1000:
                print('time out')
                done = True

            if done:
                agent.update(e)
                # print('Ep: %d score: %.2f memory: %d episode_step: %d time: %d:%02d:%02d' %
                #       (e, episode_reward_sum, agent.counter, t, h, m, s))
                logging.info('Ep: %d score: %.2f memory: %d' % (e, episode_reward_sum, agent.counter))
                break
        running_reward = running_reward * 0.99 + t * 0.01
        # rewards.append(running_reward)
        # plot(rewards)


if __name__ == '__main__':
    main()
    print("end")
