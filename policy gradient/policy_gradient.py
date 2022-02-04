#!/usr/bin/env python
# encoding: utf-8
"""
@author: HHD
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 2529342549@qq.com
@software: pycharm
@file: policy_gradient.py
@time: 2022/2/4 下午7:56
@desc:
"""

import argparse
import gym
import numpy as np
import torch
from itertools import count
from torch import nn, optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description="Pytorch REINFORCE example")
parser.add_argument('--gpu', default=True, action='store_true')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default:0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N', help='random seed (default:543)')
parser.add_argument('--render', action='store_true', default=True, help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='interval between training status logs (default:10)')
args = parser.parse_args()

env = gym.make('CartPole-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
print('Using device: %s', device)
writer = SummaryWriter("logs")


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.linear1 = nn.Linear(4, 128)
        self.linear2 = nn.Linear(128, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        action_scores = self.linear2(x)
        return F.softmax(action_scores, dim=1)


policy = Policy().to(device)
optimizer = optim.Adam(policy.parameters(), lr=0.01)
# eps是一个很小的非负数
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    state = state.to(device)
    # 获得概率
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    action = action.to(device)
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode():
    R = 0
    policy_loss = []
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.as_tensor(rewards, dtype=torch.float)
    rewards = (rewards - rewards.mean()) / (rewards.std(0) + eps)

    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.to(device)
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main():
    running_reward = 10
    for i_episode in count(1):
        state = env.reset()
        for t in range(10000):
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            if args.render:
                env.render()
            policy.rewards.append(reward)
            if done:
                break

        running_reward = running_reward * 0.99 + t * 0.01
        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(i_episode, t, running_reward))
        writer.add_scalar("running reward", running_reward, i_episode)
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()
