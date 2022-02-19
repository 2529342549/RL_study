#!/usr/bin/env python
# encoding: utf-8
"""
@author: HHD
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited. 
@contact: 2529342549@qq.com
@software: pycharm
@file: run.py
@time: 2022/2/8 下午12:22
@desc:
"""
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import gym
import math, random
import warnings

from distlib.compat import raw_input
from matplotlib import pyplot as plt
from env_maze_dqn import Env
from logger import Log

logging = Log(__name__).getlog()
warnings.filterwarnings('ignore')
sys.path.append('..')
# 超参数
BATCH_SIZE = 32  # 样本数量
LR = 0.001  # 学习率
EPISODES = 0.99  # greedy policy
GAMMA = 0.9  # reward discount
TARGET_REPLACE_ITER = 10  # 目标网络更新频率
MEMORY_CAPACITY = 2000  # 记忆库容量
N_ACTIONS = 4
env = Env()  # 使用gym库中的环境：CartPole，且打开封装(若想了解该环境，请自行百度)
N_STATES = 2  # 状态个数

"""
torch.nn是专门为神经网络设计的模块化接口。nn构建于Autograd之上，可以用来定义和运行神经网络。
nn.Module是nn中十分重要的类，包含网络各层的定义及forward方法。
定义网络：
    需要继承nn.Module类，并实现forward方法。
    一般把网络中具有可学习参数的层放在构造函数__init__()中。
    只要在nn.Module的子类中定义了forward函数，backward函数就会被自动实现(利用Autograd)。
"""


# 定义Net类 (定义网络)
class Net(nn.Module):
    def __init__(self):  # 定义Net的一系列属性
        # nn.Module的子类函数必须在构造函数中执行父类的构造函数
        super(Net, self).__init__()  # 等价与nn.Module.__init__()
        # N_STATES=28
        self.fc1 = nn.Linear(N_STATES, 128)  # 设置第一个全连接层(输入层到隐藏层): 状态数个神经元到50个神经元
        self.fc1.weight.data.normal_(0, 0.1)  # 权重初始化 (均值为0，方差为0.1的正态分布)
        self.fc2 = nn.Linear(128, 128)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(128, N_ACTIONS)  # 设置第二个全连接层(隐藏层到输出层): 50个神经元到动作数个神经元
        self.out.weight.data.normal_(0, 0.1)  # 权重初始化 (均值为0，方差为0.1的正态分布)

    def forward(self, x):  # 定义forward函数 (x为状态)
        x = x.view(x.size(0), -1)
        # print x.shape
        x = F.relu(self.fc1(x))  # 连接输入层到隐藏层，且使用激励函数ReLU来处理经过隐藏层后的值
        x = F.relu(self.fc2(x))
        x = F.dropout(self.fc2(x))
        actions_value = self.out(x)  # 连接隐藏层到输出层，获得最终的输出值 (即动作值)
        return actions_value  # 返回动作值


# 定义DQN类 (定义两个网络)
class DQN(object):
    def __init__(self):  # 定义DQN的一系列属性
        self.eval_net, self.target_net = Net(), Net()  # 利用Net创建两个神经网络: 评估网络和目标网络
        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, 6))  # 初始化记忆库，一行代表一个transition
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)  # 使用Adam优化器 (输入为评估网络的参数和学习率)
        self.loss_func = nn.MSELoss()  # 使用均方损失函数 (loss(xi, yi)=(xi-yi)^2)
        self.start_epoch = 3157
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.load_models = input("Is load model?")
        # self.load_models = True
        self.loss = 0
        self.q_eval = 0
        self.q_target = 0
        if self.load_models:
            self.epsilon = 0
            checkpoint = torch.load("data/" + "model.pt")
            # print(checkpoint.keys())
            # print(checkpoint['epoch'])
            # print(checkpoint)
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.eval_net.load_state_dict(checkpoint['eval_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            # self.start_epoch = checkpoint['epoch'] + 1
            # print("load model")

    def choose_action(self, x):  # 定义动作选择函数 (x为状态)
        x = torch.unsqueeze(torch.FloatTensor(x), 0)  # 将x转换成32-bit floating point形式，并在dim=0增加维数为1的维度
        # print x
        if np.random.uniform() > self.epsilon:  # 生成一个在[0, 1)内的随机数，如果小于EPSILON，选择最优动作
            actions_value = self.eval_net.forward(x)  # 通过对评估网络输入状态x，前向传播获得动作值
            action = torch.max(actions_value, 1)[1].data.numpy()  # 输出每一行最大值的索引，并转化为numpy ndarray形式
            action = action[0]  # 输出action的第一个数
        else:  # 随机选择动作
            action = np.random.randint(0, N_ACTIONS)  # 这里action随机等于0或1 (N_ACTIONS = 2)
        return action  # 返回选择的动作

    def store_transition(self, s, a, r, s_):  # 定义记忆存储函数 (这里输入为一个transition)
        transition = np.hstack((s, [a, r], s_))  # 在水平方向上拼接数组
        # print len(transition)
        # 如果记忆库满了，便覆盖旧的数据
        index = self.memory_counter % MEMORY_CAPACITY  # 获取transition要置入的
        # 行数
        # print index
        # print self.memory.shape
        # print transition.shape
        # print len(transition)
        self.memory[index, :] = transition  # 置入transition
        self.memory_counter += 1  # memory_counter自加1

    def learn(self):  # 定义学习函数(记忆库已满后便开始学习)
        # 目标网络参数更新
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:  # 一开始触发，然后每100步触发
            self.target_net.load_state_dict(self.eval_net.state_dict())  # 将评估网络的参数赋给目标网络
        self.learn_step_counter += 1  # 学习步数自加1

        # 抽取记忆库中的批数据
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)  # 在[0, 2000)内随机抽取32个数，可能会重复len(sample_index):32
        # print(len(sample_index))
        b_memory = self.memory[sample_index, :]  # 抽取32个索引对应的32个transition，存入b_memory ,shape:(32, 6)
        # print(b_memory.shape)
        # 将32个s抽出，转为32-bit floating point形式，并存储到b_s中，b_s为32行6列
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])  # b_s.shape: torch.Size([32, 6])
        # print(b_s.shape)
        # 将32个a抽出，转为64-bit integer (signed)形式，并存储到b_a中 (之所以为LongTensor类型，是为了方便后面torch.gather的使用)，b_a为32行1列
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int))
        # print(b_a)
        # 将32个r抽出，转为32-bit floating point形式，并存储到b_s中，b_r为32行1列
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])
        # 将32个s_抽出，转为32-bit floating point形式，并存储到b_s中，b_s_为32行4列
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # 获取32个transition的评估值和目标值，并利用损失函数和优化器进行评估网络参数更新
        # eval_net(b_s)通过评估网络输出32行每个b_s对应的一系列动作值，然后.gather(1, b_a)代表对每行对应索引b_a的Q值提取进行聚合
        q_eval = self.eval_net(b_s).gather(1, b_a)
        # print(q_eval)
        # q_next不进行反向传递误差，所以detach；q_next表示通过目标网络输出32行每个b_s_对应的一系列动作值
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        # q_next.max(1)[0]表示只返回每一行的最大值，不返回索引(长度为32的一维张量)；.view()表示把前面所得到的一维张量变成(BATCH_SIZE, 1)的形状；最终通过公式得到目标值
        loss = self.loss_func(q_eval, q_target)
        self.loss = torch.max(loss)
        self.q_eval = torch.max(q_eval)
        self.q_target = torch.max(q_target)
        # 输入32个评估值和32个目标值，使用均方损失函数
        self.optimizer.zero_grad()  # 清空上一步的残余更新参数值
        loss.backward()  # 误差反向传播, 计算参数更新值
        self.optimizer.step()  # 更新评估网络的所有参数

    def save_model(self):
        state = {'target_net': self.target_net.state_dict(), 'eval_net': self.eval_net.state_dict(),
                 'optimizer': self.optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, "/home/hhd/PycharmProjects/RL_study_/dqn/maze/data/" + "model.pt")


def plot(args):
    ax = plt.subplot(111)
    ax.cla()
    ax.grid()
    ax.set_title('Training')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Run time')
    ax.plot(args)
    RunTime = len(args)

    # path = './AC_CartPole-v0/' + 'RunTime' + str(RunTime) + '.jpg'
    # if len(steps) % 200 == 0:
    #     plt.savefig(path)
    plt.pause(0.0001)


if __name__ == '__main__':
    key = raw_input('Output directory already exists! Overwrite the folder? (y/n)')
    if key == 'y' :
        with open(r'data/output.log', 'a+', ) as test:
            test.truncate(0)
    dqn = DQN()  # 令dqn=DQN类
    start_time = time.time()
    # epoch = 2988
    for epoch in range(dqn.start_epoch,10000):  # 400个episode循环
        state = env.reset()  # 重置环境
        # print s
        episode_reward_sum = 0  # 初始化该循环对应的episode的总奖励
        done = False

        for t in range(6000):  # 开始一个episode (每一个循环代表一步)
            action = dqn.choose_action(state)  # 输入该步对应的状态s，选择动作
            # print a
            s_, reward, done = env.step(action)  # 执行动作，获得反馈
            # print s_
            dqn.store_transition(state, action, reward, s_)  # 存储样本
            episode_reward_sum += reward  # 逐步加上一个episode内每个step的reward
            state = s_  # 更新状态
            if dqn.memory_counter > BATCH_SIZE:  # 如果累计的transition数量超过了记忆库的固定容量2000
                # 开始学习 (抽取记忆，即32个transition，并对评估网络参数进行更新，并在开始学习后每隔100次将评估网络的参数赋给目标网络)
                dqn.learn()
            if epoch % 100 == 0:
                dqn.save_model()
            if t >= 2500:
                done = True
            if done:  # 如果done为True
                # round()方法返回episode_reward_sum的小数点四舍五入到2个数字
                m, s = divmod(int(time.time() - start_time), 60)
                h, m = divmod(m, 60)
                # print e
                logging.info('Ep: %d score: %.2f memory: %d epsilon: %.2f ' %
                             (epoch, episode_reward_sum, dqn.memory_counter, dqn.epsilon))
                # print('episode%s reward_sum: %s' % (e, round(episode_reward_sum, 2)))
                # param_keys = ['epsilon']
                # param_values = [dqn.epsilon]
                # param_dictionary = dict(zip(param_keys, param_values))
                break  # 该episode结束
            if dqn.epsilon > dqn.epsilon_min:
                dqn.epsilon = dqn.epsilon - 0.0001
