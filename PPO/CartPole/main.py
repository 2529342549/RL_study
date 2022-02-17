#!/usr/bin/env python
# encoding: utf-8
"""
@author: HHD
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited. 
@contact: 2529342549@qq.com
@software: pycharm
@file: main.py
@time: 2022/2/9 下午8:24
@desc:
"""
# 主函数：简化ppo 这里先交互T_horizon个回合然后停下来学习训练，再交互，这样循环10000次
import gym
import torch
from matplotlib import pyplot as plt
from torch.distributions import Categorical

from ppo import T_horizon, PPO


def plot(steps):
    ax = plt.subplot(111)
    ax.cla()
    ax.grid()
    ax.set_title('Training')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Run time')
    ax.plot(steps)
    RunTime = len(steps)

    # path = './AC_CartPole-v0/' + 'RunTime' + str(RunTime) + '.jpg'
    # if len(steps) % 200 == 0:
    #     plt.savefig(path)
    plt.pause(0.01)


def main():
    # 创建倒立摆环境
    env = gym.make('CartPole-v0')
    model = PPO()
    score = 0.0
    running_rewards = 20
    rewards = []
    gamma = 0.9
    # 主循环
    for n_epi in range(10000):
        state = env.reset()
        done = False

        while not done:
            env.render()
            for t in range(T_horizon):
                # 由当前policy模型输出最优action
                print(type(state))
                prob = model.pi(torch.from_numpy(state).float())
                m = Categorical(prob)
                action = m.sample().item()
                # print(prob, action)
                # 用最优action进行交互
                state_prime, r, done, info = env.step(action)
                # print(r)
                # 存储交互数据，等待训练
                model.put_data((state, action, r / 100.0, state_prime, prob[action].item(), done))
                state = state_prime

                score += r
                if done:
                    break
            # print(score)
            # running_rewards = running_rewards * 0.99 + t * 0.01
            # rewards.append(running_rewards)
            # plot(rewards)
            # 模型训练
            model.train_net()
        # print(score)
        # 打印每轮的学习成绩
        if n_epi % running_rewards == 0 and n_epi != 0:
            # print("# of episode :{}, avg score : {:.1f}".format(n_epi, score / running_rewards))
            score = 0.0

    env.close()


if __name__ == '__main__':
    main()
