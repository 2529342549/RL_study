#!/usr/bin/env python
# encoding: utf-8
"""
@author: HHD
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 2529342549@qq.com
@software: pycharm
@file: main_maze2.py
@time: 2022/2/9 下午8:24
@desc:
"""
# 主函数：简化ppo 这里先交互T_horizon个回合然后停下来学习训练，再交互，这样循环10000次
import random

import gym
import torch
from matplotlib import pyplot as plt
from past.builtins import raw_input
from torch.distributions import Categorical
from logger import Log
from ppo_maze_3 import  PPO
from env_3 import Env

logging = Log(__name__).getlog()


def main():
    # 创建环境
    env = Env()
    model = PPO()
    score = 0.0
    running_rewards = 20
    rewards = []
    gamma = 0.9
    # 主循环
    for n_epi in range(10000):
        state = env.reset()
        done = False
        # print(state)
        while not done:
            env.render()
            # print(state)
            # 由当前policy模型输出最优action
            # print(type(state), state)
            prob = model.pi(torch.from_numpy(state).float())
            m = Categorical(prob)
            action = m.sample()
            # print(prob, action)
            # 用最优action进行交互# noinspection PyTypeChecker
            state_prime, r, done = env.step(action)
            # print(r)
            # 存储交互数据，等待训练
            model.put_data((state, action, r, state_prime, prob[action].item(), done))
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
        # score = score * 0.99 + t * 0.01
        logging.info("# of episode :{}, avg score : {:.1f}".format(n_epi, score))
        score = 0

    # env.close()


if __name__ == '__main__':
    key = raw_input('Output directory already exists! Overwrite the folder? (y/n)')
    if key == 'y':
        with open(r'data/output.log', 'a+', ) as test:
            test.truncate(0)

    main()
