#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/15 下午5:09
# @Site    : 
# @File    : gym_snake.py
# @Software: PyCharm
import numpy as np
import gym
from gym.spaces import Discrete


class SnakeEnv(gym.Env):
    SIZE = 100

    # 传入梯子数量和不同投掷色子方法的最大值
    def __init__(self, ladder_num, dices):
        self.ladder_num = ladder_num
        self.dices = dices
        # dict存储梯子相连的两个格子之间的关系
        self.ladders = dict(np.random.randint(1, self.SIZE, size=(self.ladder_num, 2)))
        self.observation_space = Discrete(self.SIZE + 1)
        self.action_space = Discrete(len(dices))

        # list保存可能投掷色子的最大值
        for k, v in list(self.ladders.items()):
            self.ladders[v] = k
            print('ladders info:')
            print(self.ladders)
            print('dice ranges:')
            print(self.dices)
        self.pos = 1

    def reset(self):
        self.pos = 1
        return self.pos

    # 完成一次投掷，完成位置更新后，将返回玩家的新位置，得分和其他信息
    def step(self, action):
        # 返回一个随机整型数，范围从低（包括）到高（不包括），即[low, high)
        step = np.random.randint(1, self.dices[action] + 1)
        self.pos += step
        if self.step == 100:
            return 100, 100, 1, {}
        elif self.pos > 100:
            self.pos = 200 - self.pos

        if self.pos in self.ladders:
            self.pos = self.ladders[self.pos]
        return self.pos, -1, 0, {}

    def reward(self, s):
        if s == 100:
            return 100
        else:
            return -1

    def render(self):
        pass


if __name__ == '__main__':
    env = SnakeEnv(10, [1, 3])
    env.reset()
    while True:
        state, reward, terminate, _ = env.step(0)
        print(reward, state)
        if terminate == 1:
            break
