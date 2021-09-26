#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/16 上午9:42
# @Site    : 
# @File    : gym_hello_grid.py
# @Software: PyCharm

import numpy as np
import sys

from six import StringIO, b
from gym import utils
from gym.envs.toy_text import discrete

# 所有状态都有四个动作
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class HelloGridEnv(discrete.DiscreteEnv):
    # HelloGrid环境
    MAPS = {'4x4': ["SOOO", "OXOX", "OOOX", "XOOG"]}
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, des=None, map_name='4x4'):
        """
        环境构造
        :param des:
        :param map_name:
        """
        if des is None:
            # 环境地图
            # self.des = np.asarray(MAPS[map_name], dtype='c')
            self.des = self.MAPS[map_name]
            # 获取MAPS形状（4，4）
        self.shape = des.shape

        # 动作集个数
        nA = 4
        # 状态集个数
        nS = np.prod(des.shape)

        # 设置最大的行号和列号用于索引
        MAX_X = self.shape[1]
        MAX_Y = self.shape[0]

        # 初始状态分布[1. 0. 0. ...]，并从格子S开始执行
        isd = np.array(des == b'S').astype('float64').ravel()

        isd /= isd.sum()

        # 动作状态转换概率字典
        P = {}

        """
        更新动作-状态转换概率字典P
        """
        # 对grid进行遍历
        state_grid = np.arange(nS).reshape(des.shape)
        it = np.nditer(state_grid, flags=['multi_index'])

        #
        while not it.finished:
            # 获取当前的state
            s = it.iterindex
            # 获取当前壮态所在格子的值
            y, x = it.multi_index

            # P[s][a]==[(probability,next_state,reward,done)*4]
            P[s] = {a: [] for a in range(nA)}

            s_letter = des[y][x]
            # 使用lambda表达式代替函数
            is_done = lambda letter: letter in b'GX'
            # 到达位置G才有奖励
            reward = 1.0 if s_letter in b'G' else -1.0

            if is_done(s_letter):
                # 如果达到状态G，直接更新状态-动作转换函数
                P[s][UP] = [(1.0, s, reward, True)]
                P[s][RIGHT] = [(1.0, s, reward, True)]
                P[s][DOWN] = [(1.0, s, reward, True)]
                P[s][LEFT] = [(1.0, s, reward, True)]
            else:
                # 如果没有达到状态G
                # 索引新状态位置
                ns_up = s if y == 0 else s - MAX_X
                ns_right = s if x == (MAX_Y - 1) else s + 1
                ns_down = s if y == (MAX_Y - 1) else s + MAX_X
                ns_left = s if x == 0 else s - 1

                # 新状态位置的索引对应的字母
                sl_up = des[ns_up // MAX_Y][ns_up % MAX_X]
                sl_right = des[ns_right // MAX_Y][ns_right % MAX_X]
                sl_down = des[ns_down // MAX_Y][ns_down % MAX_X]
                sl_left = des[ns_left // MAX_Y][ns_left % MAX_X]

                # 更新状态-动作转换概率
                P[s][UP] = [(1.0, ns_up, reward, is_done(sl_up))]
                P[s][RIGHT] = [(1.0, ns_right, reward, is_done(sl_right))]
                P[s][DOWN] = [(1.0, ns_down, reward, is_done(sl_down))]
                P[s][LEFT] = [(1.0, ns_left, reward, is_done(sl_left))]

            # 更新下一个状态
            it.iternext()

            self.P = P
            super(HelloGridEnv, self).__init__(nS, nA, P, isd)

    # 渲染HelloGridEnv环境
    def _render(self, mode='human', close=False):
        # 判断程序是否结束
        if close:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout

        # 格式转换
        des = self.des.tolist()
        des = [[c.decode('utf-8') for c in line] for line in des]
        state_grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(state_grid, flags=['multi_index'])

        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            if self.s == s:
                des[y][x] = utils.colorize(des[y][x], "red", highlight=True)
            it.iternext()

        outfile.write("\n".join(''.join(line) for line in des) + '\n')

        if mode != 'human':
            return outfile


if __name__ == '__main__':
    env = HelloGridEnv()
    state = env.reset()

    for _ in range(5):
        # 显示环境
        env.render()
        # 随机获得动作
        action = env.action_space.sample()
        # 执行选取的动作
        state, reward, done, info = env.step(action)

        print("action:{}({})".format(action, ["Up", "Right", "Down", "Left"][action]))
        print("done:{},observation:{},reward:{}".format(done, state, reward))

        # 如果执行之后返回的done状态为True则停止继续执行
        if done:
            print("Episode finished after {} timesteps".format(_ + 1))
            break
