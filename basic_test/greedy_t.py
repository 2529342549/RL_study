#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/16 上午9:16
# @Site    : 
# @File    : greedy_t.py
# @Software: PyCharm
import numpy as np

q_value = None


# epsilon-贪婪算法
def epsilon_greedy(nA, R, T, epsilon=0.6):
    # 初始化奖励
    r = 0
    N = [0] * nA

    for _ in range(T):
        if np.random.rand() < epsilon:
            # 探索阶段，以均匀分布随机选择
            a = np.random.randint(q_value.shape[0])
        else:
            # 利用阶段，选择价值函数最大的动作
            a = np.argmax(q_value[:])

        # 更新累积奖励和价值函数
        v = R(a)
        r = r + v

        q_value[a] = (q_value[a] * N[a] + v) / (N[a] + 1)
        N[a] += 1

        # 返回累积奖励 r
        return r
