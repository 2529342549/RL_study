#!/usr/bin/env python
# encoding: utf-8
"""
@author: HHD
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited. 
@contact: 2529342549@qq.com
@software: pycharm
@file: agent.py
@time: 2022/2/5 下午9:01
@desc:
"""

import numpy as np
from environment import theta_0 as theta_1

theta_0 = theta_1.copy()

# _pi 为agent的策略, 模块内的全局变量
_pi = None


def reset():
    global _pi, theta_0
    _pi = None
    theta_0 = theta_1.copy()

    # _pi 为agent的初始策略
    # _pi=
    simple_convert_into_pi_from_theta()


def get_pi():
    return _pi


def simple_convert_into_pi_from_theta():
    '''theta_0 转换为 pi'''
    global _pi

    beta = 1.0
    [m, n] = theta_0.shape  #
    pi = np.zeros((m, n))

    exp_theta = np.exp(beta * theta_0)

    for i in range(0, m):
        pi[i, :] = exp_theta[i, :] / np.nansum(exp_theta[i, :])  #

    pi = np.nan_to_num(pi)  # nan

    if _pi is not None:
        # 不是模块的导入时候
        # 不是初始策略
        # _pi (old value)
        # pi (new value)
        delta = np.sum(np.abs(_pi - pi))
        _pi = pi
        return delta
    else:
        # 模块的导入时候
        # 初始策略
        _pi = pi


# _pi 为agent的初始策略
# _pi=
simple_convert_into_pi_from_theta()


def get_next_s(s):
    # 策略pi 为模块内的全局变量
    direction = ["up", "right", "down", "left"]

    next_direction = np.random.choice(direction, p=_pi[s, :])
    # _pi[s,:]

    if next_direction == "up":
        action = 0
        s_next = s - 3  # 上
    elif next_direction == "right":
        action = 1
        s_next = s + 1  # 右
    elif next_direction == "down":
        action = 2
        s_next = s + 3  # 下
    elif next_direction == "left":
        action = 3
        s_next = s - 1  # 左

    return [action, s_next]


def update_theta(s_a_history):
    global theta_0

    eta = 0.1  # 学习率
    T = len(s_a_history) - 1  # 总步数

    [m, n] = theta_0.shape  # theta shape值
    delta_theta = theta_0.copy()  # Δtheta  theta的copy

    # delta_theta
    for i in range(0, m):
        for j in range(0, n):
            if not (np.isnan(theta_0[i, j])):  # theta 中不为nan的部分

                SA_i = [SA for SA in s_a_history if SA[0] == i]
                # 探索路径中为i的状态

                SA_ij = [SA for SA in s_a_history if SA == [i, j]]
                # 探索路径中状态为i的动作为j的列表

                N_i = len(SA_i)  #
                N_ij = len(SA_ij)  #

                # （修正日：180703）
                # delta_theta[i, j] = (N_ij + _pi[i, j] * N_i) / T
                delta_theta[i, j] = (N_ij - _pi[i, j] * N_i) / T

    new_theta = theta_0 + eta * delta_theta

    # 更新 参数 theta_0
    theta_0 = new_theta
