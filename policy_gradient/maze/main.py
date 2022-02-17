#!/usr/bin/env python
# encoding: utf-8
"""
@author: HHD
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited. 
@contact: 2529342549@qq.com
@software: pycharm
@file: mian.py
@time: 2022/2/5 下午9:03
@desc:
"""

import numpy as np
from bg_plot import anim_plot
from agent import get_next_s, get_pi, update_theta, simple_convert_into_pi_from_theta


def goal_maze():
    s = 0  # 出发地点
    state_history = [[0, np.nan]]  # 移动的历史记录

    while (1):  #
        action, next_s = get_next_s(s)
        state_history[-1][1] = action

        state_history.append([next_s, np.nan])

        if next_s == 8:  #
            break
        else:
            s = next_s

    return state_history


if __name__ == "__main__":

    stop_epsilon = 10 ** -8

    for i in range(1000000):
        state_history = goal_maze()  # 移动的历史记录
        # print(state_history)
        # 更新参数
        update_theta(state_history)

        # 更新策略
        delta_pi = simple_convert_into_pi_from_theta()

        print(delta_pi)  # 策略的变化值
        print("求解迷宫问题本次走的步数：" + str(len(state_history) - 1))

        if np.sum(delta_pi) < stop_epsilon:
            print("一共进行的实验次数为：" + str(i))
            print("策略为：")
            # np 显示设置有效位数为8，不显示指数A
            np.set_printoptions(precision=8, suppress=True)
            print(get_pi())
            break
        # print(state_history)
        anim_plot([x[0] for x in state_history], method=0)
    print("求解迷宫问题本次走的步数：" + str(len(state_history) - 1))
