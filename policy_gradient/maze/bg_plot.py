#!/usr/bin/env python
# encoding: utf-8
"""
@author: HHD
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited. 
@contact: 2529342549@qq.com
@software: pycharm
@file: bg_plot.py
@time: 2022/2/5 下午9:02
@desc:
"""

import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

#################################模块内的全局变量定义
_fig, _line, _anim = None, None, None
_state_history = None


#################################函数定义部分###########################
def bg_plot():
    """background背景绘制"""
    global _fig, _line

    _fig = plt.figure(figsize=(5, 5))
    ax = plt.gca()

    plt.plot([1, 1], [0, 1], color='red', linewidth=2)
    plt.plot([1, 2], [2, 2], color='red', linewidth=2)
    plt.plot([2, 2], [2, 1], color='red', linewidth=2)
    plt.plot([2, 3], [1, 1], color='red', linewidth=2)

    plt.text(0.5, 2.5, 'S0', size=14, ha='center')
    plt.text(1.5, 2.5, 'S1', size=14, ha='center')
    plt.text(2.5, 2.5, 'S2', size=14, ha='center')
    plt.text(0.5, 1.5, 'S3', size=14, ha='center')
    plt.text(1.5, 1.5, 'S4', size=14, ha='center')
    plt.text(2.5, 1.5, 'S5', size=14, ha='center')
    plt.text(0.5, 0.5, 'S6', size=14, ha='center')
    plt.text(1.5, 0.5, 'S7', size=14, ha='center')
    plt.text(2.5, 0.5, 'S8', size=14, ha='center')
    plt.text(0.5, 2.3, 'START', ha='center')
    plt.text(2.5, 0.3, 'GOAL', ha='center')

    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    plt.tick_params(axis='both', which='both', bottom='off', top='off',
                    labelbottom='off', right='off', left='off', labelleft='off')

    _line, = ax.plot([0.5], [2.5], marker="o", color='g', markersize=60)


def bg_init():
    '''动画图像初始化'''
    _line.set_data([], [])
    return (_line,)


def animate(i):
    '''动画描绘内容'''
    state = _state_history[i]  #
    x = (state % 3) + 0.5  #
    y = 2.5 - int(state / 3)  #
    _line.set_data(x, y)
    return (_line,)


# 　动画效果初始化
def anim_plot(state_history, method=0):
    global _anim, _state_history
    _state_history = state_history
    _anim = animation.FuncAnimation(_fig, animate, init_func=bg_init, frames=len(
        _state_history), interval=200, repeat=False)
    if method == 0:
        print('zhans')
        # plt绘图方式
        plt.show()
    else:
        # HTML方式展示绘图
        HTML(_anim.to_jshtml())

    #################################模块内函数执行部分###########################


# 绘制背景
bg_plot()

# 显示的获得动画对象，没有_anim对象动画效果不会动（成为静态图）
# _anim对象必须是模块内的全局变量
# anim_plot 函数提供给主函数调用
# state_history=[0, 3, 6, 3, 4, 7, 4, 3, 0, 1, 0, 3, 6, 3, 4, 7, 4, 3, 6, 3, 4, 3, 0, 3, 4, 7, 8]
# anim_plot(state_history)

#
# if __name__ == "__main__":
#     state_history = [0, 3, 6, 3, 4, 7, 4, 3, 0, 1, 0, 3, 6, 3, 4, 7, 4, 3, 6, 3, 4, 3, 0, 3, 4, 7, 8]
#
#     # HTML方式展示绘图
#     # anim_plot(state_history, 1)
#     # plt绘图方式
#     anim_plot(state_history, 0)
