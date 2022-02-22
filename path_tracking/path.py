# coding=utf-8
import time

import matplotlib.pyplot as plt
import numpy as np


def fig1():
    s1 = 0  # 弧长
    T = 0.005  # 采样周期
    vr = 10  # 速度
    A = 5
    gamma = np.linspace(0, 2 * np.pi)

    # x = np.linspace(0, 500, 100)  # 返回num均匀分布的样本，在[start, stop]。
    xd_i = 2 * A * np.cos(gamma)
    yd_i = A * np.sin(2 * gamma)

    fig = plt.figure()
    # [距离左边，下边，坐标轴宽度，坐标轴高度] 范围(0, 1)
    ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax1.plot(xd_i, yd_i, 'g')
    gamma = 0
    ax = []
    ay = []
    while gamma < 2 * np.pi:
        gamma = np.sqrt(2 * s1 / A)
        # print(gamma)
        xr = 2 * A * np.cos(gamma)
        ax.append(xr)

        yr = A * np.sin(2 * gamma)
        ay.append(yr)
        s1 = s1 + vr * T
        # print(s1)
        ax1.plot(xr, yr, 'or')
        plt.pause(0.01)
    #
    plt.show()
    return ax, ay


if __name__ == '__main__':
    fig1()
