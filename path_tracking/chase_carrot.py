# coding=utf-8
from math import atan

import matplotlib.pyplot as plt
import numpy as np

# 速度
Va = 25
phi = 0.5
Vx = Va * np.cos(phi)
Vy = Va * np.sin(phi)
Wa = (6, 65)
Wb = (12, 35)
# 初始点
Px = 5
Py = 32
plt.plot(Px, Py, 'pr')
plt.plot(Wa, Wb, '-d')

Ru = np.sqrt((Wb[0] - Py) ** 2 + (Wa[0] - Px) ** 2)
theta = abs(atan(float(Wb[1] - Wb[0]) / float(Wa[1] - Wa[0])))
thetaU = abs(atan((Py - Wb[0]) / (Px - Wa[0])))
beta = abs(theta - thetaU)
R = Ru * np.cos(beta)
# cross-track eror
e = Ru * np.sin(beta)

plt.plot((Wa[0], Px), (Wb[0], Py), '-g')

# plt.show()
delta = 5
# print theta, thetaU
xt = Wa[0] + (R + delta) * np.cos(theta)
yt = Wb[0] + (R + delta) * np.sin(theta)
plt.plot(xt, yt, '^r')
# print xt, yt
K = 0.5
K2 = 35
while abs(e) > 0:
    t = 0.05
    # print xt, yt, Px, Py
    # 偏移角
    phiD = abs(atan((yt - Py) / (xt - Px)))
    u = K * (phiD - phi) * Va - K2 * e
    if u > 1:
        u = 1
    phi = phiD
    Vy = Va * np.sin(phi) + u * t
    Vx = np.sqrt(Va * Va - Vy * Vy)
    # 位置
    Px = Px + Vx * t
    Py = Py + Vy * t

    t = t + 0.1
    plt.plot(Px, Py, 'pr ')
    plt.pause(0.1)
    # plt.show()
    # 距离原点的距离
    Ru = np.sqrt((Wb[0] - Py) ** 2 + (Wa[0] - Px) ** 2)
    # 距离原点的角度
    thetaU = abs(atan((Py - Wb[0]) / (Px - Wa[0])))

    beta = abs(theta - thetaU)
    R = Ru * np.cos(beta)
    # 轨迹偏移误差
    e = Ru * np.sin(beta)
    plt.plot((Wa[0], Px), (Wb[0], Py), '--g ')
    # carrot position
    xt = Wa[0] + (R + delta) * np.cos(theta)
    yt = Wb[0] + (R + delta) * np.sin(theta)
    plt.plot(xt, yt, '^b')
    # print xt, yt
plt.show()