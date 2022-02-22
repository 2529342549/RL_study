# coding=utf-8
from math import atan, sqrt

import matplotlib.pyplot as plt
import numpy as np

# 速度
Va = 25
phi = 0.5
# Vx = Va * np.cos(phi)
# Vy = Va * np.sin(phi)
# Wa = (6, 65)
# Wb = (12, 35)
# 初始点
Wa = [6, 12, 23, 34, 54, 66]
Wb = [12, 20, 30, 40, 23, 45]
plt.plot(Wa, Wb, '-b')
# for point in waypoints:

Px = 12
Py = 10
##################
# gamma = np.linspace(0, 3 * np.pi)
# s1 = 0  # 弧长
# T = 0.2  # 采样周期
# vr = 2  # 速度
# A = 0.9
plt.plot(Px, Py, '-d')

# for i in range(len(waypoints)):
#     plt.plot(w1[i], w2[i],'-g')
################

# plt.plot((Wa[0], Px), (Wb[0], Py), '-g')

# plt.show()
delta = 0.4

# xt = -A * gamma * np.cos(gamma)
# yt = A * gamma * np.sin(gamma)
# plt.plot(xt, yt, '^r')
K = 1
K2 = 5
wp =1

dist = sqrt((Wb[1] - Py) ** 2 + (Wa[1] - Px) ** 2)
while (wp <= 5 and dist > delta):
    Ru = np.sqrt((Wb[0] - Py) ** 2 + (Wa[0] - Px) ** 2)
    theta = abs(atan(float(Wb[1] - Wb[0]) / float(Wa[1] - Wa[0])))
    thetaU = abs(atan((Py - Wb[0]) / (Px - Wa[0])))
    beta = abs(theta - thetaU)
    R = Ru * np.cos(beta)
    # cross-track eror
    e = Ru * np.sin(beta)
    xt = Wa[0] + (R + delta) * np.cos(theta)
    yt = Wb[0] + (R + delta) * np.sin(theta)
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
    # for x, y in zip(Wa,Wb):
    #     dist=sqrt((x-xt)**2+(y-yt)**2)
    #     if delta<dist:
    #         Wa[0]=x
    #         Wb[0]=y
    dist = sqrt((Wb[1] - yt) ** 2 + (Wa[1] - xt) ** 2)
    # print dist

    if dist < delta:
        # print 'er'
        # Wa = [6, 12, 23, 34, 54, 66]
        # Wb = [12, 20, 40, 40, 23, 45]
        print(dist)
        Wa[0] = Wa[wp]
        Wb[0] = Wb[wp]
        Wa[1] = Wa[wp + 1]
        Wb[1] = Wb[wp + 1]
        dist = sqrt((Wb[1] - yt) ** 2 + (Wa[1] - xt) ** 2)
        print(Wa[0], Wb[0], Wa[1], Wb[1])
        wp = wp + 1
    plt.plot(xt, yt, '^b')
    # print xt, yt

plt.show()
