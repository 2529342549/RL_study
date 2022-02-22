# coding=utf-8
"""
Path tracking simulation with pure pursuit steering control and PID speed control.
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from path import fig1

# 前视距离系数
k = 0.1
# 前视距离
Lfc = 0.6
# 速度P控制器系数
Kp = 0.1
# 时间间隔，单位：s
dt = 0.01
# 车辆轴距，单位：m
L = 1

show_animation = True


class State:
    """
    设置车辆的当前位置 (x,y) ,车辆的偏航角度yaw, 车辆的速度v
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v


def update(state, va, delta):
    # 定义车辆的状态更新函数
    state.x = state.x + state.v * math.cos(state.yaw) * dt
    state.y = state.y + state.v * math.sin(state.yaw) * dt
    state.yaw = state.yaw + state.v / L * math.tan(delta) * dt
    state.v = state.v + va * dt

    return state


def PIDControl(target, current):
    # 纵向控制使用一个简单的P控制器
    va = Kp * (target - current)

    return va


def pure_pursuit_control(state, cx, cy, pind):
    # 横向控制（即转角控制）使用纯追踪控制器
    ind = calc_target_index(state, cx, cy)

    if pind >= ind:
        ind = pind

    if ind < len(cx):
        tx = cx[ind]
        ty = cy[ind]
    else:
        tx = cx[-1]
        ty = cy[-1]
        ind = len(cx) - 1

    alpha = math.atan2(ty - state.y, tx - state.x) - state.yaw
    if state.v < 0:  # back
        alpha = math.pi - alpha
    # 预瞄距离
    Ld = k * state.v + Lfc
    # 转角
    delta = math.atan2(2.0 * L * math.sin(alpha) / Ld, 1.0)

    return delta, ind


def calc_target_index(state, cx, cy):
    # 搜索最临近的路点
    dx = [state.x - icx for icx in cx]
    dy = [state.y - icy for icy in cy]
    d = [abs(math.sqrt(idx ** 2 + idy ** 2)) for (idx, idy) in zip(dx, dy)]
    ind = d.index(min(d))
    L = 0.0

    Ld = k * state.v + Lfc

    # 搜索最近点前视距离
    while Ld > L and (ind + 1) < len(cx):
        dx = cx[ind + 1] - cx[ind]
        dy = cy[ind + 1] - cy[ind]
        L += math.sqrt(dx ** 2 + dy ** 2)
        ind += 1

    return ind


def main():
    # 最大模拟时间
    T = 100.0
    #  设置目标路点
    cx = np.arange(0, 25, 0.1)
    cy = [5 * math.sin(0.4 * ix) for ix in cx]

    A = 5
    gamma = np.linspace(0, 2 * np.pi)

    # cx = -A * gamma * np.cos(gamma)
    # cy = A * gamma * np.sin(gamma)

    # ax = 2 * A * np.cos(gamma)
    # ay = A * np.sin(2 * gamma)
    # cx.append(ax)
    # cy.append(ay)
    # print ax,ay
    # cx.append(ax)
    # cy.append(ay)
    # cx, cy = fig1()

    target_speed = 50  # [m/s]
    # print cx,cy

    # 设置车辆的初始状态
    state = State(x=0.0, y=0, yaw=0.0, v=0.0)

    lastIndex = len(cx) - 1
    time = 0.0
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [state.v]
    t = [0.0]
    target_ind = calc_target_index(state, cx, cy)

    while T >= time and lastIndex > target_ind:
        va = PIDControl(target_speed, state.v)
        delta, target_ind = pure_pursuit_control(state, cx, cy, target_ind)
        state = update(state, va, delta)

        time = time + dt

        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)

        if show_animation:
            plt.cla()
            plt.plot(cx, cy, ".r", label="course")
            plt.plot(x, y, "-b", label="trajectory")
            plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
            plt.axis("equal")
            # plt.grid(True)
            plt.title("Speed[km/h]:" + str(state.v)[:4])
            plt.pause(0.001)

    # Test
    # assert lastIndex >= target_ind, "Cannot goal"

    # if show_animation:
    #     plt.plot(cx, cy, ".r", label="course")
    #     plt.plot(x, y, "-b", label="Path")
    #     plt.legend()
    #     plt.xlabel("x[m]")
    #     plt.ylabel("y[m]")
    #     plt.axis("equal")
    #     plt.grid(True)
    #
    #     flg, ax = plt.subplots(1)
    #     plt.plot(t, [iv * 3.6 for iv in v], "-r")
    #     plt.xlabel("Time[s]")
    #     plt.ylabel("Speed[km/h]")
    #     plt.grid(True)
    plt.show()


if __name__ == '__main__':
    print("Pure pursuit path tracking simulation start")
    main()
