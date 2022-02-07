#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/19 下午3:47
# @Site    : 
# @File    : layout.py
# @Software: PyCharm
import numpy as np
import tkinter as tk


class InitLayout(tk.Tk):
    # gridNum为格子数，gridWidth为每个格子宽度，objWidth为物体宽度
    def __init__(self, gridNum=7, gridWidth=80, objWidth=50):
        # 继承自父类的属性进行初始化
        super(InitLayout, self).__init__()
        self.title('机器人走迷宫')
        self.gridNum = gridNum
        self.gridWidth = gridWidth
        self.objWidth = objWidth
        self.borderSize = self.gridNum * self.gridWidth
        # 4种action
        self.action_space = ['up', 'down', 'right', 'left', 'zs', 'zx', 'ys', 'yx']
        self.actions_num = len(self.action_space)
        # 陷阱位置，格子位置索引
        self.blacks = [[0, 2], [1, 2], [2, 2], [2, 4], [3, 2], [3, 1], [3, 4], [3, 3], [3, 5], [4, 4],[5,2]]
        # 陷阱坐标集合
        self.blackCoors = []
        self.start_drawing()

    def start_drawing(self):
        # 初始化,Canvas显示和编辑图形
        self.drawing = tk.Canvas(self, height=self.borderSize, width=self.borderSize, bg='#c9ccd0')
        # 画线
        for col in range(0, self.borderSize, self.gridWidth):
            x0, y0, x1, y1 = col, 0, col, self.borderSize
            self.drawing.create_line(x0, y0, x1, y1)
        for row in range(0, self.borderSize, self.gridWidth):
            x0, y0, x1, y1 = 0, row, self.borderSize, row
            self.drawing.create_line(x0, y0, x1, y1)

        # 初始点
        start_pos = np.array([self.gridWidth / 2, self.gridWidth / 2])

        # 画陷阱
        for index, pos_tup in enumerate(self.blacks):
            col, row = pos_tup
            center_pos = start_pos + np.array([self.gridWidth * col, self.gridWidth * row])
            hell = self.drawing.create_oval(
                center_pos[0] - self.objWidth / 2, center_pos[1] - self.objWidth / 2,
                center_pos[0] + self.objWidth / 2, center_pos[1] + self.objWidth / 2,
                fill='black')
            # 追加到坐标集合中
            self.blackCoors.append(self.drawing.coords(hell))

        # 画终点
        dist_center = start_pos + np.array([self.gridWidth * 6, self.gridWidth * 6])
        self.dist = self.drawing.create_oval(
            dist_center[0] - self.objWidth / 2, dist_center[1] - self.objWidth / 2,
            dist_center[0] + self.objWidth / 2, dist_center[1] + self.objWidth / 2,
            fill='yellow'
        )

        # 画智能体
        self.rect = self.drawing.create_oval(
            start_pos[0] - self.objWidth / 2, start_pos[1] - self.objWidth / 2,
            start_pos[0] + self.objWidth / 2, start_pos[1] + self.objWidth / 2,
            fill='red'
        )

        self.drawing.pack()

    def reset(self):
        # 重新画智能体位置
        self.drawing.delete(self.rect)
        origin = np.array([self.gridWidth / 2, self.gridWidth / 2])
        self.rect = self.drawing.create_oval(
            origin[0] - self.objWidth / 2, origin[1] - self.objWidth / 2,
            origin[0] + self.objWidth / 2, origin[1] + self.objWidth / 2,
            fill='red'
        )

    def step(self, action):
        # 智能体移动
        s = self.drawing.coords(self.rect)
        # print('s:', s)
        agent_pos = np.array([0, 0])
        # print('agent_pos:', agent_pos)
        # 进行位置移动
        if action == 0:  # up
            if s[1] > self.gridWidth:
                agent_pos[1] -= self.gridWidth
        elif action == 1:  # down
            if s[1] < (self.gridNum - 1) * self.gridWidth:
                agent_pos[1] += self.gridWidth

        elif action == 2:  # left
            if s[0] > self.gridWidth:
                agent_pos[0] -= self.gridWidth
        elif action == 3:  # right
            if s[0] < (self.gridNum - 1) * self.gridWidth:
                agent_pos[0] += self.gridWidth
        elif action == 4:  # zs
            if s[1] > self.gridWidth and s[0] > self.gridWidth:
                agent_pos[0] -= self.gridWidth
                agent_pos[1] -= self.gridWidth
        elif action == 5:  # zx
            if s[0] > self.gridWidth and s[1] < (self.gridNum - 1) * self.gridWidth:
                agent_pos[0] -= self.gridWidth
                agent_pos[1] += self.gridWidth
        elif action == 6:  # ys
            if s[0] < (self.gridNum - 1) * self.gridWidth and s[1] > self.gridWidth:
                agent_pos[0] += self.gridWidth
                agent_pos[1] -= self.gridWidth
        elif action == 7:  # yx
            if s[0] < (self.gridNum - 1) * self.gridWidth and s[1] < (self.gridNum - 1) * self.gridWidth:
                agent_pos[0] += self.gridWidth
                agent_pos[1] += self.gridWidth

        # 移动图形
        self.drawing.move(self.rect, agent_pos[0], agent_pos[1])
        sig = self.drawing.coords(self.rect)

        # 到达终点
        if sig == self.drawing.coords(self.dist):
            reward = 100
            finished = True
            sig = 'terminal'
        # 掉进陷阱
        elif sig in self.blackCoors:
            reward = -100
            finished = True
            sig = 'terminal'
        else:
            reward = -1
            finished = False

        return sig, reward, finished

    def render(self):
        self.update()
