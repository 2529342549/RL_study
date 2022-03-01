#!/usr/bin/env python

import numpy as np
import time
import tkinter as tk
import warnings

warnings.filterwarnings('ignore')
from PIL import ImageTk, Image

np.random.seed(1)
PhotoImage = ImageTk.PhotoImage
UNIT = 30  # pixels
HEIGHT = 21  # grid height
WIDTH = 21  # grid width


# noinspection PyTypeChecker
class Env(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('PPO for Maze3')
        self.geometry('{0}x{1}'.format(HEIGHT * UNIT, HEIGHT * UNIT))
        self.shapes = self.load_images()
        self.canvas = self._build_canvas()
        self.texts = []
        self.act = 0

    def start_env(self):
        self.migong = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        self.x1, self.y1 = 10, 10
        return self.migong

    def _build_canvas(self):
        canvas = tk.Canvas(self, bg='white', height=HEIGHT * UNIT, width=WIDTH * UNIT)
        # create grids
        for c in range(0, WIDTH * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = c, 0, c, HEIGHT * UNIT
            canvas.create_line(x0, y0, x1, y1)
        for r in range(0, HEIGHT * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = 0, r, HEIGHT * UNIT, r
            canvas.create_line(x0, y0, x1, y1)

        # add img to canvas
        self.circle = canvas.create_image(225, 375, image=self.shapes[2])
        # print self.circle
        self.rectangle = canvas.create_image(315, 315, image=self.shapes[0])
        self.triangle1 = canvas.create_image(195, 195, image=self.shapes[1])
        self.triangle2 = canvas.create_image(195, 435, image=self.shapes[1])
        self.triangle3 = canvas.create_image(435, 195, image=self.shapes[1])
        self.triangle4 = canvas.create_image(435, 435, image=self.shapes[1])
        self.triangle5 = canvas.create_image(255, 345, image=self.shapes[3])
        self.triangle6 = canvas.create_image(225, 345, image=self.shapes[3])
        self.triangle7 = canvas.create_image(345, 225, image=self.shapes[3])
        self.triangle8 = canvas.create_image(345, 255, image=self.shapes[3])

        # pack all
        canvas.pack()

        return canvas

    def load_images(self):
        rectangle = PhotoImage(Image.open("../../data/img/rectangle.png").resize((20, 20)))
        triangle = PhotoImage(Image.open("../../data/img/triangle.png").resize((20, 20)))
        circle = PhotoImage(Image.open("../../data/img/circle.png").resize((20, 20)))
        yellow_rectangle = PhotoImage(Image.open("../../data/img/YellowRectangle.png").resize((20, 20)))
        return rectangle, triangle, circle, yellow_rectangle

    def coords_to_state(self, coords):
        x = int((coords[0] - 15) / 30)
        y = int((coords[1] - 15) / 30)
        return [x, y]

    def state_to_coords(self, state):
        x = int(state[0] * 30 + 15)
        y = int(state[1] * 30 + 15)
        return [x, y]

    def reset(self):
        self.update()
        # time.sleep(1)

        x, y = self.canvas.coords(self.rectangle)
        # print(x, y)
        self.canvas.move(self.rectangle, UNIT / 2 - x + 300, UNIT / 2 - y + 300)
        self.render()
        self.total_x = 0
        self.total_y = 0
        self.start_env()
        # return observation
        res_state = np.array(self.coords_to_state(self.canvas.coords(self.rectangle)))
        return res_state

    def render(self):
        time.sleep(0.003)
        self.update()

    def get_state(self):
        data = np.ravel(self.migong, order='C')

        return data

    def step(self, action):
        state = self.canvas.coords(self.rectangle)
        base_action = np.array([0, 0])
        self.render()
        # print('当前状态:', self.x1, self.y1)
        if action == 0:  # up
            if state[1] > UNIT:
                base_action[1] -= UNIT
                self.y1 -= 1
        elif action == 1:  # down
            if state[1] < (HEIGHT - 1) * UNIT:
                base_action[1] += UNIT
                self.y1 += 1
        elif action == 2:  # left
            if state[0] > UNIT:
                base_action[0] -= UNIT
                self.x1 -= 1

        elif action == 3:  # right
            if state[0] < (WIDTH - 1) * UNIT:
                base_action[0] += UNIT
                self.x1 += 1

        # move agent
        self.canvas.move(self.rectangle, base_action[0], base_action[1])
        # move rectangle to top level of canvas
        self.canvas.tag_raise(self.rectangle)
        # next_state = self.canvas.coords(self.rectangle)
        # next_state = [self.x1, self.y1]
        # print next_state
        # _state = self.coords_to_state(next_state)

        _state = [self.x1, self.y1]
        # print('下一状态：', _state)

        # reward function
        # print(self.migong[_state[0]][_state[1]])
        if self.migong[_state[0]][_state[1]] == 2:
            reward = 2
            done = True
        elif self.migong[_state[0]][_state[1]] == 3:
            reward = -0.5
            done = True
        elif _state[0] in [6, 14] or _state[1] in [6, 14]:
            reward = -0.5
            done = True
        else:
            reward = 0
            done = False

        return np.array(_state), reward, done
