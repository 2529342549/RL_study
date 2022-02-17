#!/usr/bin/env python


import numpy as np
import time
import tkinter as tk
from PIL import ImageTk, Image

np.random.seed(1)
PhotoImage = ImageTk.PhotoImage
UNIT = 30  # pixels
HEIGHT = 21  # grid height
WIDTH = 21  # grid width


class Env(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('Q Learning')
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
                       [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        self.x1, self.y1 = 10, 10
        # print(self.migong[self.x1][self.y1])
        self.end_game = 0
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
        self.rectangle = canvas.create_image(315, 345, image=self.shapes[0])
        self.triangle1 = canvas.create_image(195, 195, image=self.shapes[1])
        self.triangle2 = canvas.create_image(195, 435, image=self.shapes[1])
        self.triangle3 = canvas.create_image(435, 195, image=self.shapes[1])
        self.triangle4 = canvas.create_image(435, 435, image=self.shapes[1])
        self.circle = canvas.create_image(255, 225, image=self.shapes[2])
        self.yellow_rectangle1 = canvas.create_image(255, 315, image=self.shapes[1])
        self.yellow_rectangle2 = canvas.create_image(285, 315, image=self.shapes[1])
        self.yellow_rectangle3 = canvas.create_image(345, 225, image=self.shapes[1])
        self.yellow_rectangle4 = canvas.create_image(345, 255, image=self.shapes[1])

        # pack all
        canvas.pack()

        return canvas

    def load_images(self):
        rectangle = PhotoImage(Image.open("/home/hhd/PycharmProjects/RL_study_/Qlearning_maze/img/rectangle.png").resize((20, 20)))
        triangle = PhotoImage(Image.open("/home/hhd/PycharmProjects/RL_study_/Qlearning_maze/img/triangle.png").resize((20, 20)))
        circle = PhotoImage(Image.open("/home/hhd/PycharmProjects/RL_study_/Qlearning_maze/img/circle.png").resize((20, 20)))
        yellow_rectangle = PhotoImage(
            Image.open("/home/hhd/PycharmProjects/RL_study_/Qlearning_maze/img/YellowRectangle.png").resize((20, 20)))
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
        time.sleep(0.03)
        self.update()

    def get_state(self):
        data = np.ravel(self.migong, order='C')

        return data

    def step(self, action):
        state = self.canvas.coords(self.rectangle)
        base_action = np.array([0, 0])
        self.render()

        if action == 0:  # up
            if state[1] > UNIT:
                base_action[1] -= UNIT
                self.migong[self.x1][self.y1] = 0
                self.migong[self.x1][self.y1 - 1] = 1
                self.y1 -= 1
        elif action == 1:  # down
            if state[1] < (HEIGHT - 1) * UNIT:
                base_action[1] += UNIT
                self.migong[self.x1][self.y1] = 0
                self.migong[self.x1][self.y1 + 1] = 1
                self.y1 += 1
        elif action == 2:  # left
            if state[0] > UNIT:
                base_action[0] -= UNIT
                self.migong[self.x1][self.y1] = 0
                self.migong[self.x1 - 1][self.y1] = 1
                self.x1 -= 1

        elif action == 3:  # right
            if state[0] < (WIDTH - 1) * UNIT:
                base_action[0] += UNIT
                self.migong[self.x1][self.y1] = 0
                self.migong[self.x1 + 1][self.y1] = 1
                self.x1 += 1

        # move agent
        self.canvas.move(self.rectangle, base_action[0], base_action[1])
        # move rectangle to top level of canvas
        self.canvas.tag_raise(self.rectangle)
        next_state = self.canvas.coords(self.rectangle)
        # print next_state
        _state = self.coords_to_state(next_state)
        # print(_state)

        _circle = self.coords_to_state(self.canvas.coords(self.circle))
        # reward function

        if next_state == self.canvas.coords(self.circle):
            reward = 20
            done = True
        elif next_state in [self.canvas.coords(self.triangle1), self.canvas.coords(self.triangle2),
                            self.canvas.coords(self.triangle3), self.canvas.coords(self.triangle4),
                            self.canvas.coords(self.yellow_rectangle1), self.canvas.coords(self.yellow_rectangle2),
                            self.canvas.coords(self.yellow_rectangle3), self.canvas.coords(self.yellow_rectangle4)]:
            # print 'coll'
            reward = -20
            done = True
        # elif
            # print self.migong
            # reward = -10
            # done = True
        else:
            reward = -0.1
            done = False

        # next_state = self.coords_to_state(next_state)
        next_state = np.array(self.coords_to_state(next_state))
        # state = self.coords_to_state(state)
        # print(state, next_state, [self.x1, self.y1])
        # print self.get_state()
        return self.get_state(), reward, done
