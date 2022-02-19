import tkinter as tk
import numpy as np
import time
import decimal as dm


class Env(tk.Tk):

    def __init__(self, render_speed=0.003, width=10, height=10):

        super(Env, self).__init__()
        self.render_speed = render_speed
        self.epi = 0
        self.steps = 1
        self.width = width
        self.height = height
        self.action_space = [0, 1, 2, 3]
        self.action_size = len(self.action_space)
        self.player_loc = [1, 1]
        self.before_player_loc = None
        #  0
        # wall 1
        # agent 2
        # goal 3
        # obstacle 4
        self.game_board = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                           [1, 2, 0, 0, 0, 4, 1, 0, 0, 1],
                           [1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                           [1, 0, 0, 0, 1, 0, 1, 0, 4, 1],
                           [1, 0, 1, 0, 4, 0, 0, 0, 0, 1],
                           [1, 0, 1, 0, 1, 0, 0, 0, 0, 1],
                           [1, 0, 1, 0, 1, 0, 0, 0, 0, 1],
                           [1, 0, 1, 0, 0, 0, 1, 0, 0, 1],
                           [1, 0, 0, 0, 0, 0, 1, 0, 3, 1],
                           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        self.reward = 0
        self.images = self._load_images()
        self.canvas = self._build_canvas()

    def _load_images(self):
        character = tk.PhotoImage(file="/home/hhd/PycharmProjects/RL_study_/PPO/maze_ppo/sprite/character.png")
        trap = tk.PhotoImage(file="/home/hhd/PycharmProjects/RL_study_/PPO/maze_ppo/sprite/box.png")
        mark = tk.PhotoImage(file="/home/hhd/PycharmProjects/RL_study_/PPO/maze_ppo/sprite/mark.png")
        block = tk.PhotoImage(file="/home/hhd/PycharmProjects/RL_study_/PPO/maze_ppo/sprite/block.png")
        trap_with_character = tk.PhotoImage(file="/home/hhd/PycharmProjects/RL_study_/PPO/maze_ppo/sprite/box and mark.png")
        mark_with_character = tk.PhotoImage(file="/home/hhd/PycharmProjects/RL_study_/PPO/maze_ppo/sprite/character and mark.png")

        return block, character, mark, trap, trap_with_character, mark_with_character

    def _build_canvas(self):
        pixel = 32
        canvas = tk.Canvas(self, bg='black', height=pixel * self.height + 50, width=pixel * self.width)
        # for i in range(0, pixel*self.height, 32):
        #     x0, y0, x1, y1 = i, 0, i, pixel*self.height
        #     canvas.create_line(x0, y0, x1, y1,fill = "white")
        #
        # for j in range(0, pixel*self.height, 32):
        #     x0, y0, x1, y1 = 0, j, pixel*self.height, j
        #     canvas.create_line(x0, y0, x1, y1,fill = "white")

        for j in range(self.height):
            for i in range(self.width):
                k = self.game_board[j][i]
                x = i * 32
                y = j * 32

                if k == 1:
                    canvas.create_image(x, y, anchor="nw", image=self.images[0])
                elif k == 2:
                    canvas.create_image(x, y, anchor="nw", image=self.images[1])
                elif k == 3:
                    canvas.create_image(x, y, anchor="nw", image=self.images[2])
                elif k == 4:
                    canvas.create_image(x, y, anchor="nw", image=self.images[3])
                elif k == 5:
                    canvas.create_image(x, y, anchor="nw", image=self.images[4])
                elif k == 6:
                    canvas.create_image(x, y, anchor="nw", image=self.images[5])

        texts = str(self.epi) + "episodes"
        canvas.create_text(175, 330, text=texts, font=('Helvetica', 10), fill="white")
        canvas.pack()
        return canvas

    def reset(self):
        self.update()
        self.steps = 1
        self.tot_reward = 0
        self.action_space = [0, 1, 2, 3]
        self.action_size = len(self.action_space)
        self.game_board = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                           [1, 2, 0, 0, 0, 0, 1, 0, 0, 1],
                           [1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                           [1, 0, 0, 0, 1, 0, 1, 0, 1, 1],
                           [1, 0, 1, 0, 1, 0, 0, 0, 0, 1],
                           [1, 0, 1, 0, 1, 0, 0, 0, 0, 1],
                           [1, 0, 1, 0, 1, 0, 0, 0, 0, 1],
                           [1, 0, 1, 0, 0, 0, 1, 0, 0, 1],
                           [1, 0, 0, 0, 0, 0, 1, 0, 3, 1],
                           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        self.reward = 0
        self.player_loc = [1, 1]
        self.before_player_loc = []
        self.canvas.destroy()
        self.canvas = self._build_canvas()
        # self.render()

    def render(self):
        time.sleep(self.render_speed)
        self.update()
        # pass

    def set_up(self, dir):
        player_x = self.player_loc[0]
        player_y = self.player_loc[1]
        # 0 1 2 3
        # "l", "u", "d", "r"
        # 遇到障碍物 5
        # 遇到目标 6
        before_x = player_x
        before_y = player_y

        if dir == 0:
            after_x = player_x - 1
            after_y = player_y

        elif dir == 1:
            after_x = player_x + 1
            after_y = player_y


        elif dir == 2:
            after_x = player_x
            after_y = player_y + 1

        elif dir == 3:
            after_x = player_x
            after_y = player_y - 1

        """
        self.game_board = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                           [1, 2, 0, 0, 0, 0, 1, 0, 0, 1],
                           [1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                           [1, 0, 0, 0, 1, 0, 1, 0, 1, 1],
                           [1, 0, 1, 0, 1, 0, 0, 0, 0, 1],
                           [1, 0, 1, 0, 1, 0, 0, 0, 0, 1],
                           [1, 0, 1, 0, 1, 0, 0, 0, 0, 1],
                           [1, 0, 1, 0, 0, 0, 1, 0, 0, 1],
                           [1, 0, 0, 0, 0, 0, 1, 0, 3, 1],
                           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        """

        if self.game_board[after_y][after_x] == 3:
            self.game_board[after_y][after_x] = 6
            self.game_board[before_y][before_x] = 0
            self.tot_reward += 10
            self.reward = 10
            self.player_loc = [after_x, after_y]
            return True

        # check wall
        elif self.game_board[after_y][after_x] == 1:
            self.tot_reward -= dm.Decimal(1.1)
            self.reward = -1.1
            return True

        else:
            self.game_board[after_y][after_x] = 2
            self.game_board[before_y][before_x] = 0
            self.player_loc = [after_x, after_y]

            if (self.player_loc == self.before_player_loc):
                self.reward = -0.6
                self.tot_reward -= dm.Decimal(0.6)

            else:
                self.reward = -0.1
                self.tot_reward -= dm.Decimal(0.1)

            self.before_player_loc = [before_x, before_y]

            return False

    def get_state(self):
        data = np.ravel(self.game_board, order='C')

        return data

    def step(self, action):

        done = self.set_up(action)
        self.steps += 1
        self.canvas.destroy()
        self.canvas = self._build_canvas()
        self.render()
        self.tot_reward = round(self.tot_reward, 1)

        return self.get_state(), self.reward, done
