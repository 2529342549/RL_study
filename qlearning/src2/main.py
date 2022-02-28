#!/usr/bin/env python
# encoding: utf-8
"""
@author: HHD
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited. 
@contact: 2529342549@qq.com
@software: pycharm
@file: main_maze2.py
@time: 2022/2/5 下午5:17
@desc:
"""

from environment import Maze
from qlenrning import QLearningTable


def update():
    for episode in range(100):
        total_reward = 0
        # initial observation
        observation = env.reset()
        # print(observation)
        while True:
            # fresh env
            env.render()
            # RL choose action based on observation
            action = RL.choose_action(str(observation))
            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            # RL learn from this transition11
            RL.learn(str(observation), action, reward, str(observation_))
            # swap observation
            observation = observation_
            # break while loop when end of this episode
            total_reward += reward
            if done:
                print("{} episode reward is {}".format(episode, total_reward))
                break
    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))
    env.after(100, update)
    env.mainloop()
