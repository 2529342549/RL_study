#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/19 下午7:32
# @Site    : 
# @File    : runner.py
# @Software: PyCharm

import time
from layout import InitLayout
from agent import SarsaTable
from agent import SarsaLambdaTable

from agent import QLearningTable

# sarsa
def start():
    # 总训练次数
    TOTAL_EXPLORE_EPOCH = 500
    # 95%后查看结果
    SHOW_SLOW_EFFECT = TOTAL_EXPLORE_EPOCH * 0.95

    for epoc in range(TOTAL_EXPLORE_EPOCH):
        # 智能体初始坐标
        observation = env.reset()
        # 每次训练结束时reward=0
        total_reward = 0

        # sarsa_lambda选择状态
        action = MyRL.action_select(str(observation))

        while True:
            env.render()
            # 移动,获取reward
            # time.sleep(0.9)

            next_observation, reward, done = env.step(action)

            # 选择下一个状态
            action_ = MyRL.action_select(str(next_observation))

            # sarsa_lambda学习，更新q-table
            MyRL.learn(str(observation), action, reward, str(next_observation), action_)

            # swap
            observation = next_observation
            action = action_

            # 将结果慢速显示
            if epoc > SHOW_SLOW_EFFECT:
                time.sleep(0.3)
            # 总分累加
            total_reward += reward

            if done:
                print('========== 第%d回合 R：%.6f==========' % (epoc, total_reward))
                break

    # MyRL.q_table.to_csv('Q-table.csv')
    env.destroy()

# sarsa_lambda
# def start():
#     # 总训练次数
#     TOTAL_EXPLORE_EPOCH = 500
#     # 95%后查看结果
#     SHOW_SLOW_EFFECT = TOTAL_EXPLORE_EPOCH * 0.95
#
#     for epoc in range(TOTAL_EXPLORE_EPOCH):
#         # 智能体初始坐标
#         observation = env.reset()
#         # 每次训练结束时reward=0
#         total_reward = 0
#
#         # sarsa_lambda选择状态
#         action = MyRL.action_select(str(observation))
#
#         while True:
#             env.render()
#             # 移动,获取reward
#             # time.sleep(0.9)
#
#             next_observation, reward, done = env.step(action)
#
#             # 选择下一个状态
#             action_ = MyRL.action_select(str(next_observation))
#
#             # sarsa_lambda学习，更新q-table
#             MyRL.learn(str(observation), action, reward, str(next_observation), action_)
#
#             # swap
#             observation = next_observation
#             action = action_
#
#             # 将结果慢速显示
#             if epoc > SHOW_SLOW_EFFECT:
#                 time.sleep(0.3)
#             # 总分累加
#             total_reward += reward
#
#             if done:
#                 print('========== 第%d回合 R：%.6f==========' % (epoc, total_reward))
#                 break
#
#     # MyRL.q_table.to_csv('Q-table.csv')
#     env.destroy()


# Q_learning
# def start():
#     # 总训练次数
#     TOTAL_EXPLORE_EPOCH = 400
#     # 95%后查看结果
#     SHOW_SLOW_EFFECT = TOTAL_EXPLORE_EPOCH * 0.95
#
#     for epoc in range(TOTAL_EXPLORE_EPOCH):
#         # 智能体初始坐标
#         observation = env.reset()
#         # 每次训练结束时reward=0
#         total_reward = 0
#
#
#         while True:
#             env.render()
#             # 移动,获取reward
#             # time.sleep(0.9)
#
#             action = MyRL.action_select(str(observation))
#             next_observation, reward, done = env.step(action)
#
#             # 选择下一个状态
#             action_ = MyRL.action_select(str(next_observation))
#
#             # 学习，更新q-table
#             MyRL.learn(str(observation), action, reward, str(next_observation))
#
#
#             # swap
#             observation = next_observation
#
#             # 将结果慢速显示
#             if epoc > SHOW_SLOW_EFFECT:
#                 time.sleep(0.3)
#             # 总分累加
#             total_reward += reward
#
#             if done:
#                 print('==========epoch %d R：%.6f==========' % (epoc, total_reward))
#                 # print('*********第%d回合********' % epoc)
#                 break
#
#     # MyRL.q_table.to_csv('Q-table.csv')
#     env.destroy()

if __name__ == '__main__':
    env = InitLayout()
    MyRL = SarsaTable(actions=range(env.actions_num))
    # MyRL = SarsaLambdaTable(actions=range(env.actions_num))
    # MyRL = QLearningTable(actions=range(env.actions_num))

    env.after(10, start)
    env.mainloop()
