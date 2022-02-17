#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/13 上午11:12
# @Site    : 
# @File    : gym_mountain_car.py
# @Software: PyCharm
import time

import gym
import numpy as np

env = gym.make('MountainCar-v0')
print('观测空间 = {}'.format(env.observation_space))
print('动作空间 = {}'.format(env.action_space))
print('观测范围 = {} ～ {}'.format(env.observation_space.low, env.observation_space.high))
print('动作数 = {}'.format(env.action_space.n))


class BespokeAgent:
    def __init__(self, env):
        pass

    # 决策,根据指定确定性策略决定动作
    def decide(self, observation):
        position, velocity = observation
        lb = min(-0.09 * (position + 0.25) ** 2 + 0.03, 0.3 * (position + 0.9) ** 4 - 0.008)
        ub = -0.07 * (position + 0.38) ** 2 + 0.06
        if lb < velocity < ub:
            action = 2
        else:
            action = 0
        return action  # 返回动作

    # 学习
    def learn(self, *args):
        pass


agent = BespokeAgent(env)


def play_montecarlo(env, agent, render=False, train=False):
    # 记录总回合奖励，初始值为0
    episode_reward = 0
    # 重置游戏环境，开始新回合
    observation = env.reset()
    while True:  # 循环，直到游戏结束
        # 判断是否显示
        if render:
            # 显示图形界面
            env.render()
        action = agent.decide(observation)
        # 执行动作
        next_observation, reward, done, _ = env.step(action)
        # 收集i回合奖励
        episode_reward += reward
        if train:  # 判断是否训练agent
            agent.learn(observation, action, reward, done)
        if done:  # 回合结束
            break

        observation = next_observation
    return episode_reward


env.seed(0)
episode_reward = play_montecarlo(env, agent, render=True)
print('回合奖励 ={}'.format(episode_reward))
# 运行100回合求平均以测试性能
episode_rewards = [play_montecarlo(env, agent) for _ in range(100)]
print('平均回合奖励 ={}'.format(np.mean(episode_rewards)))
# 关闭图形界面
time.sleep(5)

env.close()
