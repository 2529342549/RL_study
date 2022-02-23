#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/24 下午7:40
# @Site    :
# @File    : maze_main.py
# @Software: PyCharm
import gym
from maze_env import FrozenLakeWapper
from maze_agent import Qlearning


def run_episode(env, agent):
    total_steps = 0
    total_reward = 0

    observation = env.reset()

    while True:
        action = agent.sample(observation)
        next_observation, reward, done, _ = env.step(action)
        # 训练
        agent.learn(observation, action, reward, next_observation, done)

        observation = next_observation
        total_reward += reward
        total_steps += 1

        env.render()
        if done:
            break
    return total_reward, total_steps


def main():
    env = gym.make("FrozenLake-v1", is_slippery=False)
    env = FrozenLakeWapper(env)

    agent = Qlearning(observation_n=env.observation_space.n, action_n=env.action_space.n, learning_rate=0.1, gamma=0.9,
                      e_greed=0.1)

    for episode in range(500):
        ep_reward, ep_steps = run_episode(env, agent)
        print('Episode %s: steps = %s , reward = %.1f' % (episode, ep_steps, ep_reward))


if __name__ == '__main__':
    main()
