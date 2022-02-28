#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：RL_study_ 
@File    ：main_maze2.py
@Author  ：HHD
@Date    ：2022/2/27 下午8:33 
"""
import argparse
import logging
import os
import shutil
import sys

from distlib.compat import raw_input

from dqn.maze.dqlearning import DQN, env, BATCH_SIZE


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--output_dir', type=str, default='data/output')
    args = parser.parse_args()
    key = raw_input('Output directory already exists! Overwrite the folder? (y/n)')
    if key == 'y':
        shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)
    file_handler = logging.FileHandler('data/output/output.log', mode='a')
    stdout_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(level=logging.INFO, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    dqn = DQN()
    for epoch in range(dqn.start_epoch, 10000):
        state = env.reset()
        # print s
        # init toltal reward
        episode_reward_sum = 0
        done = False
        # each loop represents a step
        for t in range(6000):
            action = dqn.choose_action(state)
            # print a
            s_, reward, done = env.step(action)
            # print s_
            # store samples
            dqn.store_transition(state, action, reward, s_)
            episode_reward_sum += reward
            # update state
            state = s_
            if dqn.memory_counter > BATCH_SIZE:
                # 开始学习 (抽取记忆，即32个transition，并对评估网络参数进行更新，并在开始学习后每隔100次将评估网络的参数赋给目标网络)
                dqn.learn()
            if epoch % 100 == 0:
                dqn.save_model()
            if t >= 2500:
                done = True
            if done:
                # print e
                logging.info(
                    'Ep: %d score: %.2f memory: %d epsilon: %.2f ' % (epoch, episode_reward_sum, dqn.memory_counter, dqn.epsilon))
                # param_keys = ['epsilon']
                # param_values = [dqn.epsilon]
                # param_dictionary = dict(zip(param_keys, param_values))
                break  # 该episode结束
            if dqn.epsilon > dqn.epsilon_min:
                dqn.epsilon = dqn.epsilon - 0.0001


if __name__ == '__main__':
    main()
