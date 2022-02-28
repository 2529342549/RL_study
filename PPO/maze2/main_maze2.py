#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：RL_study_ 
@File    ：main_maze2.py
@Author  ：HHD
@Date    ：2022/2/27 下午9:01 
"""
import argparse
import logging
import os
import shutil
import sys
from env_maze2 import Env
from distlib.compat import raw_input

from PPO.maze2.ppo_maze2 import PPO, Transition


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
    agent = PPO()
    running_reward = 10
    env = Env()
    for e in range(10000):
        state = env.reset()

        # print(type(state), len(state))  # type:'numpy.ndarray' # len:364
        # print state
        episode_reward_sum = 0
        episode_step = 6000
        # state = env.get_state()
        # print(state)
        for t in range(episode_step):
            env.render()
            action, action_prob = agent.select_action(state)
            next_state, reward, done = env.step(action)
            trans = Transition(state, action, action_prob, reward, next_state)
            # if render: env.render()
            agent.store_transition(trans)
            # print next_state
            state = next_state
            episode_reward_sum += reward
            if e % 50 == 0:
                agent.save_param(e)
            # if t >= 1000:
            #     print('time out')
            #     done = True

            if done:
                agent.update(e)
                # running_reward = running_reward * 0.99 + t * 0.01
                # print('Ep: %d score: %.2f memory: %d episode_step: %d time: %d:%02d:%02d' %
                #       (e, episode_reward_sum, agent.counter, t, h, m, s))
                logging.info('Ep: %d score: %.2f memory: %d' % (e, episode_reward_sum, agent.counter))
                break

        # rewards.append(running_reward)
        # plot(rewards)


if __name__ == '__main__':
    main()
    print("end")
