#!/usr/bin/env python
# encoding: utf-8
"""
@author: HHD
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited. 
@contact: 2529342549@qq.com
@software: pycharm
@file: plot.py
@time: 2022/2/12 下午5:16
@desc:
"""
import re
import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

sns.set()
parser = argparse.ArgumentParser()
# parser.add_argument('--log_files', type=str, default='../data/output.log')
parser.add_argument('--window_size', type=int, default=50)
args = parser.parse_args()


def running_mean(x, n):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[n:] - cumsum[:-n]) / float(n)


def main():
    # define the names of the models you want to plot and the longest episodes you want to show
    max_episodes = 6000
    log_file = ('/home/hhd/PycharmProjects/RL_study_/PPO/maze3/data/output.log')
    with open(log_file, 'r') as file:
        log = file.read()

    train_pattern = r"# of episode :(.*), avg score : (.*)"
    train_reward = []
    for r in re.findall(train_pattern, log):
        train_reward.append(float(r[1]))

    train_reward = train_reward[:max_episodes]
    # print train_reward
    train_reward_smooth = running_mean(train_reward, args.window_size)
    _, ax4 = plt.subplots()

    ax4_legends = []
    ax4.plot(range(len(train_reward_smooth)), train_reward_smooth)
    # ax4_legends.append(models[i])
    ax4_legends.append('ppo')
    ax4.legend(ax4_legends, shadow=True, loc='best')
    # ax4.grid(True)
    ax4 = plt.gca()
    # ax4.patch.set_facecolor('xkcd:mint green')
    ax4.spines['top'].set_visible(False)  # 去掉上边框
    ax4.spines['right'].set_visible(False)  # 去掉右边框
    # ax4.patch.set_facecolor("green")
    ax4.patch.set_alpha(0.5)
    ax4.set_xlabel('Episodes')
    ax4.set_ylabel('Reward')
    ax4.set_title('Cumulative Discounted Reward')

    plt.show()


if __name__ == '__main__':
    main()
