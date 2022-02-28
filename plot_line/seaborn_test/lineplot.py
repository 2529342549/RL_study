#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：RL_study_ 
@File    ：lineplot.py
@Author  ：HHD
@Date    ：2022/2/23 下午9:04 
"""
import argparse
import re

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set()


def running_mean(data, n):
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[n:] - cumsum[:-n]) / float(n)


def get_data():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_files', type=str, nargs='+')
    parser.add_argument('--window_size', type=int, default=300)
    args = parser.parse_args()
    train_pattern = r"TRAIN_new in episode (?P<episode>\d+) has success rate: (?P<sr>[0-1].\d+), " \
                    r"collision rate: (?P<cr>[0-1].\d+), nav time: (?P<time>\d+.\d+), " \
                    r"total reward: (?P<reward>[-+]?\d+.\d+)"
    new_train_pattern = r"TRAIN in episode (?P<episode>\d+) has success rate: (?P<sr>[0-1].\d+), " \
                        r"collision rate: (?P<cr>[0-1].\d+), nav time: (?P<time>\d+.\d+), " \
                        r"total reward: (?P<reward>[-+]?\d+.\d+)"
    train_reward = []
    train_reward2 = []
    max_episodes = 10000
    for _, log_file in enumerate(args.log_files):
        with open(log_file, 'r') as file:
            log = file.read()
        # print(log)
        for r in re.findall(train_pattern, log):
            train_reward.append(float(r[1]))
        # print(train_reward)
        train_reward = train_reward[:max_episodes]
        # smooth training plot
        train_reward_smooth1 = running_mean(train_reward, args.window_size)
        # print(train_reward_smooth1)
        for r in re.findall(new_train_pattern, log):
            train_reward2.append(float(r[1]))
        train_reward2 = train_reward2[:max_episodes]

        train_reward_smooth2 = running_mean(train_reward2, args.window_size)
        return train_reward_smooth1, train_reward_smooth2


def main():
    reward1, reward2 = get_data()
    # print(reward1)
    # print(reward2)
    rewards = np.concatenate((reward1, reward2))
    episode1 = range(len(reward1))
    episode2 = range(len(reward2))
    episode = np.concatenate((episode1, episode2))
    sns.lineplot(x=episode, y=rewards)
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.show()


if __name__ == '__main__':
    # get_data()
    main()
