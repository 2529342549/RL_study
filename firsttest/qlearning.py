#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/11 下午3:55
# @Site    : 
# @File    : q_learning.py
# @Software: PyCharm

import time
import numpy as np
import pandas as pd

# reproducible
np.random.seed(2)
# the length of the 1 dimensional world
N_STATES = 6
# available actions
ACTIONS = ['left', 'right']
# greedy police
EPSILON = 0.9
# learning rate
ALPHA = 0.1
# discount factor
LAMBDA = 0.9
# maximum episodes
MAX_EPISODES = 13
# fresh time for one move
FRESH_TIME = 0.01


def create_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),  # q_table initial values
        columns=actions,  # actions' name
    )
    # print(table)
    return table


def choose_action(state, q_table):
    # this is how to choose an action
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.idxmax()
    return action_name


def get_env_feedback(S, A):
    # This is how agent will interact with the environment
    if A == 'right':  # move right
        if S == N_STATES - 2:
            S = 'terminal'
            R = 1
        else:
            S = S + 1
            R = 0
    else:
        R = 0
        if S == 0:
            S = S
        else:
            S = S - 1
    return S, R


def update_env(S, episode, step_counter):
    # this is how environment be update
    env_list = ['-'] * (N_STATES - 1) + ['T']
    if S == 'terminal':
        interaction = 'Episode %s: total_steps=%s' % (episode + 1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                               ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def rl():
    # main part of RL loop
    q_table = create_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated:
            A = choose_action(S, q_table)
            # take action & get next state and reward
            S_, R = get_env_feedback(S, A)
            q_predict = q_table.loc[S, A]
            if S_ != 'terminal':
                # next state is not terminal
                q_target = R + LAMBDA * q_table.iloc[S_, :].max()
            else:
                # next state is terminal
                q_target = R
                # terminal this episode
                is_terminated = True
            # update
            q_table.loc[S, A] += ALPHA * (q_target - q_predict)
            # move to next state
            S = S_

            update_env(S, episode, step_counter + 1)
            step_counter += 1
    return q_table


if __name__ == '__main__':
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)
