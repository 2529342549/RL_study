#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/13 上午10:14
# @Site    : 
# @File    : use_gym_test.py
# @Software: PyCharm
import time

import gym
from gym import envs

env = gym.make('Pong-v0')
print(env)

env_specs = envs.registry.all()
# print(env_specs)
env_ids = [env_spec.id for env_spec in env_specs]
# print(env_ids)
env.reset()
# step接收agent的动作作为参数。
re_step = env.step(0)  # 返回四个参数，观测(observation)，奖励(reward)，回合结束指示(done)，其他信息(info)。
# print(re_step)  # (array([ 0.02856488, -0.14600424, -0.01346741,  0.33676739]), 1.0, False, {})
action = env.action_space.sample()  # 从动作空间中随即选取一个动作。
a_step = env.step(action)
# 每次调用env.step()会让环境前进一步。所以，env.step()往往放在循环里。
print(a_step)
env.render()
time.sleep(5)
env.close()