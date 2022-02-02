#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2013-2021, Node Supply Chain Manager Corporation Limited.
@file: batch_train.py
@time: 2021/9/27 18:58
@desc:
'''
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")

for i in range(100):
    writer.add_scalar("y=2x", 3 * i, i)

writer.close()
