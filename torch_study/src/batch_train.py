#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2013-2021, Node Supply Chain Manager Corporation Limited.
@file: batch_train.py
@time: 2021/9/27 18:58
@desc:
'''
import torch
import torch.utils.data as Data

torch.manual_seed(1)
BATCH_SIZE = 5

x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)

torch_dataset = Data.TensorDataset(x, y)
loader=Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)

if __name__ == '__main__':
    for epoch in range(3):
        for step, (batch_x, batch_y) in enumerate(loader):
            # training....
            print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
                  batch_x.numpy(), '| batch y: ', batch_y.numpy())


