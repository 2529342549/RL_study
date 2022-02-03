#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2013-2021, Node Supply Chain Manager Corporation Limited. 
@file: torch_numpy.py
@time: 2021/9/27 15:50
@desc:
'''
import numpy
import torch
import numpy as np

# np_data = np.arange(6).reshape(2, 3)
# torch_data = torch.from_numpy(np_data)
# tensor_to_array=torch_data.numpy()
# print(np_data, torch_data,tensor_to_array)

# abs
data = [-1, -2, 1, 2]
tensor = torch.FloatTensor(data)
print(torch.abs(tensor))
print(np.abs(data))
