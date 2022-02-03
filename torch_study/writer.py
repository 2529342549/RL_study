#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2013-2021, Node Supply Chain Manager Corporation Limited.
@file: batch_train.py
@time: 2021/9/27 18:58
@desc:
'''
import numpy
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter("logs")
image_path="D:\\pythonProjects\\RL_study_\\torch_study\data\\train\\ants_image\\0013035.jpg"
img_PIL=Image.open(image_path)
img_array=numpy.array(img_PIL)
# print(type(img_array))
writer.add_image("test",img_array,1,dataformats='HWC')


for i in range(100):
    writer.add_scalar("y=2x", 3 * i, i)

writer.close()
