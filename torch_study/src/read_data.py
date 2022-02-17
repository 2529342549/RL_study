#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2013-2021, Node Supply Chain Manager Corporation Limited.
@file: batch_train.py
@time: 2021/9/27 18:58
@desc:
'''
import os

from torch.utils.data import Dataset

import cv2
from PIL import Image


class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)


root_dir = '/home/hhd/PycharmProjects/RL_study_/torch_study/dataset/train/'
ant_label_dir = 'ants'
bees_label_dir = 'bees'
ants_dataset = MyData(root_dir, ant_label_dir)
bees_dataset = MyData(root_dir, bees_label_dir)
