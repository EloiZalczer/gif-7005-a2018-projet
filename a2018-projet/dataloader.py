#!/usr/bin/python

from os import listdir, sys, path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import h5py
import numpy as np

class H5Dataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        h5_file = h5py.File(h5_path, 'r')
        self.x = h5_file.get('x')
        self.y = h5_file.get('y')
        self.length = len(self.x)

    def __getitem__(self, index):
        data = self.x[index].astype('float32')
        labels = self.y[index].astype('float32')
        sample = (data, labels)
        return sample

    def __len__(self):
        return self.length

class DataLoader:
    def __init__(self, filename, **kwargs):
        self.filepath = path.abspath(filename)
        print(self.filepath)
        self.dataloader_args = kwargs

    def load_data(self):
        dataset = H5Dataset(self.filepath)
        print("Dataset size : ", len(dataset))
        dataloader = torch.utils.data.DataLoader(
            dataset,
            **self.dataloader_args
        )

        return dataloader