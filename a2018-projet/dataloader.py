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
        self.classes = [137, 81, 74, 111, 307, 328, 339, 336, 283, 288, 432, 418, 500, 0, 8, 22, 14, 441, 427, 444, 388]
        h5_file = h5py.File(h5_path, 'r+')

        # Code commente : pre-traitement du jeu de donnees (execute une seule fois)

        # arrY = h5_file['y'][:]
        # arrX = h5_file['x'][:]
        # print("ArrX shape : ", arrX.shape)
        # empty_indexes = np.where(np.any(arrY == True, axis=1) == True)[0]
        # print("Len empty indexes : ", len(empty_indexes))
        # resY = arrY[empty_indexes][:]
        # resX = arrX[empty_indexes][:]
        # print("ResY shape : ", resY.shape)
        # print("ResX shape : ", resX.shape)
        # h5_file.__delitem__('x')
        # h5_file['x'] = resX
        # h5_file.__delitem__('y')
        # h5_file['y'] = resY

        # arrY = h5_file['y'][:]
        # print(arrY)
        # res = arrY[:, self.classes]
        # print(res)
        # h5_file.__delitem__('y')
        # h5_file['y'] = res

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