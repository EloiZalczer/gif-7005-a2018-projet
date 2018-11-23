#!/usr/bin/python

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class FullyConnectedNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.C1 = nn.Conv1d(10, 10, 5)

        self.L1 = nn.Linear(124, 4000)
        self.L2 = nn.Linear(4000, 4000)
        self.L3 = nn.Linear(4000, 527)

    def forward(self, x):

        y = self.L1(F.relu(self.C1.forward(x)))
        y = self.L2(F.relu(y))
        y = self.L3(F.relu(y))

        y = y.mean(dim=1)

        return torch.sigmoid(y)

