#!/usr/bin/python

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class Resnet(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = resnet18(pretrained=True)

        dim_before_fc = self.model.fc.in_features

        self.model.conv1 = nn.Conv2d(10, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model.fc = nn.Linear(dim_before_fc, 21)

        # for name, param in self.model.named_parameters():
        #     if name != "fc.weight" and name != "fc.bias":
        #         param.requires_grad = False

    def forward(self, x):
        y = self.model.forward(x)

        return torch.sigmoid(y)