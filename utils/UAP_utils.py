# -*- coding: utf-8 -*-
"""


@author: victor
"""
import torch
import torch.nn as nn

#Define the UAP as an additive noise to images, if X is a image in tensor form, UAP(X) returns X+delta, where delta is the UAP.
class UAP(nn.Module):
    def __init__(self,
                shape=(28, 28),
                num_channels=1,
                mean=[0.5],
                std=[0.5],
                use_cuda=True):
        super(UAP, self).__init__()

        self.use_cuda = use_cuda
        self.num_channels = num_channels
        self.shape = shape
        self.uap = nn.Parameter(torch.zeros(size=(num_channels, *shape), requires_grad=True))

        self.mean_tensor = torch.ones(1, num_channels, *shape)
        for idx in range(num_channels):
            self.mean_tensor[:,idx] *= mean[idx]
        if use_cuda:
            self.mean_tensor = self.mean_tensor.cuda()

        self.std_tensor = torch.ones(1, num_channels, *shape)
        for idx in range(num_channels):
            self.std_tensor[:,idx] *= std[idx]
        if use_cuda:
            self.std_tensor = self.std_tensor.cuda()
