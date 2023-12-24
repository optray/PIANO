import torch.nn as nn
import numpy as np
import copy
import torch
import torch.nn.functional as F


class SimCLR(nn.Module):
    def __init__(self, encoder, h_size, z_size):

        super(SimCLR, self).__init__()

        '''
        input size: batch_size * 25 * 25 * 1
        encoder output size: batch_size * h_size
        '''

        self.h_size = h_size
        self.z_size = z_size

        self.encoder = encoder
        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.h_size, self.h_size, bias=False),
            nn.ReLU(),
            nn.Linear(self.h_size, z_size, bias=False),
        )

    def forward(self, x_i, x_j):
        num = x_i.shape[0]
        h_i = self.encoder(x_i).reshape(num, -1)
        h_j = self.encoder(x_j).reshape(num, -1)

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        return h_i, h_j, z_i, z_j