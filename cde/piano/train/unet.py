import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.nn.parameter import Parameter

import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial

from timeit import default_timer


class DynamicConv(nn.Module):
    def __init__(self,K,T,d_embds,in_channel,out_channel,kernel_size=1,stride=1,padding=0,dilation=1,grounps=1,bias=True,init_weight=True):
        super().__init__()
        self.in_planes=in_channel
        self.out_planes=out_channel
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.dilation=dilation
        self.groups=grounps
        self.bias=bias
        self.K=K
        self.T=T
        self.init_weight=init_weight
        self.d_embds = d_embds
        self.attfc0 = nn.Sequential(nn.Linear(self.d_embds, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, K))

        self.weight=nn.Parameter(torch.randn(K,out_channel,in_channel//grounps,kernel_size),requires_grad=True)
        if(bias):
            self.bias=nn.Parameter(torch.randn(K,out_channel),requires_grad=True)
        else:
            self.bias=None
        
        if(self.init_weight):
            self._initialize_weights()

        #TODO 初始化
    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def forward(self,x,embedding):
        bs,in_planels,h = x.shape
        att = self.attfc0(embedding)
        att = F.softmax(att/self.T, dim=-1)  #bs,K
        x=x.reshape(1,-1,h)
        weight=self.weight.view(self.K,-1) #K,-1
        aggregate_weight=torch.mm(att,weight).view(bs*self.out_planes,self.in_planes//self.groups,self.kernel_size) #bs*out_p,in_p,k,k

        if(self.bias is not None):
            bias=self.bias.view(self.K,-1) #K,out_p
            aggregate_bias=torch.mm(att,bias).view(-1) #bs,out_p
            output=F.conv1d(x,weight=aggregate_weight,bias=aggregate_bias,stride=self.stride,padding=self.padding,groups=self.groups*bs,dilation=self.dilation)
        else:
            output=F.conv1d(x,weight=aggregate_weight,bias=None,stride=self.stride,padding=self.padding,groups=self.groups*bs,dilation=self.dilation)
        
        output=output.view(bs,self.out_planes,h)
        return output

class PIANO_UNet(nn.Module):

    def __init__(self, K=4, T=10, d_embds=128, out_channels=20, init_features=16):
        super(PIANO_UNet, self).__init__()

        features = init_features
        in_channels = out_channels 
        self.encoder1 = DynamicConv(K,T,d_embds,in_channels,init_features)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder2 = PIANO_UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder3 = PIANO_UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder4 = PIANO_UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.bottleneck = PIANO_UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose1d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = PIANO_UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose1d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = PIANO_UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose1d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = PIANO_UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose1d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = PIANO_UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv1d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x, embedding):
        x = x.permute(0, 2, 1)
        enc1 = self.encoder1(x, embedding)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1).permute(0, 2, 1)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm1d(num_features=features)),
                    (name + "tanh1", nn.ReLU()),
                    (
                        name + "conv2",
                        nn.Conv1d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm1d(num_features=features)),
                    (name + "tanh2", nn.ReLU()),
                ]
            )
        )