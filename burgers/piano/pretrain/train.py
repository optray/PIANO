from tools import argu_samples
from nt_xent import NT_Xent


import sys
import copy
import random
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
plt.rcParams["animation.html"] = "jshtml"


def train(config, net):
    device = config.device
    tw_in = config.tw_in
    temperature = config.temperature
    batch_size = config.batch_size
    net = net.to(device)
    optimizer = optim.AdamW(net.parameters(), config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    data = torch.load(config.data_path).to(device)
    train_num = data.shape[0]
    criterion = NT_Xent(batch_size, temperature)
    for step in range(config.num_iterations+1):
        net.train()
        idx = torch.randint(train_num, size = (batch_size,)).to(device)
        data_batch = data[idx, ::2, :]
        data_a, data_b = argu_samples(data_batch, tw_in)
        _, _, z_a, z_b = net(data_a, data_b)
        loss = criterion(z_a, z_b)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        if step % 50 == 0:
            print('########################training loss', step)
            print(loss.detach().item())
            sys.stdout.flush()