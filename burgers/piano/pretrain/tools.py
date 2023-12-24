import torch
import random
import numpy as np
import os


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     os.environ['PYTHONHASHSEED'] = str(seed) 
     torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.benchmark = False
     torch.backends.cudnn.enabled = True



def argu_samples(data, tw_in, window=32):
    '''
    generate the argumentations of the samples, respectively
    input
    data: tensor, size = (b_size, num_steps, num_grid)
    t_size / s_size: size of the temporal / spatial  dimensions of the generated samples 
    output
    data_a / data_b: size = (b_size, t_size, s_size)
    '''
    device = data.device
    batch_size = data.shape[0]
    size = data.shape[1]
    tw = data.shape[2]
    data_a, data_b = torch.zeros(batch_size, window, tw_in).to(device), torch.zeros(batch_size, window, tw_in).to(device)
    for i in range(batch_size):
        t_idx_a, t_idx_b = random.randint(0, tw - tw_in), random.randint(0, tw - tw_in)
        w_idx = random.randint(0, size - window)
        data_a[i, :, :] = data[i, w_idx: w_idx+window, t_idx_a: t_idx_a + tw_in]
        data_b[i, :, :] = data[i,  w_idx: w_idx+window, t_idx_b: t_idx_b + tw_in] 
    return data_a, data_b