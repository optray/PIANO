import sys
from train import train
from fno import PIANO_FNO
from ffno import FNOFactorized2DBlock
from unet import PIANO_UNet

import copy
from datetime import datetime
import random
import json
import argparse
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
plt.rcParams["animation.html"] = "jshtml"

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


def main(cfg):
    if not os.path.exists(f'log'):
        os.mkdir(f'log')
    if not os.path.exists(f'model'):
        os.mkdir(f'model')
    setup_seed(cfg.seed)
    dateTimeObj = datetime.now()
    timestring = f'{dateTimeObj.date().month}{dateTimeObj.date().day}{dateTimeObj.time().hour}{dateTimeObj.time().minute}'
    logfile = f'log/{cfg.model}_seed_{cfg.seed}_{cfg.num_iterations}_num_rollout_{cfg.num_rollout}_lr_{cfg.lr}_weight_decay_{cfg.weight_decay}_{timestring}.csv'

    with open('log/cfg_'+ timestring +'.txt', 'w') as f:
        json.dump(cfg.__dict__, f, indent=2)

    sys.stdout = open(logfile, 'w')

    print('--------args----------')
    for k in list(vars(cfg).keys()):
        print('%s: %s' % (k, vars(cfg)[k]))
    print('--------args----------\n')

    sys.stdout.flush()
    if cfg.model == 'fno':
        net = PIANO_FNO(width=21)
    if cfg.model == 'unet':
        net = PIANO_UNet()
    if cfg.model == 'ffno':
        net = FNOFactorized2DBlock()
    train(cfg, net, timestring)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyper-parameters of pretraining')

    parser.add_argument('--num_rollout', type=int, 
                        default = 2,
                        help='number of rollout steps')

    parser.add_argument('--data_path', type=str, 
                        default=os.path.abspath(os.path.join(os.getcwd(), "../..")) + '/data/dataset/',
                        help='path of data')

    parser.add_argument('--embedding_path', type=str, 
                        default=os.path.abspath(os.path.dirname(os.getcwd())) + '/pretrain/embedding/',
                        help='path of embeddings')

    parser.add_argument('--device', type=str, default='cuda:2',
                        help='Used device')
    
    parser.add_argument('--seed', type=int, default=0,
            help='seed')

    parser.add_argument('--w', type=int, default=20,
            help='rollout steps')

    parser.add_argument('--batch_size', type=int, default=100,
            help='batchsize of the operator learning')
    
    parser.add_argument('--step_size', type=int, default=5000,
            help='step_size of optim')
        
    parser.add_argument('--gamma', type=float, default=0.8,
            help='gamma of optim')
    
    parser.add_argument('--lr', type=float, default=0.01,
            help='lr of optim')

    parser.add_argument('--weight_decay', type=float, default=0.00,
            help='lr of optim')

    parser.add_argument('--num_iterations', type=int, default=20000,
            help='num_iterations of optim')
    
    parser.add_argument('--model', type=str, default='fno',
            help='backbone model')
    
    cfg = parser.parse_args()
    for seed in [0]:
        cfg.model = 'fno'
        cfg.seed = seed
        cfg.lr = 0.01
        main(cfg)

    # cfg = parser.parse_args()
    # for seed in [0]:
    #     cfg.model = 'ffno'
    #     cfg.seed = seed
    #     cfg.lr = 0.001
    #     main(cfg)

    # cfg = parser.parse_args()
    # for seed in [0]:
    #     cfg.model = 'unet'
    #     cfg.seed = seed
    #     cfg.lr = 0.002
    #     main(cfg)