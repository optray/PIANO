from simclr import SimCLR
from train import train
from tools import setup_seed
from encoder import FNO_pretrain

import sys
import copy
import json
import argparse
from datetime import datetime
import random
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
plt.rcParams["animation.html"] = "jshtml"


def main(cfg):
    if not os.path.exists(f'log'):
        os.mkdir(f'log')
    if not os.path.exists(f'model'):
        os.mkdir(f'model')
    if not os.path.exists(f'embedding'):
        os.mkdir(f'embedding')
    setup_seed(cfg.seed)
    dateTimeObj = datetime.now()
    timestring = f'{dateTimeObj.date().month}{dateTimeObj.date().day}{dateTimeObj.time().hour}{dateTimeObj.time().minute}'
    logfile = f'log/fno_{cfg.seed}_m_{cfg.modes}_w_{cfg.width}_z_{cfg.z_size}_batch_size_{cfg.batch_size}_temp_{cfg.temperature}_{timestring}.csv'

    with open('log/cfg_'+ timestring +'.txt', 'w') as f:
        json.dump(cfg.__dict__, f, indent=2)

    sys.stdout = open(logfile, 'w')

    print('--------args----------')
    for k in list(vars(cfg).keys()):
        print('%s: %s' % (k, vars(cfg)[k]))
    print('--------args----------\n')

    sys.stdout.flush()
    encoder = FNO_pretrain(cfg.modes, cfg.width, cfg.h_size)
    net = SimCLR(encoder, cfg.h_size, cfg.z_size)
    train(cfg, net)
    torch.save(net.state_dict(), f'model/pretrain_{cfg.seed}.pt')


    # calculate the PI embeddings

    net.load_state_dict(torch.load('model/pretrain_'+ str(seed) + '.pt'))
    net.eval()
    device, data_path, tw_in = cfg.device, cfg.data_path, cfg.tw_in
    data_path = os.path.abspath(os.path.join(os.getcwd(), "../..")) + '/data/dataset/'
    train_data = torch.load(data_path+'data_train').to(device)[..., ::2,:tw_in].float()
    test_data = torch.load(data_path+'data_test').to(device)[..., ::2,:tw_in].float()
    val_data = torch.load(data_path+'data_val').to(device)[..., ::2,:tw_in].float()

    train_data_a = train_data[:, :20, :]
    train_data_b = train_data[:, -20:, :]

    test_data_a = test_data[:, :20, :]
    test_data_b = test_data[:, -20:, :]

    val_data_a = val_data[:, :20, :]
    val_data_b = val_data[:, -20:, :]

    train_embedding_a = net.encoder(train_data_a)
    train_embedding_b = net.encoder(train_data_b)
    train_embedding = torch.concat([train_embedding_a, train_embedding_b], dim=-1)
    train_embedding = train_embedding/torch.norm(train_embedding, dim=1)[..., None]
    torch.save(train_embedding, 'embedding/train_embedding_'+str(seed))

    test_embedding_a = net.encoder(test_data_a).detach()
    test_embedding_b = net.encoder(test_data_b).detach()
    test_embedding = torch.concat([test_embedding_a, test_embedding_b], dim=-1)
    test_embedding = test_embedding /torch.norm(test_embedding, dim=1)[..., None]
    torch.save(test_embedding, 'embedding/test_embedding_'+str(seed))

    val_embedding_a = net.encoder(val_data_a).detach()
    val_embedding_b = net.encoder(val_data_b).detach()
    val_embedding = torch.concat([val_embedding_a, val_embedding_b], dim=-1)
    val_embedding = val_embedding / torch.norm(val_embedding, dim=1)[..., None]
    torch.save(val_embedding, 'embedding/val_embedding_'+str(seed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyper-parameters of pretraining')
    
    parser.add_argument('--data_path', type=str, 
                        default=os.path.abspath(os.path.join(os.getcwd(), "../..")) + '/data/dataset/data_train',
                        help='path of data')

    parser.add_argument('--device', type=str, default='cuda:1',
                        help='Used device')
    
    parser.add_argument('--seed', type=int, default=0,
            help='seed')

    parser.add_argument('--modes', type=int, default=8,
            help='modes of encoder')

    parser.add_argument('--width', type=int, default=16,
            help='width of encoder')

    parser.add_argument('--h_size', type=int, default=64,
            help='widths of outputs of the encoder')

    parser.add_argument('--z_size', type=int, default=32,
            help='widths of outputs of the projector')

    parser.add_argument('--tw_in', type=int, default=20,
            help='frames of the input image')

    parser.add_argument('--temperature', type=float, default=0.5,
            help='temperature of the contrastive learning')

    parser.add_argument('--batch_size', type=int, default=512,
            help='batchsize of the contrastive learning')
    
    parser.add_argument('--step_size', type=int, default=4000,
            help='step_size of optim')
        
    parser.add_argument('--gamma', type=float, default=0.5,
            help='gamma of optim')
    
    parser.add_argument('--lr', type=float, default=0.001,
            help='lr of optim')

    parser.add_argument('--num_iterations', type=int, default=20000,
            help='num_iterations of optim')
    
    cfg = parser.parse_args()
    for seed in [0]:
        cfg.seed = seed
        main(cfg)
