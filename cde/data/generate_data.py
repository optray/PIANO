import math
import torch
import os
import random
import numpy as np
from pde import PDE, CartesianGrid, MemoryStorage, ScalarField, plot_kymograph
from initial_field import GaussianRF


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


setup_seed(0)
size_x = 128
size_t = 200
T = 5.0
delta_t = 0.0001
num_train = 1000
num_test = 200
num_val = 200

data_train = np.zeros([num_train, size_t, size_x])
idx_train = np.zeros(num_train)
a = GaussianRF(size_x)
b = a.sample(num_train)
bc = [{"value": 0.2}, 
{"curvature": 0.2},
{"derivative": 0.2},
{"type" :"mixed", "value" : 1, "const": 0.2}]

grid = CartesianGrid([[0, 1]], 128) # generate grid
field = ScalarField(grid, 2) # generate initial condition
for i in range(num_train):
    maxvalue = np.nan
    while np.isnan(maxvalue):
        idx = random.randint(0, len(bc) - 1)
        eq = PDE({"c": "0.1 * laplace(c) + 0.1*c+ 0.1*sin(6.28*x)"}, bc=bc[idx])
        field.data = np.array(b[i]).reshape(-1,)
        storage = MemoryStorage() # store intermediate information of the simulation
        res = eq.solve(field, T, dt=delta_t, tracker=storage.tracker(T/size_t)) # solve the PDE
        data_train[i] = np.array(storage.data)[1:, :]
        idx_train[i] = idx
        maxvalue = np.abs(data_train[i]).max()
    print(i, np.abs(data_train[i]).max(), bc[idx], maxvalue)

data_train = torch.tensor(data_train).permute(0, 2, 1)
idx_train = torch.tensor(idx_train)
torch.save(data_train, 'dataset/data_train')
torch.save(idx_train, 'dataset/idx_train')


size_x = 128
size_t = 240
T = 6.0
data_test = np.zeros([num_test, size_t, size_x])
idx_test = np.zeros(num_test)
a = GaussianRF(size_x)
b = a.sample(num_test)

for i in range(num_test):
    maxvalue = np.nan
    while np.isnan(maxvalue):
        idx = random.randint(0, len(bc) - 1)
        eq = PDE({"c": "0.1 * laplace(c) + 0.1*c+ 0.1*sin(6.28*x)"}, bc=bc[idx])
        field.data = np.array(b[i]).reshape(-1,)
        storage = MemoryStorage() # store intermediate information of the simulation
        res = eq.solve(field, T, dt=delta_t, tracker=storage.tracker(T/size_t)) # solve the PDE
        data_test[i] = np.array(storage.data)[1:, :]
        idx_test[i] = idx
        maxvalue = np.abs(data_test[i]).max()
    print(i, np.abs(data_test[i]).max(), bc[idx])

data_test = torch.tensor(data_test).permute(0, 2, 1)
idx_test = torch.tensor(idx_test)
torch.save(data_test, 'dataset/data_test')
torch.save(idx_test, 'dataset/idx_test')

size_x = 128
size_t = 200
T = 5.0

data_val = np.zeros([num_val, size_t, size_x])
idx_val = np.zeros(num_val)
a = GaussianRF(size_x)
b = a.sample(num_val)
for i in range(num_val):
    maxvalue = np.nan
    while np.isnan(maxvalue):
        idx = random.randint(0, len(bc) - 1)
        eq = PDE({"c": "0.1 * laplace(c) + 0.1*c + 0.1*sin(6.28*x)"}, bc=bc[idx])
        field.data = np.array(b[i]).reshape(-1,)
        storage = MemoryStorage() # store intermediate information of the simulation
        res = eq.solve(field, T, dt=delta_t, tracker=storage.tracker(T/size_t)) # solve the PDE
        data_val[i] = np.array(storage.data)[1:, :]
        idx_val[i] = idx
        maxvalue = np.abs(data_test[i]).max()
    print(i, np.abs(data_val[i]).max(), bc[idx])

data_val = torch.tensor(data_val).permute(0, 2, 1)
idx_val = torch.tensor(idx_val)
torch.save(data_val, 'dataset/data_val')
torch.save(idx_val, 'dataset/idx_val')