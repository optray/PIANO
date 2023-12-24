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


def test(net, input_data, output_data, embedding, w, train_step):
    net.eval()
    t = 0
    train_step = train_step - w
    pre_data = torch.zeros_like(output_data)
    step = output_data.shape[-1]
    for i in range(step//w):
        temp = net(input_data, embedding).detach()
        pre_data[..., i * w : i * w + w] = temp
        input_data = temp
        t = t+w
    error_in = (torch.norm(pre_data[...,:train_step] - output_data[...,:train_step], dim=1) / torch.norm(output_data[...,:train_step], dim=1)).mean(axis=-1)
    error_inf_in = torch.abs(pre_data[...,:train_step] - output_data[...,:train_step]).max(axis=1)[0].max(axis=1)[0] / torch.abs(output_data[...,:train_step]).max(axis=1)[0].max(axis=1)[0]
    print('DOMAIN***************************')
    print(error_in.mean().detach().item())
    print(error_in.std().detach().item())
    print(error_inf_in.mean().detach().item())
    print(error_inf_in.std().detach().item())

    if train_step < step:
        print('FUTURE***************************')
        error_out = (torch.norm(pre_data[...,train_step:] - output_data[...,train_step:], dim=1) / torch.norm(output_data[...,train_step:], dim=1)).mean(axis=-1)
        error_inf_out = torch.abs(pre_data[...,train_step:] - output_data[...,train_step:]).max(axis=1)[0].max(axis=1)[0] / torch.abs(output_data[...,train_step:]).max(axis=1)[0].max(axis=1)[0]
        print(error_out.mean().detach().item())
        print(error_out.std().detach().item())
        print(error_inf_out.mean().detach().item())
        print(error_inf_out.std().detach().item())

    return error_in.mean().detach().item()

def train(config, net, timestring):
    device = config.device
    w = config.w
    data_path = config.data_path
    embedding_path = config.embedding_path

    train_data = torch.load(data_path+'data_train').to(device).float()
    train_embedding = torch.load(embedding_path+f'train_embedding_{config.seed}').to(device)
    num_train = train_data.shape[0]

    test_data = torch.load(data_path+'data_test').to(device).float()
    test_embedding = torch.load(embedding_path+f'test_embedding_{config.seed}').to(device)
    num_test = test_data.shape[0]

    val_data = torch.load(data_path+'data_val').to(device).float()
    val_embedding = torch.load(embedding_path+f'val_embedding_{config.seed}').to(device)
    num_val = val_data.shape[0]

    train_step = train_data.shape[-1]
    test_step = test_data.shape[-1] 
    size = train_data.shape[-2]

    net = net.to(device)
    optimizer = optim.AdamW(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    batch_size = config.batch_size
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    err_record = 1e50
    early_stop = 0
    for step in range(config.num_iterations+1):
        net.train()
        l = list(range(num_train))
        random.shuffle(l)
        for i in range(int(num_train//batch_size)):
            input_data, output_data = torch.zeros(batch_size, size, size, w).to(device), torch.zeros(batch_size, size, size, w).to(device)
            idx = l[i * batch_size : (i + 1) * batch_size]
            n = random.randint(1, config.num_rollout)
            steps = [t for t in range(0, train_step-(n+1)*w+1)]
            random_steps = random.choices(steps, k=batch_size)
            for k in range(batch_size):
                t = random_steps[k]
                input_data[k] = train_data[idx[k], :, :, t:t+w]
                output_data[k] = train_data[idx[k], :, :, t+n*w:t+(n+1)*w]
            with torch.no_grad():
                for _ in range(n-1):    
                    pre_data = net(input_data, train_embedding[idx])
                    input_data = pre_data
            pre_data = net(input_data, train_embedding[idx])        
            loss = torch.sqrt(torch.mean(torch.square(pre_data - output_data)))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        early_stop+=1
        if step % 20 == 0:
            print(step, '#################################')
            print(loss.detach().item())
            print('training error (mean and std)')
            train_error = (torch.norm(pre_data - output_data, dim=1) / torch.norm(output_data, dim=1)).mean(axis=-1)
            print(train_error.mean().detach().item())
            
            input_data = val_data[..., :w]
            output_data = val_data[..., w:]
            print('val error (mean and std)')
            val_error = test(net, input_data, output_data, val_embedding, w, train_step)
            
            if val_error < err_record:
                early_stop=0
                err_record = val_error
                print('----------------------------MODEL_UPDATED-------------------------------')
                torch.save(net.state_dict(), f'model/{config.model}_{config.seed}_{timestring}.pt')
                print('test loss (mean and std)')
                input_data = test_data[..., :w]
                output_data = test_data[..., w:]
                test(net, input_data, output_data, test_embedding, w, train_step)

            sys.stdout.flush()
        if early_stop > 2000:
            break

    print('----------------------------FINAL_RESULT-----------------------------')
    print('test loss (mean and std)')
    net.load_state_dict(torch.load(f'model/{config.model}_{config.seed}_{timestring}.pt'))
    input_data = test_data[..., :w]
    output_data = test_data[..., w:]
    test(net, input_data, output_data, test_embedding, w, train_step)
    sys.stdout.flush()