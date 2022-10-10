# -*- coding: utf-8 -*-
import time
from model import PBML
from train import eval_model, train_model
import torch
import random   
import numpy as np


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


model_name = 'PBML'
encoder_name='BERT'

dataset2config = {  "FewRel":  {"taskname":"Relation Classification",
                               "meta_lr": 5e-5,
                               "task_lr": 1e-2,
                               "weight decay": 1e-2,
                               "batch_size": 32,
                               "train_iters": 1000,
                               "steps": 30,
                               "max_length":90,
                               "warmup_step":200
                               },
                    "HuffPost": {"taskname":"Headlines Classification",
                                "meta_lr": 1e-5,
                                "task_lr": 1e-3,
                                "weight decay": 2e-1,
                                "batch_size": 16, # 32
                                "train_iters": 1000,
                                "steps": 50, # 50
                                "max_length":90,
                                "warmup_step":100
                               },
                    "Reuters": {"taskname":"Article Classification",
                                "meta_lr": 2e-5,
                                "task_lr": 5e-3,
                                "weight decay": 1e-1,
                                "batch_size": 4, 
                                "train_iters": 500,
                                "steps": 20, # 20
                                "max_length":300,
                                "warmup_step":50
                               },
                    "Amazon": {"taskname":"Review Classification",
                                "meta_lr": 3e-6,
                                "task_lr": 1e-2,
                                "weight decay": 3e-1,
                                "batch_size": 8, # 16
                                "train_iters": 2000,
                                "steps": 20, # 20
                                "max_length":300,
                                "warmup_step":200
                               },}

benchmark = "Amazon"  # {"FewRel","HuffPost","Reuters","Amazon"}
taskname = dataset2config[benchmark]['taskname']
meta_lr = dataset2config[benchmark]['meta_lr']
task_lr  = dataset2config[benchmark]['task_lr']
weight_decay = dataset2config[benchmark]['weight decay']
B = dataset2config[benchmark]['batch_size']
Train_iter = dataset2config[benchmark]['train_iters']
Fast_tuning_steps = dataset2config[benchmark]['steps']
max_length = dataset2config[benchmark]['max_length']
warmup_step = dataset2config[benchmark]['warmup_step']

noise_rate = 0 #  from 0 to 10

N = 9
K = 5
Q = 1

Val_iter = 2000
Val_step = 1000000

save_ckpt = f'./checkpoint/{benchmark}_MAML.pth'
load_ckpt = None
best_acc = 0.0



print('----------------------------------------------------')
print("{}-way-{}-shot Few-Shot {}".format(N, K,taskname))
print("Model: {}".format(model_name))
print("Encoder: {}".format(encoder_name))
print('----------------------------------------------------')


data_dir = {'benchmark': benchmark,
            'train':f'./data/{benchmark}/train.json',
            'val':f'./data/{benchmark}/val.json',
            'test':f'./data/{benchmark}/test.json', 
            'noise_rate': noise_rate,
            'candidates': f'./data/{benchmark}/candidate_ebds.json',
            'pb_dropout': 0.5}                
                               
start_time=time.time()

pbml=PBML(B,N,K,max_length,data_dir)

# train_model(pbml,B,N,K,Q,data_dir,
#             meta_lr=meta_lr,
#             task_lr=task_lr,
#             weight_decay = weight_decay,
#             train_iter=Train_iter,
#             val_iter=Val_iter,
#             val_step=Val_step, 
#             steps = Fast_tuning_steps,
#             save_ckpt = save_ckpt, load_ckpt= load_ckpt,
#             best_acc = best_acc,
#             warmup_step = warmup_step
#             )

load_ckpt = f'./checkpoint/{benchmark}_MAML.pth'
eval_model(pbml,N,K,Q,eval_iter=10000, steps=Fast_tuning_steps,task_lr=task_lr, noise_rate = 0,file_name=f'./data/{benchmark}/test.json',load_ckpt=load_ckpt)


time_use=time.time()-start_time
h=int(time_use/3600)
time_use-=h*3600
m=int(time_use/60)
time_use-=m*60
s=int(time_use)
print('Totally used',h,'hours',m,'minutes',s,'seconds')
