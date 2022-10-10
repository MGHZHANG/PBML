# -*- coding: utf-8 -*-
import torch
from torch import autograd
from torch.nn import functional as F
from transformers import AdamW,get_linear_schedule_with_warmup
from dataloader import FewshotDataset
import sys

from model import PBML



def fast_tuning(W,support,support_label,query,net,steps,task_lr,N,K):
    '''
       W:               label word embedding matrix                             [N, hidden_size] 
       support:         support instance hidden states at [MASK] place          [N*K, hidden_size]    
       support_label:   support instance label id:                              [N*K]
       query:           query instance hidden states at [MASK] place            [total_Q, hidden_size]
       steps：          fast-tuning steps                                       
       task_lr:         fast-tuning learning rate for task-adaptation          
    '''
    prototype_label = torch.tensor( [i for i in range(N)]).cuda() # [0,1,2,...N]
    # attention score calc
    idx = torch.zeros(N*K).long().cuda()
    for i in range(N): idx[i*K:(i+1)*K] = i # [0,0,...0,1,1...1,...N-1...N-1]
    att=(support * W[idx]).sum(-1).reshape(N,K) # ([N*K,bert_size]·[N*K,bert_size]).sum(-1) = [N*K] ——>  [N,K]
    T = 3
    att = F.softmax(att/T,-1).detach() # [N,K]
    # att: attention scores α_i^j

    for _ in range(steps):
        logits_for_instances, logits_for_classes = net(support,W) # [N*K, N], [N, N]
        if att is None:
            loss_s2v = net.loss(logits_for_instances, support_label)
            loss_v2s = net.loss(logits_for_classes, prototype_label)

            loss = loss_s2v + loss_v2s

            grads = autograd.grad(loss,W)
            W = W - task_lr*grads[0]
        else:
            Att = att.flatten() # [N*K]
            loss = torch.FloatTensor([0.0] * (N*K)).cuda()
            for i in range(N*K):
                loss[i]  = net.loss(logits_for_instances[i].unsqueeze(0),support_label[i])/N 
            loss_tot = Att.dot(loss)
            grad = autograd.grad(loss_tot,W)
            W = W - task_lr*grad[0]

    logits_q = net(query, W)[0] # [total_Q, n_way] 
    return logits_q

def train_one_batch(idx,class_names,support0,support_label,query0,query_label,net,steps,task_lr):
    '''
    idx:                batch index         
    class_names：       N categories names (or name id)             List[class_name * N]
    support0:           raw support texts                           List[{tokens:[],h:[],t:[]} * (N*K)]
    support_label:      support instance labels                     [N*K]
    query0:             raw query texts                             List[{tokens:[],h:[],t:[]} * total_Q]
    query_label:        query instance labels                       [total_Q]
    net:                PBML model
    steps：             fast-tuning steps                                       
    task_lr:            fast-tuning learning rate for task-adaptation
    '''
    N, K = net.n_way, net.k_shot
    support, query = net.coder(support0), net.coder(query0) # [N*K,bert_size]
    candidate_word_embeddings =net.get_info(class_names) # [N * [candidate word embeddings]]

    net.W[idx] = net.prework(candidate_word_embeddings)

    logits_q = fast_tuning(net.W[idx],support,support_label,query,net,steps,task_lr,N,K)

    return net.loss(logits_q, query_label),   net.accuracy(logits_q, query_label)


def test_model(data_loader,model,val_iter,steps,task_lr):
    accs=0.0
    model.eval()

    for it in range(val_iter):
        net = model
        class_name,support,support_label,query,query_label = data_loader[0]
        loss,right = train_one_batch(0,class_name, support, support_label,query,query_label,net,steps,task_lr)
        accs += right 
        sys.stdout.write('[EVAL] step: {0:4} | accuracy: {1:3.2f}%'.format(it + 1, 100 * accs / (it+1)) + '\r')
        sys.stdout.flush()

    return accs/val_iter


def train_model(model:PBML, B,N,K,Q,data_dir,
            meta_lr=5e-5, 
            task_lr=1e-2,
            weight_decay = 1e-2,
            train_iter=2000,
            val_iter=2000,
            val_step=50,
            steps=30,
            save_ckpt = None,
            load_ckpt = None,
            best_acc = 0.0,
            fp16 = True,
            warmup_step = 200):

    n_way_k_shot = str(N) + '-way-' + str(K) + '-shot'
    print('Start training ' + n_way_k_shot)
    cuda = torch.cuda.is_available()
    if cuda: model = model.cuda()

    if load_ckpt:
        state_dict = torch.load(load_ckpt)['state_dict']
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                print('ignore {}'.format(name))
                continue
            print('load {} from {}'.format(name, load_ckpt))
            own_state[name].copy_(param)
    
    
    data_loader={}
    data_loader['train'] = FewshotDataset(data_dir['train'],N,K,Q,data_dir['noise_rate']) 
    data_loader['val'] = FewshotDataset(data_dir['val'],N,K,Q,data_dir['noise_rate'])
    data_loader['test'] = FewshotDataset(data_dir['test'],N,K,Q,data_dir['noise_rate'])

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    coder_named_params = list(model.coder.named_parameters())

    for name, param in coder_named_params:
        if name in {'bert_ebd.word_embeddings.weight','bert_ebd.position_embeddings.weight','bert_ebd.token_type_embeddings.weight'}:
            param.requires_grad = False
            pass


    optim_params=[{'params':[p for n, p in coder_named_params 
                    if not any(nd in n for nd in no_decay)],'lr':meta_lr,'weight_decay': weight_decay},
                  {'params': [p for n, p in coder_named_params 
                    if any(nd in n for nd in no_decay)],'lr':meta_lr, 'weight_decay': 0.0},
                ]
       

    meta_optimizer=AdamW(optim_params)
    scheduler = get_linear_schedule_with_warmup(meta_optimizer, num_warmup_steps=warmup_step, num_training_steps=train_iter)

    if fp16:
        from apex import amp
        model, meta_optimizer = amp.initialize(model, meta_optimizer, opt_level='O1')

    iter_loss, iter_right, iter_sample = 0.0, 0.0, 0.0

    model.train()

    for it in range(train_iter):
        meta_loss, meta_right = 0.0, 0.0

        for batch in range(B):
            class_name, support, support_label, query, query_label = data_loader['train'][0]
            loss, right =train_one_batch(batch,class_name,support,support_label,query,query_label,model,steps,task_lr)
            
            meta_loss += loss
            meta_right += right
        
        meta_loss /= B
        meta_right /= B

        meta_optimizer.zero_grad()
        if fp16:
            with amp.scale_loss(meta_loss, meta_optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            meta_loss.backward()
        meta_optimizer.step()
        scheduler.step()
    
        iter_loss += meta_loss
        iter_right += meta_right
        iter_sample += 1 

        sys.stdout.write('step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%'.format(it + 1, iter_loss / iter_sample, 100 * iter_right / iter_sample) + '\r')
        sys.stdout.flush()

        if (it+1)%val_step==0:
            print("")
            iter_loss, iter_right, iter_sample = 0.0,0.0,0.0
            acc = test_model(data_loader['val'], model, val_iter, steps,task_lr)
            print("")
            model.train()
            if acc > best_acc:
                print('Best checkpoint!')
                torch.save({'state_dict': model.state_dict()}, save_ckpt)

                best_acc = acc

    print("\n####################\n")
    print('Finish training model! Best acc: '+str(best_acc))


def eval_model(model,N,K,Q,eval_iter=10000,steps=30,task_lr=1e-2, noise_rate = 0,file_name=None,load_ckpt = None):
    if torch.cuda.is_available(): model = model.cuda()

    if load_ckpt:
        state_dict = torch.load(load_ckpt)['state_dict']
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                # print('ignore {}'.format(name))
                continue
            # print('load {} from {}'.format(name, load_ckpt))
            own_state[name].copy_(param)

    accs=0.0
    model.eval()
    data_loader = FewshotDataset(file_name,N,K,Q,noise_rate)
    tot = {}
    neg = {}
    for it in range(eval_iter):
        net = model
        class_name,support,support_label,query,query_label = data_loader[0]
        _,right = train_one_batch(0,class_name, support, support_label,query,query_label,net,steps,task_lr)
        accs += right 
        for i in class_name:
            if i not in tot:
                tot[i]=1
            else:
                tot[i]+=1
        if right <1:
            for i in class_name:
                if i not in neg:
                    neg[i]=1
                else:
                    neg[i]+=1
        sys.stdout.write('[EVAL] step: {0:4} | accuracy: {1:3.2f}%'.format(it + 1, 100 * accs / (it+1)) + '\r')
        sys.stdout.flush()
    print("")
    print(tot)
    print(neg)
    print("")

    return accs/eval_iter