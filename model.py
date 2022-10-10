# -*- coding: utf-8 -*-
import os
import json
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertModel,BertTokenizer,BertForMaskedLM

def gelu(x):
    return x  * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class PBML(nn.Module):
    def __init__(self, B, N, K, max_length, data_dir):
        nn.Module.__init__(self)

        self.batch = B
        self.n_way = N
        self.k_shot = K
        self.max_length = max_length
        self.data_dir = data_dir
        self.hidden_size = 768 # bert-base  

        self.cost = nn.NLLLoss()
        self.coder = BERT(N,max_length,data_dir) 
        self.initializer = Initializer(N,K, data_dir) 
        
        self.W = [None] * self.batch # label word embedding matrix

    def loss(self,logits,label):
        return self.cost(logits.log(),label.view(-1)) 

    def accuracy(self,logits,label):
        label = label.view(-1)
        _, pred = torch.max(logits,1) 
        return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))

    def forward(self, inputs, W): # inputs: [N*K or total_Q, hidden_size] 
                                  # W: [n_way, hidden_size]  
        # s2w
        logits_for_instances = F.linear(inputs, W, bias=None) #[N*K or total_Q or 1 ,n_way]
        # w2s
        logits_for_classes = F.linear(W, torch.mean(inputs.view(self.n_way, inputs.shape[0]//self.n_way,768),dim=1), bias=None)

        return F.softmax(logits_for_instances,dim=-1), F.softmax(logits_for_classes,dim=-1)

    def get_info(self,class_names): # list of class_name
        return self.initializer.get_embedding(class_names) # [N * [candidate word embeddings]]

    def prework(self,candidate_word_embeddings): # meta-info: [N, hidden_size]
                                                # support:   [N*K, bert_size]
        return self.initializer(candidate_word_embeddings)

class Initializer(nn.Module):
    def __init__(self, N, K, data_dir):
        super(Initializer,self).__init__()
        self.n_way = N
        self.k_shot = K
        self.embedding_dim = 768

        candidate_info = data_dir['candidates'] # candidate word info
        if candidate_info is None or not os.path.isfile(candidate_info):
            raise Exception("[ERROR] candidate words information file doesn't exist")

        self.cl2embed = json.load(open(candidate_info,'r')) # {class_name: candidate word embeddings}

        for key in self.cl2embed.keys():
            self.cl2embed[key] = torch.Tensor(self.cl2embed[key]).cuda()
        
        self.miml_trans = nn.Linear(768,768)

    def get_embedding(self, class_names):
        # read candidate word embeddings from the class name 
        res = []
        for i in range(len(class_names)): 
            class_name = class_names[i]
            vec_list = self.cl2embed[class_name].float()
            res.append(vec_list) 
        return res  # [N * [candidate word embeddings]]

    def forward(self, inputs): # inputs: [N * [candidate word embeddings]]
        # average pooling
        W = torch.zeros(len(inputs), self.embedding_dim).cuda()
        for idx in range (len(inputs)):
            # W[idx] = torch.mean(inputs[idx], 0).requires_grad_(True) # [hidden_size] candidates mean pooler
            W[idx] = self.miml_trans(torch.mean(inputs[idx], 0))
            # W[idx] = inputs[idx][0].requires_grad_(True) # [hidden_size] without kg
        if self.k_shot == 1:
            W = F.normalize(W,dim=-1)
        elif self.k_shot == 5:
            W = 0.5 * F.normalize(W,dim=-1)

        return W
               
class BERT(nn.Module):
    def __init__(self, N, max_length, data_dir, blank_padding=True):
        super(BERT,self).__init__()
        self.cuda = torch.cuda.is_available()
        self.n_way = N
        self.max_length = max_length
        self.blank_padding = blank_padding
        self.pretrained_path = 'bert-base-uncased'

        bert_model = BertModel.from_pretrained(self.pretrained_path)
        self.get_extended_attention_mask = bert_model.get_extended_attention_mask
        self.bert_ebd = bert_model.embeddings
        self.bert_encoder = bert_model.encoder

        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_path)
        self.dropout = nn.Dropout(data_dir['pb_dropout'])
        self.benchmark = data_dir['benchmark']
        
        mlm = BertForMaskedLM.from_pretrained(self.pretrained_path)
        D = mlm.cls.state_dict()
        (pred_bias, tf_dw, tf_db, tf_lnw, tf_lnb, dec_w, dec_b) = (D['predictions.bias'],
                                                            D['predictions.transform.dense.weight'],
                                                            D['predictions.transform.dense.bias'], 
                                                            D['predictions.transform.LayerNorm.weight'], 
                                                            D['predictions.transform.LayerNorm.bias'], 
                                                            D['predictions.decoder.weight'],
                                                            D['predictions.decoder.bias'])
        self.LayerNorm = nn.LayerNorm(768,eps = 1e-12) 
        self.LayerNorm.weight.data, self.LayerNorm.bias.data = tf_lnw,tf_lnb
        self.tf_dense = nn.Linear(768,768)
        self.tf_dense.weight.data,self.tf_dense.bias.data = tf_dw,tf_db

        # soft template params
        if self.benchmark == "FewRel":
            self.soft_prompt = nn.Parameter(torch.rand(4,768))
            soft_token = ['is', '[MASK]', 'of', '.']   
        elif self.benchmark == "HuffPost":
            self.soft_prompt = nn.Parameter(torch.rand(4,768))
            soft_token = ['article', 'about', '[MASK]','.']
        if self.benchmark == "Reuters":
            self.soft_prompt = nn.Parameter(torch.rand(5,768))
            soft_token = ['article','is','about', '[MASK]',':'] 
        if self.benchmark == "Amazon":
            self.soft_prompt = nn.Parameter(torch.rand(5,768))
            soft_token = ['article','is','about', '[MASK]',':']
        
        soft_token_id = self.tokenizer.convert_tokens_to_ids(soft_token)
        for i in range(len(soft_token)):
            self.soft_prompt.data[i] = self.bert_ebd.word_embeddings.weight.data[soft_token_id[i]]

    def forward(self,inputs):
        if self.benchmark == "FewRel":
            return self.forward_FewRel(inputs)
        elif self.benchmark == "HuffPost":
            return self.forward_HuffPost(inputs)
        elif self.benchmark == "Reuters":
            return self.forward_Reuters(inputs)
        elif self.benchmark == "Amazon":
            return self.forward_Amazon(inputs)

    def forward_FewRel(self,inputs): # [raw_tokens_dict * (N*K or total_Q)]
        input_ebds, MASK_INDs,att_masks,outputs = [],[],[],[]
        for _ in inputs: 
            indexed_token, indexed_head, indexed_tail, avai_len = self.tokenize_FewRel(_)
            after_ebd_text = self.bert_ebd.word_embeddings(indexed_token) # [1,avai_len] ——> [1, avai_len, 768]
            after_ebd_head = self.bert_ebd.word_embeddings(indexed_head)  # [1,len_head] ——> [1, len_head, 768]
            after_ebd_tail = self.bert_ebd.word_embeddings(indexed_tail)  # [1,len_tail] ——> [1, len_tail, 768]
            input_ebd = torch.cat((after_ebd_text, after_ebd_head, self.soft_prompt[:3].unsqueeze(0)),1) # text head is [mask] of 

            MASK_INDs.append(avai_len + indexed_head.shape[-1] + 1) 
            input_ebd = torch.cat((input_ebd, after_ebd_tail, self.soft_prompt[3].unsqueeze(0).unsqueeze(0), self.bert_ebd.word_embeddings(torch.tensor(102).cuda()).unsqueeze(0).unsqueeze(0) ),1) # text head is [mask] of tail . [SEP]
            
            # mask calculation
            att_mask = torch.zeros(1,self.max_length)
            if self.cuda: att_mask = att_mask.cuda()
            att_mask[0][:input_ebd.shape[1]]=1 # [1, max_length]

            # padding tensor
            while input_ebd.shape[1] < self. max_length:
                input_ebd = torch.cat((input_ebd, self.bert_ebd.word_embeddings(torch.tensor(0).cuda()).unsqueeze(0).unsqueeze(0)), 1)

            input_ebd = input_ebd[:,:self.max_length,:]
            input_ebds.append(input_ebd)

            input_shape = att_mask.size()
            device = indexed_token.device
            
            extented_att_mask = self.get_extended_attention_mask(att_mask, input_shape,device) 
            att_masks.append(extented_att_mask)

        input_ebds = torch.cat(input_ebds,0) # [N*K, max_length，768]
        tensor_masks = torch.cat(att_masks,0) # [N*K, max_length] then extend
        sequence_output= self.bert_encoder(self.bert_ebd(inputs_embeds = input_ebds) , attention_mask = tensor_masks).last_hidden_state # [N*K, max_length, bert_size]

        
        for i in range(input_ebds.size(0)): 
            outputs.append(self.entity_start_state(MASK_INDs[i],sequence_output[i]))
            # [[1,bert_size*2] * (N*K)]
        tensor_outputs = torch.cat(outputs,0)  # [N*K,bert_size*2=hidden_size] 

        # dropout 
        tensor_outputs = self.dropout(tensor_outputs) 

        return tensor_outputs

    def forward_HuffPost(self,inputs): # [sentence * (N*K or total_Q)]
        input_ebds, MASK_INDs,att_masks,outputs = [],[],[],[]
        for _ in inputs: 
            indexed_token, avai_len = self.tokenize_HuffPost(_)
            after_ebd = self.bert_ebd.word_embeddings(indexed_token) # [1,avai_lens, 768]
            after_ebd = torch.cat((after_ebd, self.soft_prompt.unsqueeze(0)),1) # cat on dim1 : [1,avai_lens, 768]  and [1,soft_token_lens, 768]
            after_ebd = torch.cat((after_ebd, self.bert_ebd.word_embeddings(torch.tensor(102).cuda()).unsqueeze(0).unsqueeze(0)  ),1) 

            MASK_INDs.append(avai_len+2) # article is about
            att_mask = torch.zeros(1,self.max_length)
            if self.cuda: att_mask = att_mask.cuda()
            att_mask[0][:after_ebd.shape[1]]=1 # [1, max_length]
            # padding tensor
            while after_ebd.shape[1] < self. max_length:
                after_ebd = torch.cat((after_ebd, self.bert_ebd.word_embeddings(torch.tensor(0).cuda()).unsqueeze(0).unsqueeze(0)), 1)
            after_ebd = after_ebd[:,:self.max_length,:]
            input_ebds.append(after_ebd)

            input_shape = att_mask.size()
            device = indexed_token.device

            extented_att_mask = self.get_extended_attention_mask(att_mask, input_shape,device) 
            att_masks.append(extented_att_mask)
    
        input_ebds = torch.cat(input_ebds,0) # [N*K, max_length，768]
        tensor_masks = torch.cat(att_masks,0) 
        sequence_output= self.bert_encoder(self.bert_ebd(inputs_embeds = input_ebds) , attention_mask = tensor_masks).last_hidden_state # [N*K, max_length, bert_size]
        for i in range(input_ebds.size(0)): 
            outputs.append(self.entity_start_state(MASK_INDs[i],sequence_output[i]))

        tensor_outputs = torch.cat(outputs,0)  
        # dropout 
        tensor_outputs = self.dropout(tensor_outputs)
        return tensor_outputs

    def forward_Amazon(self,inputs): # [sentence * (N*K or total_Q)]
        input_ebds, MASK_INDs,att_masks,outputs = [],[],[],[]
        for _ in inputs: 
            indexed_token, avai_len = self.tokenize_Amazon(_)
            after_ebd = self.bert_ebd.word_embeddings(indexed_token) # [1,avai_lens, 768]
            after_ebd = torch.cat((self.bert_ebd.word_embeddings(torch.tensor(101).cuda()).unsqueeze(0).unsqueeze(0), self.soft_prompt.unsqueeze(0) , after_ebd),1) # cat on dim1 : [1,avai_lens, 768]  and [1,soft_token_lens, 768]
            MASK_INDs.append(4) 

            att_mask = torch.zeros(1,self.max_length)
            if self.cuda: att_mask = att_mask.cuda()
            att_mask[0][:after_ebd.shape[1]] = 1 # [1, max_length]

            # padding tensor
            while after_ebd.shape[1] < self. max_length:
                after_ebd = torch.cat((after_ebd, self.bert_ebd.word_embeddings(torch.tensor(0).cuda()).unsqueeze(0).unsqueeze(0)), 1)
            after_ebd = after_ebd[:,:self.max_length,:]
            input_ebds.append(after_ebd)

            input_shape = att_mask.size()
            device = indexed_token.device

            extented_att_mask = self.get_extended_attention_mask(att_mask, input_shape,device) 
            att_masks.append(extented_att_mask)


        
        input_ebds = torch.cat(input_ebds,0) # [N*K, max_length，768]
        tensor_masks = torch.cat(att_masks,0) 
        sequence_output= self.bert_encoder(self.bert_ebd(inputs_embeds = input_ebds) , attention_mask = tensor_masks).last_hidden_state # [N*K, max_length, bert_size]

        for i in range(input_ebds.size(0)):
            outputs.append(self.entity_start_state(MASK_INDs[i],sequence_output[i]))
        tensor_outputs = torch.cat(outputs,0)  # [N*K,bert_size*2=hidden_size] 
        # dropout 
        tensor_outputs = self.dropout(tensor_outputs)

        return tensor_outputs   

    def forward_Reuters(self,inputs): # [sentence * (N*K or total_Q)]
        input_ebds, MASK_INDs,att_masks,outputs = [],[],[],[]
        for _ in inputs: 
            indexed_token, avai_len = self.tokenize_Reuters(_)
            after_ebd = self.bert_ebd.word_embeddings(indexed_token) # [1,avai_lens, 768]
            after_ebd = torch.cat((self.bert_ebd.word_embeddings(torch.tensor(101).cuda()).unsqueeze(0).unsqueeze(0), self.soft_prompt.unsqueeze(0) , after_ebd),1) # cat on dim1 : [1,avai_lens, 768]  and [1,soft_token_lens, 768]
            MASK_INDs.append(4) 

            att_mask = torch.zeros(1,self.max_length)
            if self.cuda: att_mask = att_mask.cuda()
            att_mask[0][:after_ebd.shape[1]] = 1 # [1, max_length]
            # padding tensor
            while after_ebd.shape[1] < self. max_length:
                after_ebd = torch.cat((after_ebd, self.bert_ebd.word_embeddings(torch.tensor(0).cuda()).unsqueeze(0).unsqueeze(0)), 1)
            after_ebd = after_ebd[:,:self.max_length,:]
            input_ebds.append(after_ebd)

            input_shape = att_mask.size()
            device = indexed_token.device

            extented_att_mask = self.get_extended_attention_mask(att_mask, input_shape,device) 
            att_masks.append(extented_att_mask)

        input_ebds = torch.cat(input_ebds,0) # [N*K, max_length，768]
        tensor_masks = torch.cat(att_masks,0)
        sequence_output= self.bert_encoder(self.bert_ebd(inputs_embeds = input_ebds) , attention_mask = tensor_masks).last_hidden_state # [N*K, max_length, bert_size]
        for i in range(input_ebds.size(0)):
            outputs.append(self.entity_start_state(MASK_INDs[i],sequence_output[i]))
        tensor_outputs = torch.cat(outputs,0)
        # dropout 
        tensor_outputs = self.dropout(tensor_outputs)
        return tensor_outputs

    def entity_start_state(self,MASK_IND,sequence_output): #  sequence_output: [max_length, bert_size]
        if MASK_IND >= self.max_length:
            MASK_IND = 0
        res = sequence_output[MASK_IND]
        res = self.LayerNorm(gelu(self.tf_dense(res)))

        return res.unsqueeze(0) # [1, hidden_size]

    def tokenize_FewRel(self,inputs): #input: raw_tokens_dict
        tokens = inputs['tokens']
        pos_head = inputs['h'][2][0]
        pos_tail = inputs['t'][2][0]

        re_tokens,cur_pos = ['[CLS]',],0

        for token in tokens:
            token=token.lower() 
            if cur_pos == pos_head[0]: 
                re_tokens.append('[unused0]')
            if cur_pos == pos_tail[0]: 
                re_tokens.append('[unused1]')

            re_tokens+=self.tokenizer.tokenize(token)

            if cur_pos==pos_head[-1]-1: re_tokens.append('[unused2]') 
            if cur_pos==pos_tail[-1]-1: re_tokens.append('[unused3]')
            
            cur_pos+=1
        re_tokens.append('[SEP]')

        head = []
        tail = []
        for cur_pos in range(pos_head[0],pos_head[-1]):
            head += self.tokenizer.tokenize(tokens[cur_pos])
        for cur_pos in range(pos_tail[0],pos_tail[-1]):
            tail += self.tokenizer.tokenize(tokens[cur_pos])
        
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(re_tokens)
        indexed_head = self.tokenizer.convert_tokens_to_ids(head)
        indexed_tail = self.tokenizer.convert_tokens_to_ids(tail)
        avai_len = len(indexed_tokens)

        indexed_tokens = torch.tensor(indexed_tokens).unsqueeze(0).long() 
        indexed_head = torch.tensor(indexed_head).unsqueeze(0).long() 
        indexed_tail = torch.tensor(indexed_tail).unsqueeze(0).long() 

        if self.cuda: indexed_tokens,indexed_head,indexed_tail = indexed_tokens.cuda(), indexed_head.cuda(), indexed_tail.cuda()
        return indexed_tokens, indexed_head, indexed_tail, avai_len
    
    def tokenize_HuffPost(self,inputs): #input: sentence type: list
        tokens = ['[CLS]']
        for token in inputs:
            token = token.lower()
            tokens += self.tokenizer.tokenize(token)
        tokens.append('[SEP]')

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        avai_len = len(indexed_tokens)

        
        indexed_tokens = torch.tensor(indexed_tokens).unsqueeze(0).long()
        if self.cuda: indexed_tokens =  indexed_tokens.cuda()
        return indexed_tokens, avai_len
    
    def tokenize_Amazon(self,inputs): #input: sentence type: list
        
        tokens = []
        for token in inputs:
            token = token.lower()
            tokens += self.tokenizer.tokenize(token)
        tokens.append('[SEP]')

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        avai_len = len(indexed_tokens)
        
        indexed_tokens = torch.tensor(indexed_tokens).unsqueeze(0).long() 
        if self.cuda: indexed_tokens =  indexed_tokens.cuda()
        return indexed_tokens, avai_len

    def tokenize_Reuters(self,inputs): #input: sentence type: list
        tokens = []
        for token in inputs:
            token = token.lower()
            tokens += self.tokenizer.tokenize(token)
        tokens.append('[SEP]')

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        avai_len = len(indexed_tokens)

        indexed_tokens = torch.tensor(indexed_tokens).unsqueeze(0).long()
        if self.cuda: indexed_tokens =  indexed_tokens.cuda()
        return indexed_tokens, avai_len