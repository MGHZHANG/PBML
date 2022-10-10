# -*- coding: utf-8 -*-
import os
import json
import random
import torch
import torch.utils.data as data

class FewshotDataset(data.Dataset):
    def __init__(self,file_name,N,K,Q,noise_rate):
        super(FewshotDataset,self).__init__()
        if not os.path.isfile(file_name):
            raise Exception("[ERROR] Data file doesn't exist")
        self.json_data = json.load(open(file_name,'r',encoding='utf-8'))
        self.classes = list(self.json_data.keys())
        self.N, self.K, self.Q = N,K,Q
        self.noise_rate = noise_rate

    def __len__(self):
        return 1000000000

    def __getitem__(self,index):
        N, K, Q = self.N, self.K, self.Q
        class_name = random.sample(self.classes,N) # N categories
        support, support_label, query, query_label = [],[],[],[]
        for i,name in enumerate(class_name):
            cl = self.json_data[name]
            samples = random.sample(cl,K+Q)
            for j in range(K):
                support.append([samples[j],i])
            for j in range(K,K+Q):
                query.append([samples[j],i])

        query=random.sample(query,N*Q) # shuffle query order

        for i in range(N*K):
            support_label.append(support[i][1])
            support[i]=support[i][0]

        for i in range(N*Q):
            query_label.append(query[i][1])
            query[i]=query[i][0]

        if self.noise_rate>0: # replace support instance with noised instance from other categories
            other_classes=[]
            for _ in self.classes:
                if _ not in class_name:
                    other_classes.append(_)
            for i in range(N*K):
                if(random.randint(1,10)<=self.noise_rate):
                    noise_name=random.sample(other_classes,1)
                    cl=self.json_data[noise_name[0]]
                    support[i]=random.sample(cl,1)[0]

        support_label = torch.tensor(support_label).long()
        query_label = torch.tensor(query_label).long()
        
        if torch.cuda.is_available():support_label,query_label=support_label.cuda(),query_label.cuda()
        return class_name,support,support_label,query,query_label



