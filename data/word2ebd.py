# transform the candidate words into BERT-WORD embeddings

import torch
import json
from transformers import BertTokenizer, BertModel

bert = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
emb_M = bert.embeddings.word_embeddings.weight

def show_tokenized_name(idx2name):
    for key in idx2name:
        name_list = idx2name[key]
        for name in name_list:
             print(tokenizer.tokenize(name))
                
def save_ebd_list(idx2name,outdir):
    cl2ebd_list = dict()
    for cl in idx2name:
        vec_list = []
        names = idx2name[cl]
        for name in names:
            tokenized_name = tokenizer.tokenize(name)
            indexed_name = tokenizer.convert_tokens_to_ids(tokenized_name)
            ebd = torch.mean(emb_M[indexed_name],0).tolist()
            vec_list.append(ebd)
        cl2ebd_list[cl] = vec_list
        
    with open(outdir,'w') as fout:
        fout.write(json.dumps(cl2ebd_list))
        fout.write('\n')

for benchmark in {"Amazon","Reuters","HuffPost"}:
    idx2name = json.load(open(f'data/{benchmark}/candidate_words.json','r'))
    save_ebd_list(idx2name,f'data/{benchmark}/candidate_ebds.json')


# candidate ebds for FewRel (from P-info)

word2vec={} # word： vec
WikiData={} # Pid: dictionary name alias etc.
num2embed={}

def read_info(file_name):
    WikiDatafile=json.load(open(file_name,'r',encoding='utf-8'))
    for relation in WikiDatafile:
        name_list = [] 
        name_list.append(relation['name'].lower())
        for i in range(len(relation['alias'])):
            name_list.append(relation['alias'][i].lower())
        relation['name'] =name_list
        WikiData[relation['id']]= name_list

read_info('data/FewRel/P_info.json')
cl2ebd_list = dict()

for cl in WikiData:
    vec_list = []
    for name in WikiData[cl]:
        name = name.lower()
        name = name.split()
        tokenized_name = []
        for w in name:
            tokenized_name += tokenizer.tokenize(w)
        indexed_name = tokenizer.convert_tokens_to_ids(tokenized_name)
        ebd = torch.mean(emb_M[indexed_name],0).tolist()
        
        vec_list.append(ebd)
    cl2ebd_list[cl] = vec_list

with open('data/FewRel/candidate_ebds.json','w') as fout:
    fout.write(json.dumps(cl2ebd_list))
    fout.write('\n')