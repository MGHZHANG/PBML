import json
from collections import defaultdict
train = defaultdict(list) 
val = defaultdict(list) 
test = defaultdict(list) 
train_classes = [2, 3, 4, 7, 11, 12, 13, 18, 19, 20] 
val_classes = [1, 22, 23, 6, 9] 
test_classes = [0, 5, 14, 15, 8, 10, 16, 17, 21] 

with open('amazon.json', 'r') as f:
    for line in f:
        row = json.loads(line)
        if row['label'] in train_classes:
            train[row['label']].append(row['text'])
        elif row['label'] in val_classes:
            val[row['label']].append(row['text'])
        elif row['label'] in test_classes:
            test[row['label']].append(row['text'])
            
benchmark = "Amazon"
with open(f'{benchmark}/train.json','w') as fout:
    fout.write(json.dumps(train))
    fout.write('\n')
with open(f'{benchmark}/val.json','w') as fout:
    fout.write(json.dumps(val))
    fout.write('\n')
with open(f'{benchmark}/test.json','w') as fout:
    fout.write(json.dumps(test))
    fout.write('\n')