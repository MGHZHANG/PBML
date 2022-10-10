import json
from collections import defaultdict
train = defaultdict(list) 
val = defaultdict(list) 
test = defaultdict(list) 

train_classes = list(range(20))
val_classes = list(range(20,25))
test_classes= list(range(25,41))
test_classes.remove(39) # duplicates

with open('huffpost.json', 'r') as f:
    for line in f:
        row = json.loads(line)
        label, text = row['label'], row['text']
        
        if label in train_classes:
            train[label].append(text)
        elif label in val_classes:
            val[label].append(text)
        elif label in test_classes:
            test[label].append(text)

benchmark = "HuffPost"
with open(f'{benchmark}/train.json','w') as fout:
    fout.write(json.dumps(train))
    fout.write('\n')
with open(f'{benchmark}/val.json','w') as fout:
    fout.write(json.dumps(val))
    fout.write('\n')
with open(f'{benchmark}/test.json','w') as fout:
    fout.write(json.dumps(test))
    fout.write('\n')