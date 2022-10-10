import json
from collections import defaultdict

train_reuters = defaultdict(list) 
val_reuters = defaultdict(list) 
test_reuters = defaultdict(list) 

train_classes = list(range(15)) 
val_classes = list(range(15,20)) 
test_classes = list(range(20,31)) 

with open('reuters.json', 'r') as f:
    for line in f:
        row = json.loads(line)
        label, text = row['label'], row['text']
        for i in range(len(text)):
            if text[i] == 'mln':
                text[i] = 'million'
            elif text[i] == 'pct':
                text[i] = 'percent'
            elif text[i] == 'dlrs':
                text[i] = 'dollars'
                
        if label in train_classes:
            train_reuters[label].append(text)
        elif label in val_classes:
            val_reuters[label].append(text)
        elif label in test_classes:
            test_reuters[label].append(text)

benchmark = "Reuters"
with open(f'{benchmark}/train.json','w') as fout:
    fout.write(json.dumps(train_reuters))
    fout.write('\n')
with open(f'{benchmark}/val.json','w') as fout:
    fout.write(json.dumps(val_reuters))
    fout.write('\n')
with open(f'{benchmark}/test.json','w') as fout:
    fout.write(json.dumps(test_reuters))
    fout.write('\n')