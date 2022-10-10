# PBML
Our goal is to improve few-shot text classification performance by combining meta-learning with prompting. Our proposed model PBML assigns template&encoder learning to the meta-learner and label words learning to base-learners, resplectively. We conduct extensive experiments on four widely-used text classification datasets: FewRel, HuffPost headlines, Reuters, and Amazon product data.

# Dataset
### FewRel 
A dataset for few-shot relation classification, containing 100 relations. Each statement has an entity pair and is annotated with the corresponding relation. The position of the entity pair is given, and the goal is to predict the correct relation based on the context. The 100 relations are split into 64, 16, and 20 for training, validation, and test, respectively. 

### HuffPost headlines 
A dataset for topic classification. It contains news headlines published on HuffPost between 2012 and 2018 (Misra, 2018). The 41 topics are split into 20, 5, 16 for training, validation and test respectively. These headlines are shorter and more colloquial texts.

### Reuters-2157 
A dataset of Reuters articles over 31 classes (Lewis, 1997), which are split into 15, 5, 11 for training, validation and test respectively. These articles are longer and more grammatical texts.

### Amazon product data 
A dataset contains customer reviews from 24 product categories. Our goal is to predict the product category based on the content of the review. The 24 classes are split into 10, 5, 9 for training, validation and test respectively.


## Download 
You may download the FewRel training data (JSON file named train_wiki.json or train.json) and validation data (JSON file named val_wiki.json or val.json)from https://github.com/thunlp/FewRel/tree/master/data or https://github.com/thunlp/MIML/tree/main/data (same dataset while marking entity positions under different conventions. Our word adopt the latter version.) The FewRel testing set is not publicly available for fair comparison, so you need to visit the benchmark website: https://thunlp.github.io/fewrel.html for testing on test set.

For HuffPost headlines, Reuters, and Amazon product, you may download the dataset processed by Bao et al.,(2020), from https://people.csail.mit.edu/yujia/files/distributional-signatures/data.zip Then you need to run the scripts named `split_xxx.py` in `PBML/data` to split the data into train, val and test set.

You need to put the source data into the corresponding folder of `PBML/data`


# Candidate label words

### Format
+ `data/FewRel/P-info.json` provides for each relation, a list of alias, serving as candidate words. 
+ For HuffPost, Reuters, and Amazon, `candidate_words.json` should contain candidate words of each class (you can define your own candidate words). 
+ `candidate_ebds.json` contains candidate word embeddings of each class. (you may run `data/word2ebd.py` to obtain candidate embeddings)


# Code
+ `train.py` contains fast-tuning model and meta-traning framework
+ `model.py` contains the overall model architechture.
+ Run `main.py` to call the above two files and start meta-training. 

# Requirements
+ Pytorch>=0.4.1
+ Python3
+ numpy
+ transformers
+ json
+ apex (https://github.com/NVIDIA/apex)
