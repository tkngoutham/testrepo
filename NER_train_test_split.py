#Make the necessary imports
from nltk.tag import pos_tag
from sklearn_crfsuite import CRF, metrics
from sklearn.metrics import make_scorer,confusion_matrix
from pprint import pprint
from sklearn.metrics import f1_score,classification_report
from sklearn.pipeline import Pipeline
import string
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from deep_utils.utils.multi_label_utils.stratify import stratify_train_test_split_multi_label
#deep utils repo https://github.com/pooya-mohammadi/deep_utils
#data repo: https://github.com/practical-nlp/practical-nlp-code/blob/master/Ch5/ 

#data repo 
train_path='/Users/practical-nlp-code/Ch5/Data/conlldata/train.txt'
test_path='/Users/practical-nlp-code/Ch5/Data/conlldata/test.txt'



def load_data(file_path):
    max_seq_len=0
    unique_labels=set()
    X,y,words,tags = [],[],[],[]
    fh = open(file_path)
    for line in fh:
        line = line.strip()
        if "\t" not in line:
            if max_seq_len<len(tags):
                max_seq_len=len(tags)
            X.append(words)
            y.append(tags)
            unique_labels.update(set(tags))
            words,tags = [],[]
        else:
            word, tag = line.split("\t")
            words.append(word)
            tags.append(tag)
    fh.close()
    return X,y,max_seq_len,unique_labels



def call_stratified_sampling(X,y,max_seq_len,test_size=0.2):
    padded_y=[]
    for x_sample,y_sample in zip(X,y):
        O_idx=labels_to_idx['O']
        pad_len=max_seq_len-len(y_sample)
        y_idx=[labels_to_idx[l] for l in y_sample]
        y_sample=y_idx+[O_idx]*pad_len
        padded_y.append(y_sample)

    X=np.array(X)
    padded_y=np.array(padded_y)

    x_train, x_test, y_train, y_test = stratify_train_test_split_multi_label(X, padded_y, test_size, closest_ratio=False)        
    return x_train, x_test, y_train, y_test

 

 
X,y,max_seq_len,unique_labels = load_data(train_path)

labels_to_idx = {k:idx for idx,k in enumerate(unique_labels)}
idx_to_labels = {idx:k for k,idx in labels_to_idx.items()}
x_train, x_test, y_train, y_test = call_stratified_sampling(X, y, max_seq_len ,test_size=0.15)





