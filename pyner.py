#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 00:40:01 2018

@author: sayambhu
"""

import numpy as np
from gensim.models import Word2Vec
import nltk

#import sklearn
import pycrfsuite






#Actual data
    
a=open("ner.txt").read()

Dat=a.splitlines()

for i in range(len(Dat)):
    Dat[i]=Dat[i].split()
    

Datarr=[]
Labelarr=[]
i=0
e_size=100
while i<len(Dat):
    #print(i)
    Sents=[]
    Sentlabels=[]
    while len(Dat[i])!=0:
        Sents.append(Dat[i][0])
        Sentlabels.append(Dat[i][1])
        i+=1
    i+=1
    Datarr.append(Sents)
    Labelarr.append(Sentlabels)

def TrTeDiv82(X,y,i,k):
    
    nrow=len(X)
    d=int(nrow/k)
    testX=X[(d*i):(d*(i+1))]
    testy=y[(d*i):(d*(i+1))]
    trainX=X.copy()
    trainy=y.copy()
    del trainX[(d*i):(d*(i+1))]
    del trainy[(d*i):(d*(i+1))]
    return [testX,testy,trainX,trainy]

def vali(X,y,testindex,valindex):
    [testX,testy,trainX,trainy]=TrTeDiv82(X,y,testindex,10)
    [valX,valy,trainX,trainy]=TrTeDiv82(trainX,trainy,valindex,9)
    
    return [valX,valy,testX,testy,trainX,trainy]

#Create the validation and train and test sets
[valX,valy,testX,testy,trainX,trainy]=vali(Datarr,Labelarr,9,8)
#Saving a file with word and tag only

def save_train_tag_txt(X,y, file_name):
    with open(file_name, 'w') as f:
        for i,sents in enumerate(X):
            tags=nltk.pos_tag(sents)
            for j,words in enumerate(sents):
                f.write(tags[j][1])
                f.write(" ")
                f.write(words)
                f.write(" ")
                f.write(y[i][j])
                f.write("\n")
            f.write("\n")
 
#Saving a model with tag and embedding           
def save_train_emb_tag_txt(X_emb,X,y, file_name):
    with open(file_name, 'w') as f:
        for i,sents in enumerate(X):
            tags=nltk.pos_tag(sents)
            for j,words in enumerate(sents):
                f.write(tags[j][1])
                f.write(" ")
                f.write(str(words[0].isupper()))
                f.write(" ")
                f.write(words.lower())
                f.write(" ")
                f.write(' '.join(str(d) for d in X_emb[i][j]))
                f.write(" ")
                f.write(y[i][j])
                f.write("\n")
            f.write("\n")            
def egen(X,wvecs):
    vecs=[]
    for sents in X:
        sent=[]
        for words in sents:
            try:
                sent.append(wvecs.wv[words])
            except KeyError:
                sent.append(np.zeros([e_size,1]))
        vecs.append(sent)
    return vecs


e_size=100

wvecs=Word2Vec(sentences=trainX, size=e_size,window=5,min_count=1,workers=4)
emb_train=egen(trainX,wvecs)
emb_val=egen(valX,wvecs)
emb_test=egen(testX,wvecs)

# =============================================================================
# save_train_tag_txt(trainX,trainy, "train_tag_word.txt")
# save_train_tag_txt(valX,valy, "val_tag_word.txt")
# save_train_tag_txt(testX,testy, "test_tag_word.txt")
# =============================================================================


save_train_emb_tag_txt(emb_train,trainX,trainy, "train_word_tag_emb.txt")
save_train_emb_tag_txt(emb_val,valX,valy, "val_word_tag_emb.txt")
save_train_emb_tag_txt(emb_test,testX,testy, "test_word_tag_emb.txt")


# =============================================================================
# train_crf_wordtag=open("train_tag_word.txt").read()
# test_crf_wordtag=open("test_tag_word.txt").read()
# =============================================================================

train_crf_wordtagemb=open("train_word_tag_emb.txt").read().splitlines()
test_crf_wordtagemb=open("test_word_tag_emb.txt").read().splitlines()
train_crf_wordtagemb=[x.split() for x in train_crf_wordtagemb]
test_crf_wordtagemb=[x.split() for x in test_crf_wordtagemb]


trainer = pycrfsuite.Trainer(verbose=False)

x_train =[]
y_train =[]

x_test=[]
y_test=[]

x_try=[]
y_try=[]        

for i in range(len(train_crf_wordtagemb)):
    
    if len(train_crf_wordtagemb[i])==0:
        x_train.append(x_try)
        y_train.append(y_try)
        x_try=[]
        y_try=[]
    else:
        x_try.append(train_crf_wordtagemb[i][0:len(train_crf_wordtagemb[i])-1])
        y_try.append(train_crf_wordtagemb[i][len(train_crf_wordtagemb[i])-1])

x_try=[]
y_try=[] 

for i in range(len(test_crf_wordtagemb)):
    
    if len(test_crf_wordtagemb[i])==0:
        x_test.append(x_try)
        y_test.append(y_try)
        x_try=[]
        y_try=[]
    else:
        x_try.append(test_crf_wordtagemb[i][0:len(test_crf_wordtagemb[i])-1])
        y_try.append(test_crf_wordtagemb[i][len(test_crf_wordtagemb[i])-1])   


#Try word 2 feature
# =============================================================================
# def word2feature(word):
#     features={"POS":"NN",
#               "Capital":True,
#               "Word":"The",
#               "Embdeding":0
#             }
#     features.update({
#             "POS":word[0],
#             "Capital":word[1],
#             "Word":word[2],
#             "Embdeding":word[3:(len(word))]
#             
#             })
#     return features
# 
# 
# x_trainf=[]
# for sents in x_train:
#     x_try=[]
#     for words in sents:
#         x_try.append(word2feature(words))
#     x_trainf.append(x_try)
#         
# x_testf=[]
# for sents in x_test:
#     x_try=[]
#     for words in sents:
#         x_try.append(word2feature(words))
#     x_testf.append(x_try)
# =============================================================================






algorithm = {'lbfgs', 'l2sgd', 'ap', 'pa', 'arow'}
graphModel={'crf1d',}
trainer.select('pa',type='crf1d')
for i in range(len(x_train)):
    trainer.append(x_train[i],y_train[i])
    
# =============================================================================
# for i in range(len(x_test)):
#     trainer.append(x_test[i],y_test[i],group=1)
# =============================================================================

trainer.train( 'pycrfmodel')

tagger = pycrfsuite.Tagger()
tagger.open('pycrfmodel')
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

conf=[]
precal=[]
with open('outcrf.txt', 'w') as f:
    for i in range(len(x_test)):
        predict=tagger.tag(x_test[i])
        for j in range(len(predict)):
            f.write(x_test[i][j][2])
            f.write(" ")
            f.write(y_test[i][j])
            f.write(" ")
            f.write(predict[j])
            f.write("\n")
        f.write("\n")
        conf.append(confusion_matrix(y_test[i],predict,labels=['O','D','T']))
        precal.append(precision_recall_fscore_support(y_test[i], predict, average='macro'))

avgconf=np.sum(conf,axis=0)  
precalarr=np.zeros([len(precal),4])
for i in range(len(precal)):
    for j in range(4):
        precalarr[i,j]=precal[i][j]

avgprecal=np.sum(precalarr,axis=0)/len(precal)
    
RecallD=avgconf[1,1]/np.sum(avgconf[:,1])
PrecisionD=avgconf[1,1]/np.sum(avgconf[1,:])

RecallT=avgconf[2,2]/np.sum(avgconf[:,2])
PrecisionT=avgconf[2,2]/np.sum(avgconf[2,:])



















