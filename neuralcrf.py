#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 19:12:14 2018

@author: sayambhu
"""
import numpy as np
import nltk
from collections import Counter
import re
from nltk.stem import SnowballStemmer

#Actual data
snow = SnowballStemmer('english')  
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
        Sents.append(snow.stem(Dat[i][0]))
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

tag_trainX=[]
pos_trainX=[nltk.pos_tag(x) for x in trainX]
for i in range(len(trainX)):
    tag_trainX.append([pos_trainX[i][j][1] for j in range(len(pos_trainX[i]))])

tag_valX=[]
pos_valX=[nltk.pos_tag(x) for x in valX]
for i in range(len(valX)):
    tag_valX.append([pos_valX[i][j][1] for j in range(len(pos_valX[i]))])

tag_testX=[]
pos_testX=[nltk.pos_tag(x) for x in testX]
for i in range(len(testX)):
    tag_testX.append([pos_testX[i][j][1] for j in range(len(pos_testX[i]))])

Unique=[]
for x in tag_trainX:
    Unique+=x

Uniquetagtr=list(set(Unique))
Uniquetagtr.append('<UNK>')

Unique=[]
for x in trainy:
    Unique+=x

UniqueODT=['O','D','T']


Unique=[]
for x in trainX:
    Unique+=x
Unique_wordtr=list(set(Unique))

Freqwords=Counter(Unique)

Onefreqwords=[words for words in Freqwords if Freqwords[words]==1]

#Some numbers to be removed
WordRemove=Onefreqwords[2:22]

for i in range(len(Unique)):
    for j in range(len(WordRemove)):
        Unique[i] = re.sub(WordRemove[j], '<UNK>',Unique[i] )
     

Unique_wordtr=list(set(Unique))    
mak=0
for i in range(len(Datarr)):
    mak=max(mak,len(Datarr[i]))
mak=75
word2idx = {w: i for i, w in enumerate(Unique_wordtr)}
tag2idx = {t: i for i, t in enumerate(Uniquetagtr)}
ODT2idx={t: i for i, t in enumerate(UniqueODT)}

from keras.preprocessing.sequence import pad_sequences

def senttagtoid(wordsent,tagsent,labelsent):
    sentid=[]
    tagid=[]
    labelid=[]
    for i in range(len(wordsent)):
        if wordsent[i] in word2idx:
            sentid.append(word2idx[wordsent[i]])
            labelid.append(ODT2idx[labelsent[i]])
        else:
            sentid.append(word2idx['<UNK>'])
            labelid.append(ODT2idx['O'])
        if tagsent[i] in tag2idx:
            tagid.append(tag2idx[tagsent[i]])
        else:
            tagid.append(tag2idx['<UNK>'])
        
            
    return sentid,tagid,labelid
Xtr=[]
ytr=[]
Xte=[]
yte=[]
Xval=[]
yval=[]

#test
for i in range(len(trainX)):
    x1,x2,y=senttagtoid(trainX[i],tag_trainX[i],trainy[i])
    Xtr.append(x1)
    ytr.append(y)
    

for i in range(len(valX)):
    x1,x2,y=senttagtoid(valX[i],tag_valX[i],valy[i])
    Xval.append(x1)
    yval.append(y)

for i in range(len(testX)):
    x1,x2,y=senttagtoid(testX[i],tag_testX[i],testy[i])
    Xte.append(x1)
    yte.append(y)
xc=np.concatenate((Xtr,Xval),axis=0)
yc=np.concatenate((ytr,yval),axis=0)
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional

from keras_contrib.layers import CRF

Xtr = pad_sequences(maxlen=mak, sequences=xc, padding="post", value=word2idx['<UNK>'])
Xte = pad_sequences(maxlen=mak, sequences=Xte, padding="post", value=word2idx['<UNK>'])
ytr=pad_sequences(maxlen=mak, sequences=yc, padding="post", value=ODT2idx["O"])
yte=pad_sequences(maxlen=mak, sequences=yte, padding="post", value=ODT2idx["O"])

from keras.utils import to_categorical

ytr = [to_categorical(i, num_classes=len(ODT2idx)) for i in ytr]
ytes = [to_categorical(i, num_classes=len(ODT2idx)) for i in yte]

input = Input(shape=(mak,))
model = Embedding(input_dim=len(Unique_wordtr), output_dim=100,
                  input_length=mak, mask_zero=True)(input)  # 20-dim embedding
model = Bidirectional(LSTM(units=50, return_sequences=True,
                           recurrent_dropout=0.1))(model)  # variational biLSTM
#model = TimeDistributed(Dense(50, activation="relu"))(model)  # a dense layer as suggested by neuralNer
crf = CRF(len(UniqueODT))  # CRF layer
out = crf(model)  # output

model = Model(input, out)

model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])

model.summary()
history = model.fit(Xtr, np.array(ytr), batch_size=25, epochs=5,validation_split=0.05, verbose=1)


#Testing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
conf=[]
precal=[]
with open('outlstm.txt', 'w') as f:
    for i in range(len(Xte)):
        p = model.predict(np.array([Xte[i]]))
        p = np.argmax(p, axis=-1)
        true = np.argmax(ytes[i], -1)
        for j in range(len(true)):
            f.write(Unique_wordtr[Xte[i][j]])
            print(Unique_wordtr[Xte[i][j]]," ")
            f.write(" ")
            f.write(UniqueODT[true.tolist()[j]])
            print(UniqueODT[true.tolist()[j]]," ")
            f.write(" ")
            f.write(UniqueODT[p[0].tolist()[j]])
            print(UniqueODT[p[0].tolist()[j]," \n")
            f.write("\n")
        print("\n")
        f.write("\n")
        conf.append(confusion_matrix(true.tolist(),p[0].tolist(),labels=[0,1,2]))
        precal.append(precision_recall_fscore_support(true.tolist(), p[0].tolist(), average='micro'))

Confusion=np.sum(conf,axis=0)
RecallD=Confusion[1,1]/np.sum(Confusion[:,1])
PrecisionD=Confusion[1,1]/np.sum(Confusion[1,:])

RecallT=Confusion[2,2]/np.sum(Confusion[:,2])
PrecisionT=Confusion[2,2]/np.sum(Confusion[2,:])












