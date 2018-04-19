# NLU-Assignment-3
The third assignment of NLU which is for Named Entity Recognition


pyner uses the pycrf suite.
neuralcrf uses keras contrib CRF model.

These will have to be installed for the programs to run

For pycrfsuite: pip install python-crfsuite

for keras contrib : sudo pip install git+https://www.github.com/keras-team/keras-contrib.git

Both codes take it in the ner tagged input file named "ner.txt" and gives as output the test file which contains each individual token in each line along with their corresponding true and predicted label respectively.

