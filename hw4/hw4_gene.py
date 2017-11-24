import os, sys, csv
import numpy
import pandas
from collections import OrderedDict

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM
#from keras.layers import TimeDistributed
#from keras.preprocessing import sequence
#from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint

# fix random seed for reproducibility
# numpy.random.seed(7)

max_sentence =   40
batch_       =   32
voca_filter  =    5
thr          = 0.25

# readin
# vocabulary
X_temp = pandas.read_csv( sys.argv[1], sep='\n', header=None).values

vocabulary = numpy.load('vocabulary.npy')
vocabulary = OrderedDict( zip( vocabulary, range(vocabulary.shape[0])) )
# testing
X_test = numpy.zeros( ( X_temp.shape[0], max_sentence), dtype=int)

no_found_int = vocabulary.get('<unk>')
for i in range( X_temp.shape[0] ):
    count = max_sentence - 1
    for word in range( len( X_temp[i][0].split(' ') ) - 1, -1, -1):
        temp_word = X_temp[i][0].split(' ')[word]
        if temp_word != '' and count>(-1):
            # not found in vocabulary
            if vocabulary.get(temp_word) == None:
                X_test[i][count] = no_found_int
            else:
                X_test[i][count] = vocabulary.get(temp_word)
            count -= 1

# for debugging
print('X_test(samples, max_sentence):', X_test.shape)
print('--------------------------------')


# Start testing
model = load_model(sys.argv[3])
print(model.summary())
y = model.predict( X_test, batch_size=128, verbose=1)
print(y)

with open(sys.argv[2], 'w', newline='') as file:
    for i in range(y.shape[0]):
        if y[i] > (0.5+thr):
            file.write( '1 +++$+++ ' + X_temp[i][0] + '\n' )
        if y[i] < (0.5-thr):
            file.write( '0 +++$+++ ' + X_temp[i][0] + '\n' )
