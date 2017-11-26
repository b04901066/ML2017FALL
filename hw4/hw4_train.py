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
voca_filter  =   10

# readin
X_temp = pandas.read_csv( sys.argv[1], sep='\s\+\+\+\$\+\+\+\s', header=None, engine='python').values
if len(sys.argv) > 3:
    X_temp = numpy.append( X_temp, pandas.read_csv( sys.argv[3], sep='\s\+\+\+\$\+\+\+\s', header=None, engine='python').values, axis=0)
# .isalnum() .isalpha() .isdigit()
vocabulary = OrderedDict( { '' : voca_filter+1 , '<unk>' : voca_filter+1 } )

for i in range( X_temp.shape[0] ):
    for word in range( len( X_temp[i][1].split(' ') ) ):
        temp_word = X_temp[i][1].split(' ')[word]
        vocabulary[temp_word] = vocabulary.get(temp_word, 0) + 1
vocabulary = numpy.array([ k for k,v in vocabulary.items() if v>voca_filter ])
print(vocabulary.shape)
print(vocabulary)
numpy.save('./vocabulary.npy', vocabulary)
vocabulary = OrderedDict( zip( vocabulary, range(vocabulary.shape[0])) )

# training
X_train = numpy.zeros( ( X_temp.shape[0], max_sentence), dtype=int)

no_found_int = vocabulary.get('<unk>')
for i in range( X_temp.shape[0] ):
    count = max_sentence - 1
    for word in range( len( X_temp[i][1].split(' ') ) - 1, -1, -1):
        temp_word = X_temp[i][1].split(' ')[word]
        if temp_word != '' and count>(-1):
            # not found in vocabulary
            if vocabulary.get(temp_word) == None:
                X_train[i][count] = no_found_int
            else:
                X_train[i][count] = vocabulary.get(temp_word)
            count -= 1

y_train = numpy.copy( X_temp[:,0] ).astype(int)
# for debugging
print('X_train(samples, max_sentence):', X_train.shape)
print('--------------------------------')
print('y_train(samples,):', y_train.shape)
print('--------------------------------')

# Start training
model = Sequential()
model.add( Embedding( len(vocabulary), 256, input_length=max_sentence))
model.add( LSTM( 512, return_sequences=True, stateful=False))
model.add( Dropout(0.25))
model.add( LSTM( 512, return_sequences=False, return_state=False, stateful=False))
model.add( Dropout(0.25))
model.add( Dense( 256, activation='relu'))
model.add( Dropout(0.5))
model.add( Dense( 256, activation='relu'))
model.add( Dropout(0.5))
model.add( Dense( 1, activation='sigmoid'))
model.compile( loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, batch_size=batch_, epochs=20, validation_split=0.1, callbacks=[ EarlyStopping( monitor='val_acc', patience=3), ModelCheckpoint( filepath=sys.argv[2], monitor='val_acc', verbose=1, save_best_only=True)])
#model.save(sys.argv[2])
