import os, sys, csv, numpy, pandas
from collections import OrderedDict

import tensorflow as tf
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Embedding, Flatten, Reshape, Concatenate, Dot, Input, Add
from keras.preprocessing import sequence
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

# fix random seed for reproducibility
# numpy.random.seed(7)

# readin
# <test.csv path> <prediction file path> <movies.csv path> <users.csv path>
# TrainDataID, UserID,MovieID,Rating
train = pandas.read_csv( sys.argv[1], sep=',', dtype=int).values[:,1:4]
numpy.random.shuffle( train )

# Start training
in_U = Input( shape=(1,) )
in_M = Input( shape=(1,) )

# , embeddings_initializer='zeros'
w_U = Flatten()( Embedding( max(train[:,0]), 10)( in_U ) )
w_M = Flatten()( Embedding( max(train[:,1]), 10)( in_M ) )
b_U = Flatten()( Embedding( max(train[:,0]),  1, embeddings_initializer='zeros')( in_U ) )
b_M = Flatten()( Embedding( max(train[:,1]),  1, embeddings_initializer='zeros')( in_M ) )

added = Dot(axes=1)( [w_U, w_M] )
out = Add()( [added, b_U, b_M] )
model = keras.models.Model(inputs=[in_U, in_M], outputs=out)
model.compile( loss='MSE', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit([train[:,0], train[:,1]], train[:,2], batch_size=64, epochs=20, validation_split=0.1, callbacks=[ EarlyStopping( monitor='val_loss', patience=3), ModelCheckpoint( filepath=sys.argv[2], monitor='val_loss', verbose=1, save_best_only=True)])