import sys
import csv
import numpy
import pandas
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# readin
# (32561, 106)
x_train = pandas.read_csv(sys.argv[1]).values.astype(numpy.float)
# (32561, 1)
y_train = pandas.read_csv(sys.argv[2]).values.astype(numpy.float)

# 0~1
_max = numpy.amax(x_train, axis=0)
_min = numpy.amin(x_train, axis=0)
numpy.save('train_max.npy', _max)
numpy.save('train_min.npy', _min)
for i in range(x_train.shape[1]):
    if ( _max[i] - _min[i] ) != 0:
        x_train[:,i] = ( x_train[:,i] - _min[i] ) / ( _max[i] - _min[i] )

model = Sequential()
model.add(Dense(1024, input_dim=106, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(x_train, y_train, epochs=24, batch_size=128)
model.save('hw2_best_model.h5')
