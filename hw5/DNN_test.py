import os, sys, csv, numpy, pandas
from collections import OrderedDict

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
# TestDataID, UserID,MovieID,Rating
test = pandas.read_csv( sys.argv[1], sep=',', dtype=int).values

# Start testing
model = load_model('model/dnn1.h5')
rating = model.predict( [test[:,1], test[:,2]], batch_size=128, verbose=1)

for i in range(2,16):
    model = load_model('model/dnn' + str(i) + '.h5')
    rating += model.predict( [test[:,1], test[:,2]], batch_size=128, verbose=1)
rating /= 15.0
rating[rating>=5] = 5
rating[rating<=1] = 1
with open(sys.argv[2], 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['TestDataID', 'Rating'])
    for i in range(rating.shape[0]):
        spamwriter.writerow([i+1, rating[i][0]])