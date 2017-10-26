import sys
import csv
import numpy
import pandas
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# readin
# (16281, 106)
x_test = pandas.read_csv(sys.argv[1]).values.astype(numpy.float)

# 0~1
_max = numpy.load('train_max.npy')
_min = numpy.load('train_min.npy')
for i in range(x_test.shape[1]):
    if ( _max[i] - _min[i] ) != 0:
        x_test[:,i] = ( x_test[:,i] - _min[i] ) / ( _max[i] - _min[i] )

model = load_model('hw2_best_model.h5')
y = model.predict(x_test, batch_size=128)
print(y)

with open(sys.argv[2], 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['id', 'label'])
    for i in range(y.shape[0]):
        if y[i] > 0.5:
            spamwriter.writerow([ str(i+1) , '1' ])
        else:
            spamwriter.writerow([ str(i+1) , '0' ])
