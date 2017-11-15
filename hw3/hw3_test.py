import sys
import csv
import numpy
import pandas
import keras
from keras.models import load_model

height = 48
width  = 48
channels = 1
category_count = 7
# readin
# test.csv
temp = pandas.read_csv( sys.argv[1], sep=',').values
X_test = numpy.zeros( ( temp.shape[0], height * width), dtype=float)
for i in range(X_test.shape[0]):
    X_test[i] = numpy.asarray( temp[i, 1].split(' '), dtype=float )
X_test = X_test.reshape(-1, height, width, channels) / 255.0
print('X_test(samples, height, width, channels):', X_test.shape)
# load model and predict
print( 'Load model:' , sys.argv[3] )
model = load_model( sys.argv[3] )
label = model.predict( X_test )
for i in range(4, len(sys.argv)):
    print( 'Load model:' , sys.argv[i] )
    model = load_model( sys.argv[i] )
    label += model.predict( X_test )
label = numpy.argmax( label , axis=1 )
# output
with open(sys.argv[2], 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['id', 'label'])
    for i in range(label.shape[0]):
        spamwriter.writerow([i, label[i]])