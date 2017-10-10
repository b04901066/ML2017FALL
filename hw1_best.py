import sys
import csv
import numpy
import pandas

# python3.6 this.py test.csv res.csv model


# readin test.csv
data = numpy.array(pandas.read_csv(sys.argv[1], header = None))
# remove_label
data = numpy.delete(data,numpy.s_[0:2],1)
# remove_NoRain format
data[data == 'NR'] = 0
data = data.astype(numpy.float)

test_x = []

for r in range(data.shape[0]):
    if r%18 == 0:
        test_x.append([])
    for i in range(9):
       test_x[r//18].append( data[r, i] )

test_x = numpy.array(test_x)

# add square term
test_x = numpy.concatenate((test_x,test_x**2), axis=1)

# add bias
test_x = numpy.concatenate((numpy.ones((test_x.shape[0],1)),test_x), axis=1)

# readin model
model_w = numpy.load(sys.argv[3]+'.npy')

# Start testing
with open(sys.argv[2], 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['id', 'value'])
    for i in range(test_x.shape[0]):
        result = numpy.dot(model_w, test_x[i])
        if result < 0:
            result = 0
        spamwriter.writerow([ 'id_'+str(i) , result ])
