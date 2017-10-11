import sys
import numpy
import pandas

# python3 this.py train.csv model_out

lr = 1e+4
# iteration = 1000000

# readin train.csv
train_data = pandas.read_csv(sys.argv[1], encoding = 'Big5').values

# remove_label reshape
train_data = numpy.delete(train_data, numpy.s_[0:3],1)
data = numpy.append(train_data[0:18,0:24], train_data[18:18*2,0:24], axis=1)
for i in range(2,240):
    data = numpy.append(data, train_data[18*i:18*(i+1),0:24], axis=1)

# remove_NoRain format
data[data == 'NR'] = 0
data = data.astype(numpy.float)

# parse
x = []
y = []
# 每 12 個月
for month in range(12):
    # 一個月取連續10小時的data可以有471筆
    for hour in range(471):
        x.append([])
        # 18種污染物
        for t in range(18):
            # 連續9小時
            for s in range(9):
                x[471*month+hour].append(data[t][480*month+hour+s] )
        y.append(data[9][480*month+hour+9])

x = numpy.array(x)
y = numpy.array(y)

# add square term
x = numpy.concatenate((x,x**2), axis=1)

# add bias
x = numpy.concatenate((numpy.ones((x.shape[0],1)),x), axis=1)

# init
model_w = numpy.zeros(len(x[0]))

# close_form
# model_w = numpy.matmul(numpy.matmul(numpy.linalg.inv(numpy.matmul(x.transpose(),x)),x.transpose()),y)

# Start training
x_t = x.transpose()
s_gra = numpy.zeros(len(x[0]))

err = 10000000000000
i = 0
# for i in range(iteration):
while True :
    loss = y - numpy.dot(x, model_w)
    gra = numpy.dot(x_t,loss) * (-2)
    s_gra += gra**2
    model_w = model_w - lr * gra / numpy.sqrt(s_gra)
    error = numpy.sum(loss**2)
    print ('iteration: %d | Cost: %f  ' % ( i, numpy.sqrt( error / len(x) ) ))
    if err < error and i > 100:
        break
    err = error
    i = i + 1

# output_model
numpy.save(sys.argv[2]+'.npy', model_w)
