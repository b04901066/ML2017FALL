import sys, csv, numpy, pandas
import keras
from keras.models import load_model
model = load_model('model/mf8d.h5')
model.summary()
test = pandas.read_csv( sys.argv[1], sep=',', dtype=int).values
rating = model.predict( [test[:,1], test[:,2]], batch_size=128, verbose=1)
rating[rating>=5] = 5
rating[rating<=1] = 1
with open(sys.argv[2], 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['TestDataID', 'Rating'])
    for i in range(rating.shape[0]):
        spamwriter.writerow([i+1, rating[i][0]])