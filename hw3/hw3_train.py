import sys
import csv
import numpy
import pandas
# numpy.random.seed(0)

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam, Adadelta
from keras.callbacks import EarlyStopping, ModelCheckpoint

height = 48
width  = 48
channels = 1
category_count = 7
# readin
# train.csv
temp = pandas.read_csv( sys.argv[1], sep=',').values
y_train = numpy.copy( (temp[:,0]).astype(numpy.int16) )
y_train = keras.utils.to_categorical( y_train, num_classes=category_count)

X_train = numpy.zeros( ( temp.shape[0], height * width), dtype=float)
for i in range(X_train.shape[0]):
    X_train[i] = numpy.asarray( temp[i, 1].split(' '), dtype=float )
X_train = X_train.reshape( -1, height, width, channels)

print('X_train(samples, height, width, channels):', X_train.shape)
print('y_train(samples, num_classes):', y_train.shape)
# Start training

model = Sequential()

model.add(Conv2D(  64, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu', input_shape=( height, width, channels)))
model.add(MaxPooling2D((5, 5), strides=(2, 2), padding='same'))
model.add( Dropout(0.25))

model.add(Conv2D(  64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(Conv2D(  64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(AveragePooling2D((3, 3), strides=(2, 2), padding='valid'))
model.add( Dropout(0.25))

model.add(Conv2D( 128, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
model.add(Conv2D( 128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(AveragePooling2D((3, 3), strides=(2, 2), padding='same'))
model.add( Dropout(0.25))

model.add( Flatten())
model.add( Dense(1024, activation='relu'))
model.add( Dropout(0.25))
model.add( Dense(1024, activation='relu'))
model.add( Dropout(0.25))
model.add( Dense(1024, activation='relu'))
model.add( Dropout(0.25))
model.add( Dense(category_count, activation='softmax'))
'''
model = Sequential()
model.add(Conv2D(  64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu', input_shape=( height, width, channels)))
model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
model.add(Dropout(0.25))

model.add(Conv2D( 128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
model.add(Dropout(0.25))

model.add(Conv2D( 256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
model.add(Dropout(0.5))

model.add(Conv2D( 512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
model.add(Dropout(0.75))

model.add( Flatten())
model.add( Dense( 256, activation='relu'))
model.add( Dropout(0.5))
model.add( Dense( 256, activation='relu'))
model.add( Dropout(0.5))
model.add( Dense( 256, activation='relu'))
model.add( Dropout(0.5))
model.add( Dense(category_count, activation='softmax'))

model.summary()
'''
# model = load_model(sys.argv[3])
#sgd = SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)
#adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
datagen = ImageDataGenerator(featurewise_center=False,
                             samplewise_center=False,
                             featurewise_std_normalization=False,
                             samplewise_std_normalization=False,
                             zca_whitening=False,
                             zca_epsilon=1e-6,
                             rotation_range=25,
                             width_shift_range=0.25,
                             height_shift_range=0.25,
                             shear_range=0.,
                             zoom_range=0.,
                             channel_shift_range=0.,
                             fill_mode='nearest',
                             cval=0.,
                             horizontal_flip=True,
                             vertical_flip=False,
                             rescale=1./255,
                             preprocessing_function=None)
test_datagen = ImageDataGenerator(rescale=1./255)
datagen.fit(X_train)
#Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
for i in range(10):
    model.fit_generator( datagen.flow(X_train[:22400], y_train[:22400], batch_size=64),
                         steps_per_epoch=(X_train[:22400].shape[0] / 64), epochs=100+i*100, validation_data=test_datagen.flow(X_train[22400:], y_train[22400:], batch_size=64), validation_steps=(X_train[22400:].shape[0] / 64), initial_epoch=i*100)
    model.save('./TA'+str(100+i*100)+'.h5')
#early_stopping = EarlyStopping(monitor='val_acc', patience=5)
#checkpointer = ModelCheckpoint(filepath='./temp.h5', monitor='val_acc', verbose=1, save_best_only=True)
#model.fit(X_train, y_train, batch_size=64, epochs=200, validation_split=0.1, callbacks=[early_stopping])
#model.save(sys.argv[2])
