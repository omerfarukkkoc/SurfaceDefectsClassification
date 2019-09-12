# -*- coding: utf-8 -*-
"""
Created on Oct 25 14:42:21 2018

@author: omerfarukkoc
"""
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import math

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD


img_width, img_height = 250, 250

top_model_weights_path = 'SurfaceDefectsWeights.h5'
train_data_dir = 'data/SurfaceDefects/train'
validation_data_dir = 'data/SurfaceDefects/validation'

epochs = 75
batch_size = 32

model = applications.VGG16(include_top=False, weights='imagenet')


datagen = ImageDataGenerator(rescale=1. / 255)

generator = datagen.flow_from_directory(
 train_data_dir,
 target_size=(img_width, img_height),
 batch_size=batch_size,
 class_mode=None,
 shuffle=False)

nb_train_samples = len(generator.filenames)
num_classes = len(generator.class_indices)

print("Num Classes: ", num_classes)


predict_size_train = int(math.ceil(nb_train_samples / batch_size))

features_train = model.predict_generator(generator, predict_size_train)

np.save('train_data.npy', features_train)


generator = datagen.flow_from_directory(
 validation_data_dir,
 target_size=(img_width, img_height),
 batch_size=batch_size,
 class_mode=None,
 shuffle=False)

nb_validation_samples = len(generator.filenames)

predict_size_validation = int(math.ceil(nb_validation_samples / batch_size))

features_validation = model.predict_generator(
 generator, predict_size_validation)

np.save('validation_data.npy', features_validation)

#######################################################


datagen_top = ImageDataGenerator(rescale=1. / 255)
generator_top = datagen_top.flow_from_directory(
 train_data_dir,
 target_size=(img_width, img_height),
 batch_size=batch_size,
 class_mode='categorical',
 shuffle=False)

nb_train_samples = len(generator_top.filenames)
num_classes = len(generator_top.class_indices)

train_data = np.load('train_data.npy')

train_labels = generator_top.classes

train_labels = to_categorical(train_labels, num_classes=num_classes)

# print(train_labels)


generator_top = datagen_top.flow_from_directory(
 validation_data_dir,
 target_size=(img_width, img_height),
 batch_size=batch_size,
 class_mode=None,
 shuffle=False)

nb_validation_samples = len(generator_top.filenames)

validation_data = np.load('validation_data.npy')

validation_labels = generator_top.classes
validation_labels = to_categorical(validation_labels, num_classes=num_classes)

model = Sequential()

model.add(Conv2D(250, (3, 3),  padding='same', input_shape=train_data.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(250, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='sigmoid'))


lr = 0.00001
sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

# model.compile(optimizer='rmsprop',
#               loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_data, train_labels,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(validation_data, validation_labels))

model.save_weights(top_model_weights_path)

model.save('SurfaceModel.model')
(eval_loss, eval_accuracy) = model.evaluate(
 validation_data, validation_labels, batch_size=batch_size, verbose=1)

print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
print("[INFO] Loss: {}".format(eval_loss))

plt.figure(1)
plt.subplot(211)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()