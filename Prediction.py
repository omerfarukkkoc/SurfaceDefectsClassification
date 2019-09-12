# -*- coding: utf-8 -*-
"""
Created on Oct 25 14:42:21 2018

@author: omerfarukkoc
"""
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Activation, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras import applications
import cv2

top_model_weights_path = 'SurfaceDefectsWeights-Best.h5'
# top_model_weights_path = 'SurfaceDefectsWeights.h5'

train_data_dir = 'data/SurfaceDefects/train'

# image_path = 'data/SurfaceDefects/TestImages/prick5.jpg'
image_path = 'data/SurfaceDefects/TestImages/scratch5.jpg'
# image_path = 'data/SurfaceDefects/TestImages/texturefailure2.jpg'


# image_path = 'data/SurfaceDefects/TestImages/mix.jpg'   # scratch5.jpg + texturefailure2.jpg

img_width, img_height = 250, 250
batch_size = 32

original_image = cv2.imread(image_path)

image = load_img(image_path, target_size=(img_width, img_height))

image = img_to_array(image)

image = image / 255

image = np.expand_dims(image, axis=0)

# datagen_top = ImageDataGenerator(rescale=1. / 255)
# generator_top = datagen_top.flow_from_directory(
#  train_data_dir,
#  target_size=(img_width, img_height),
#  batch_size=batch_size,
#  class_mode='categorical',
#  shuffle=False)
#
# nb_train_samples = len(generator_top.filenames)
# num_classes = len(generator_top.class_indices)

# class_dictionary = generator_top.class_indices
#
# print(class_dictionary)

class_dictionary = {
  "Prick": 0,
  "Scratch": 1,
  "TextureFailure": 2
}

num_classes = class_dictionary.__len__()

model = applications.VGG16(include_top=False, weights='imagenet')

prediction = model.predict(image)

model = Sequential()

model.add(Conv2D(250, (3, 3),  padding='same', input_shape=prediction.shape[1:]))
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

model.load_weights(top_model_weights_path)

class_predicted = model.predict_classes(prediction)

inID = class_predicted[0]

inv_map = {v: k for k, v in class_dictionary.items()}

label = inv_map[inID]

probabilities = model.predict_proba(prediction)

accuracy = round((probabilities.max()) * 100, 2)

print("\n\n")
print("Probabilities: {}".format(probabilities))
print("Image ID: {}, Label: {}".format(inID, label))
print("Accuracy: %{}".format(accuracy))


cv2.putText(original_image, "Predicted: {}".format(label), (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
cv2.putText(original_image, "Accuracy: %{}".format(accuracy), (10, 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)

cv2.imshow("Predicted: {}".format(label) + " / Accuracy: %{}".format(accuracy), original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()