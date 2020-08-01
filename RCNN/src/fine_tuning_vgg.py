import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras import Model
import matplotlib.pyplot as plt

dir = 'D:\Code\Pycharm_Pro\ClassicPaperReproduce\RCNN\data\CNN_IMG'
counter = 0
y_train = np.load(file='../data/train_data/label_train.npy')
y_train = y_train[0:35000]
y_train = tf.keras.utils.to_categorical(y_train)
print(len(y_train))

x_train = []
for i in range(35000):
    img_name = str(counter) + '.jpg'
    img_name = os.path.join(dir, img_name)
    x_train.append(cv2.imread(img_name))
    counter += 1

x_train = np.asarray(x_train)
np.save(file='../data/train_data/img_train.npy', arr=x_train)
print(x_train.shape)


# x_train = np.load(file='../data/train_data/img_train.npy')
print(len(x_train))

#x_train = x_train[0:1000]
#y_train = y_train[0:1000]

IMG_SHAPE = (224, 224, 3)

feature_extractor = tf.keras.applications.VGG16(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

feature_extractor.trainable = False
feature_extractor.summary()

flatten = tf.keras.layers.Flatten()
fc_1 = tf.keras.layers.Dense(units=2048, activation='relu')
logits = tf.keras.layers.Dense(units=21)
prediction_layer = tf.keras.layers.Activation('softmax')


model = tf.keras.Sequential([feature_extractor, flatten, fc_1, logits, prediction_layer])
base_learning_rate = 0.001

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss=tf.keras.losses.categorical_crossentropy)

model.summary()


# training
model.fit(x_train, y_train, epochs=2, batch_size=16)

model.save(filepath='../models/fine_tuned_VGG.h5')

vgg_dense = Model(inputs=model.input, outputs=model.get_layer('dense').output)
vgg_dense.save(filepath='../models/vgg_dense.h5')

'''
testing
vgg = tf.keras.models.load_model(filepath='../models/fine_tuned_VGG.h5')
img_name = str('18') + '.jpg'
img_name = os.path.join(dir, img_name)
x_test = cv2.imread(img_name)
x_test = np.reshape(x_test, newshape=(-1, 224, 224, 3))
print(vgg.predict(x_test))
'''

