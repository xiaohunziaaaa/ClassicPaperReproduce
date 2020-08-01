import tensorflow as tf
from tensorflow.keras import Model
import numpy as np

#using dense output to regression
# 输入为proposal的dense层输出，标签为真实bbox和region proposal的一个差值
# construct t^i

rpbox_train = np.load('../data/train_data/rpbox_train.npy')
gdbox_train = np.load('../data/train_data/gdbox_train.npy')
label_train = np.load('../data/train_data/label_train.npy')
img_train = np.load('../data/train_data/img_train.npy')
rpbox_bb_train = []
gdbox_bb_train = []
img_bb_train = []

for i in range(len(img_train)):
    if label_train[i] != 0:
        rpbox_bb_train.append(rpbox_train[i])
        gdbox_bb_train.append(gdbox_train[i])
        img_bb_train.append(img_train[i])

print(len(img_bb_train))
rpbox_bb_train = np.asarray(rpbox_bb_train)
gdbox_bb_train = np.asarray(gdbox_bb_train)
img_bb_train = np.asarray(img_bb_train)


# print(rpbox_bb_train.shape)
# transformation matrix. From (xmin, ymin, xmax, ymax) to (xcenter, ycenter, width, height)
tm = [[0.5, 0, -1, 0],
      [0, 0.5, 0, -1],
      [0.5, 0, 1, 0],
      [0, 0.5, 0, 1]]

# construt target value t
P = rpbox_bb_train @ tm
G = gdbox_bb_train @ tm

t_x = (G[:, 0] - P[:, 0])/P[:, 2]
t_y = (G[:, 1] - P[:, 1])/P[:, 3]
t_w = np.log(G[:, 2]/P[:, 2])
t_h = np.log(G[:, 3]/P[:, 3])

t_x = np.reshape(a=t_x, newshape=[-1, 1])
t_y = np.reshape(a=t_y, newshape=[-1, 1])
t_w = np.reshape(a=t_w, newshape=[-1, 1])
t_h = np.reshape(a=t_h, newshape=[-1, 1])

print(t_x[0])
target_train = np.concatenate((t_x, t_y, t_w, t_h), axis=1)
print(target_train.shape)

'''
ft_vgg16 = tf.keras.models.load_model(filepath='../models/fine_tuned_VGG.h5')
dense = Model(inputs=ft_vgg16.input, outputs=ft_vgg16.get_layer('dense').output)

dense.summary()

features = dense.predict(x=img_bb_train, batch_size=16)
print('predicted')

np.save(arr=features, file='../data/svm_bboxregression_data/regression_features.npy', allow_pickle=True)
'''
regression_feature = np.load(file='../data/svm_bboxregression_data/regression_features.npy', allow_pickle=True)
regression_feature /= 255.0
regression = tf.keras.models.Sequential()
print(regression_feature.shape)
regression.add(tf.keras.layers.Dense(4, input_shape=(2048,)))
regression.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                   loss=tf.keras.losses.mse)

regression.summary()
regression.fit(x=regression_feature, y=target_train, epochs=20, batch_size=32)

regression.save(filepath='../models/bbox_regression.h5')

print(regression_feature[0])
regression = tf.keras.models.load_model(filepath='../models/bbox_regression.h5')
print(regression.predict(regression_feature[0:1]))