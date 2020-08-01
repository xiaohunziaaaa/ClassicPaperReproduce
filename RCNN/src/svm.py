from sklearn import svm
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from joblib import dump, load
# in paper Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation
# the author using category-specific SVM to classify but not softmax output of CNN

label_train = np.load('../data/train_data/label_train.npy')
label_train = label_train[0:35000]
'''
img_train = np.load('../data/train_data/img_train.npy')
ft_vgg16 = tf.keras.models.load_model(filepath='../models/fine_tuned_VGG.h5')
dense = Model(inputs=ft_vgg16.input, outputs=ft_vgg16.get_layer('dense').output)

dense.summary()

features = dense.predict(x=img_train, batch_size=16)
print(features.shape)
np.save(arr=features, file='../data/svm_bboxregression_data/svm_feature_train.npy')
'''
features = np.load(file='../data/svm_bboxregression_data/svm_feature_train.npy')
SVM = svm.LinearSVC(multi_class='ovr')
SVM.fit(X=features, y=label_train)
dump(SVM, '../models/svm.joblib')




