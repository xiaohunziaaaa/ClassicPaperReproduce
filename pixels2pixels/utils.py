import numpy as np
import os
import matplotlib as plt
from matplotlib.pyplot import imread

def load_images(dir):
    if os.path.exists('./cityscapes.npy'):
        raw_image = np.load('./cityscapes.npy')
        print(raw_image.shape)
        print('Loading from npy, dataset shape is ', raw_image.shape)
        return raw_image
    raw_train_images = []
    dir_train = os.path.join(dir, 'train')
    dir_test = os.path.join(dir, 'val')
    for root, dirs, pics in os.walk(dir_train):
        for pic in pics:
            pic_path = os.path.join(root, pic)
            raw_train_images.append(imread(pic_path))

    raw_train_images = np.asarray(raw_train_images)
    # np.save(file='./cityscapes.npy',arr=raw_train_images)
    print('Loading from raw data, traing dataset shape is ', raw_train_images.shape)

    raw_test_images = []
    for root, dirs, pics in os.walk(dir_test):
        for pic in pics:
            pic_path = os.path.join(root, pic)
            raw_test_images.append(imread(pic_path))

    raw_test_images = np.asarray(raw_test_images)
    # np.save(file='./cityscapes.npy',arr=raw_test_images)
    print('Loading from raw data, testing dataset shape is ', raw_test_images.shape)
    return raw_train_images, raw_test_images

def imdivandpre(raw_image):
    x, y = np.split(ary=raw_image, indices_or_sections=2, axis=2)
    return x, y