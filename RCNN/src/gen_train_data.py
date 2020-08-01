import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import utils

class DateGenerator(object):
    def __init__(self, arr_file=None, brr_file=None, grr_file=None, odir=None):
        self.arr_file = arr_file
        self.brr_file = brr_file
        self.grr_file = grr_file
        self.odir = odir
        self.obj_class = {'background': 0, 'car': 1, 'person': 2, 'horse': 3, 'bicycle': 4,
                     'aeroplane': 5, 'train': 6, 'diningtable': 7, 'dog': 8, 'chair': 9,
                    'cat': 10, 'bird': 11, 'boat': 12, 'pottedplant': 13, 'tvmonitor': 14, 'sofa': 15,
                    'motorbike': 16, 'bottle': 17, 'bus': 18, 'sheep': 19, 'cow': 20}

    def gen(self, labeltr_file=None, rptr_file=None, gdtf_file=None, ishow=False):
        arr = np.load(file=self.arr_file, allow_pickle=True)
        brr = np.load(file=self.brr_file)
        grr = np.load(file=self.grr_file, allow_pickle=True)
        # img = cv2.imread(filename='test.jpg')
        # b, g, r = cv2.split(img)
        # img_rbg = cv2.merge([r, g, b])
        #img = cv2.imread('')
        counter = 0
        labels_train = []
        rpbox_train = []
        gdbox_trian = []

        for i in range(len(arr)):
            # if brr[i][1] != '1':
                # counter += 1
            print('Processing Image No.{}'.format(counter))
            label = int(brr[i][1])
            img_name = str(counter) + '.jpg'
            img_name = os.path.join(self.odir, img_name)
            # randomly choose training samples, p_backgroud = 0.003, p_object = 0.3
            rd = np.random.rand(1)
            if label == 0:      # background
                if rd < 0.007:
                    counter += 1
                    img = cv2.imread(filename=brr[i][0])
                    img_cut = img[arr[i][1]:arr[i][3], arr[i][0]:arr[i][2]]
                    img_resize = cv2.resize(src=img_cut, dsize=(224, 224))
                    cv2.imwrite(filename=img_name, img=img_resize)
                    labels_train.append(label)
                    rpbox_train.append(arr[i])
                    gdbox_trian.append(grr[i])
            else:               # objects
                if rd < 0.3:
                    counter += 1
                    img = cv2.imread(filename=brr[i][0])
                    img_cut = img[arr[i][1]:arr[i][3], arr[i][0]:arr[i][2]]
                    img_resize = cv2.resize(src=img_cut, dsize=(224, 224))
                    cv2.imwrite(filename=img_name, img=img_resize)
                    labels_train.append(label)
                    rpbox_train.append(arr[i])
                    gdbox_trian.append(grr[i])
                    if ishow == True:
                        plt.imshow(img_resize)
                        plt.show()
                        print('Lable is {}'.format(label))
                        ishow = False

        labels_train = np.asarray(labels_train)
        rpbox_train = np.asarray(rpbox_train)
        gdbox_trian = np.asarray(gdbox_trian)
        np.save(file=labeltr_file, arr=labels_train)
        np.save(file=rptr_file, arr=rpbox_train)
        np.save(file=gdtf_file, arr=gdbox_trian)

        print(counter)
        print(len(labels_train))

def main():
    arr_file = '../data/region_proposal_data/rpbox_all.npy'
    brr_file = '../data/region_proposal_data/label_all.npy'
    grr_file = '../data/region_proposal_data/gdbox_all.npy'
    odir = '../data/CNN_IMG/'

    labeltr_file = '../data/train_data/label_train.npy'
    rptr_file = '../data/train_data/rpbox_train.npy'
    gdtf_file = '../data/train_data/gdbox_train.npy'
    dg = DateGenerator(arr_file=arr_file, brr_file=brr_file, grr_file=grr_file, odir=odir)
    dg.gen(labeltr_file=labeltr_file,
           rptr_file=rptr_file,
           gdtf_file=gdtf_file, ishow=True)

if __name__ == '__main__':
    main()




