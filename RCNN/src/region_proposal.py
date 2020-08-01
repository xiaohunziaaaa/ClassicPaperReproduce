import cv2
import os
import numpy as np
import utils
import matplotlib.pyplot as plt

class RP(object):
    def __init__(self, imgdir, xmldir):
        self.img_dir = imgdir
        self.xml_dir = xmldir


    def run(self, train_file=None, rpbox_file=None, label_file=None, gdbox_file=None, is_show=False):
        '''
        :param train_file: the file indicate which image should be treated as training image
        :param rpbox_file: the file store proposed regions' coordinates
        :param label: the file store label of each proposed region
        :param gdbox_file: the ground truth box of each proposed region
        :param is_show: whether to show a picture with proposed regions
        :return: None
        '''
        # import jpg data for train
        f_OK = os.access(train_file, os.F_OK)
        if not f_OK:
            print('Training file does not exist! Please check!')
            return
        f = open(file='../data/VOC2007/ImageSets/Main/train.txt')
        img_indice = f.readlines()
        rpbox_all = []
        label_all = []
        gdbox_all = []
        obj_class = {'background': 0, 'car': 1, 'person': 2, 'horse': 3, 'bicycle': 4,
                     'aeroplane': 5, 'train': 6, 'diningtable': 7, 'dog': 8, 'chair': 9,
                    'cat': 10, 'bird': 11, 'boat': 12, 'pottedplant': 13, 'tvmonitor': 14, 'sofa': 15,
                    'motorbike': 16, 'bottle': 17, 'bus': 18, 'sheep': 19, 'cow': 20}
        counter = 0
        for img_index in img_indice:
            #if counter > 3:
            #    break
            counter += 1
            print('Processing Image No {} \r'.format(counter))
            img_index = img_index.strip('\n')
            # reading jpg
            img_name = img_index + '.jpg'
            img_name = os.path.join(self.img_dir, img_name)
            img_bgr = cv2.imread(filename=img_name)
            b, g, r = cv2.split(img_bgr)
            img_rgb = cv2.merge([r, g, b])

            img_xml = img_index + '.xml'
            img_xml = os.path.join(self.xml_dir, img_xml)
            objs = utils.getgdbox(img_xml)


            # generate region proposal using selective search, coordinated returned
            pregs = utils.getpregions(img_rgb)

            # annotate all regs
            for preg in pregs:
                x, y, w, h = preg
                rpbox = [x, y, x+w, y+h]
                # img_preg = img_rgb[y:(y + h), x:(x + w)]
                # img_preg = cv2.resize(src=img_rgb, dsize=(224, 224))
                label, gbox = utils.annotate(rpbox, objs)
                # not enough memory to save image, so save coordinates
                rpbox_all.append(rpbox)
                gdbox_all.append(gbox)
                il = [img_name, obj_class[label]]
                label_all.append(il)
                if counter == 1 and label != 'background':
                    cv2.rectangle(img_rgb, (rpbox[0], rpbox[1]),
                                  (rpbox[2], rpbox[3]), (0, 255, 0), 1, cv2.LINE_AA)
            if is_show and counter == 1:
                plt.imshow(img_rgb)
                plt.show()

        # reading xml file to extract object and corresponding region
        rpbox_all = np.asarray(rpbox_all)
        label_all = np.asarray(label_all)
        gdbox_all = np.asarray(gdbox_all)
        print(obj_class)
        np.save(arr=rpbox_all, file='../data/region_proposal_data/rpbox_all.npy', allow_pickle=True)
        np.save(arr=label_all, file='../data/region_proposal_data/label_all.npy')
        np.save(arr=gdbox_all, file='../data/region_proposal_data/gdbox_all.npy', allow_pickle=True)

def main():
    train_file='../data/VOC2007/ImageSets/Main/train.txt'
    rpbox_file = '../data/region_proposal_data/rpbox_all.npy'
    label_file = '../data/region_proposal_data/label_all.npy'
    gdbox_file = '../data/region_proposal_data/gdbox_all.npy'
    imgdir = 'D:\Code\Pycharm_Pro\ClassicPaperReproduce\RCNN\data\VOC2007\JPEGImages'
    xml_dir = 'D:\Code\Pycharm_Pro\ClassicPaperReproduce\RCNN\data\VOC2007\Annotations'

    rp = RP(imgdir=imgdir, xmldir=xml_dir)
    rp.run(train_file=train_file, rpbox_file=rpbox_file, label_file=label_file, gdbox_file=gdbox_file, is_show=False)


if __name__ == '__main__':
    main()

