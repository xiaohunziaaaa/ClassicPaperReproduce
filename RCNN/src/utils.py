import xml.etree.ElementTree as ET
import cv2
def getgdbox(file_name):
    xml = ET.parse(file_name)
    objs = []
    for obj in xml.iter('object'):
        if obj[4].tag == 'bndbox':
            name = obj[0].text
            xmin = int(obj[4][0].text)
            ymin = int(obj[4][1].text)
            xmax = int(obj[4][2].text)
            ymax = int(obj[4][3].text)
            obj_tuple = (name, xmin, ymin, xmax, ymax)
            objs.append(obj_tuple)

    return objs

def getpregions(imgs):
    cv2.setUseOptimized(True)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(imgs)
    ss.switchToSelectiveSearchFast()

    # Run selective search on the image to fetch regions.
    # Regions are returned as coordinates
    region_coordinates = ss.process()

    return region_coordinates

def annotate(preg, objs):
    # 取iou最大的ground truth box， 若iou大于阈值，返回相应标签，否则返回背景标签
    iou_max = 0
    for obj in objs:
        gbox = [obj[1], obj[2], obj[3], obj[4]]
        iou = caliou(preg, gbox)
        if iou > iou_max:
            iou_max = iou
            label = obj[0]
            box = gbox
    if iou_max > 0.6:
        return label, box
    else:
        return 'background', [0, 0, 0, 0]




def caliou(rect1, rect2):
    xmin1, ymin1, xmax1, ymax1 = rect1
    xmin2, ymin2, xmax2, ymax2 = rect2
    # 计算每个矩形的面积
    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)  # C的面积
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)  # G的面积

    # 计算相交矩形
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    area = w * h  # C∩G的面积
    iou = area / (s1 + s2 - area)
    return iou

def takesore(elem):
    return elem[1]

def nms(H):
    M = []
    H.sort(key=takesore, reverse=True)
    while len(H) > 0:
        M.append(H[0])
        predict_box = H[0][0]
        H.pop(0)
        for i in range(len(H)-1, 0, -1):
            if caliou(H[i][0], predict_box) > 0.3:
                H.pop(i)
    return M

def main():
    objs = getgdbox(file_name='../data/VOC2007/Annotations/000001.xml')
    for i in range(len(objs)):
        print(objs[i])

if __name__ == '__main__':
    main()