import os
from PIL import Image
import random


def make_text(list_img):
    random.shuffle(list_img)
    print(list_img)
    for i in range(len(list_img)):
        strs = list_img[i].split('.')[0]
        if i < 100:
            with open('data/VOCDevkit2007/VOC2007/ImageSets/Main/cchart_train.txt', 'a+') as f:
                f.writelines(strs + ' 1' + '\n')
            with open('data/VOCDevkit2007/VOC2007/ImageSets/Main/train.txt', 'a+') as f:
                f.writelines(strs + '\n')
        else:
            with open('data/VOCDevkit2007/VOC2007/ImageSets/Main/cchart_trainval.txt', 'a+') as f:
                f.writelines(strs + ' 1' + '\n')
            with open('data/VOCDevkit2007/VOC2007/ImageSets/Main/trainval.txt', 'a+') as f:
                f.writelines(strs + '\n')

    random.shuffle(list_img)
    print(list_img)
    for i in range(len(list_img)):
        strs = list_img[i].split('.')[0]
        if i < 100:
            with open('data/VOCDevkit2007/VOC2007/ImageSets/Main/cchart_val.txt', 'a+') as f:
                f.writelines(strs + ' 1' + '\n')
            with open('data/VOCDevkit2007/VOC2007/ImageSets/Main/val.txt', 'a+') as f:
                f.writelines(strs + '\n')
        else:
            with open('data/VOCDevkit2007/VOC2007/ImageSets/Main/cchart_test.txt', 'a+') as f:
                f.writelines(strs + ' 1' + '\n')
            with open('data/VOCDevkit2007/VOC2007/ImageSets/Main/test.txt', 'a+') as f:
                f.writelines(strs + '\n')
