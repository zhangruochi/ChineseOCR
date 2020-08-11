# import chart_flier_model as cfm

import cv2
import os
from yolov4_pytorch.yolo import YOLO
from PIL import Image
import numpy as np


# 接收参数：原始图片的路径
# 返回参数：切割完成后图片的路径（有序的）和验证码文字的位置（有序的）
def split_img(model,img):

    boxs = model.predict_one(img)
    img = np.array(img)
    # print(img.shape)  # (100, 320, 3)
    # top, left, bottom, right = boxes[i]
    boxs = sorted(boxs, key=lambda x: x[1])

    i = 0
    locations = []
    splited_images = []

    for box in boxs:
        img_x = img[int(box[0]):int(box[2]), int(box[1]):int(box[3])]
        if img_x.size != 0:
            pass
        else:
            continue
        x = (int(box[0]) + int(box[2])) // 2
        y = (int(box[1]) + int(box[3])) // 2
        locations.append([x, y])
        i += 1
        splited_images.append(img_x)

    return splited_images,locations


if __name__ == '__main__':
    root_path = "/home/ruochi/Documents/share/services/app/tyc_valid_img/pictures/0_1596414545845/f_0_1596414545845.png"
    image = Image.open(root_path)
    yolo = YOLO()
    splited_images, locations = split_img(yolo, image)
    print(locations)
    # print(splited_images)
