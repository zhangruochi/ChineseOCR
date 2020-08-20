import sys

sys.path.append("/home/ruochi/Documents/share/services/app")

import cv2
import os
import numpy as np
from collections import Counter
from pathlib import Path


from yolov4_pytorch.yolo import YOLO
from PIL import Image





yolo = YOLO()

def predict(direc, file):

    (direc/"0").mkdir(exist_ok=True)
    (direc/"1").mkdir(exist_ok=True)

    img = Image.open(str(file))
    boxs = yolo.predict_one(img)

    if boxs == []:
        return 

    boxs = sorted(boxs, key = lambda x: x[1])

    img = np.array(img)


    if file.name.startswith("b"):
        folder = "0"
    
    if file.name.startswith("f"):
        folder = "1"

    i = 0
    imgs = []
    locations = []
    for box in boxs:
        img_x = img[int(box[0]):int(box[2]), int(box[1]):int(box[3])]

        if img_x.size != 0:
            pass
        else:
            continue

        # print(int(box[1]),int(box[3]),int(box[0]),int(box[2]))
        x = (int(box[0])+int(box[2]))//2
        y = (int(box[1]) + int(box[3])) // 2
        locations.append([x, y])
        # cv2.imshow(str(i),img_x)
        i += 1

        savename = os.path.join(str(direc / folder), str(folder)+'_'+str(i)+'.png')
        cv2.imwrite(savename, img_x)

        img_x = cv2.resize(img_x, (28, 28))
        imgs.append(img_x/255)

if __name__ == '__main__':

    root_path = Path(r'/home/ruochi/Documents/share/wd_test_patent_pic/services/data/pictures')
    
    for direc in root_path.glob("*"):
        if (direc / "0").exists() and (direc / "1").exists():
            continue
        print(direc)
        for file in direc.glob("*.png"):
            predict(direc, file)
