import cv2
import numpy as np
import os
import time
from pathlib import Path


# 返回整数时间戳
def timestamp():
    return int(time.time() * 10000)

# 水平方向投影
def hProject(binary):
    h , w = binary.shape
    # 水平投影
    hprojection = np.zeros(binary.shape , dtype=np.uint8)
    # 创建h长度都为0的数组
    h_h = [0] * h
    for j in range(h):
        for i in range(w):
            if binary[j,i] == 0:
                h_h[j] += 1
    # 画出投影图
    for j in range(h):
        for i in range(h_h[j]):
            hprojection[j,i] = 255

    print('h_h is {}'.format(h_h))

    return h_h

# 垂直方向投影
def vProject(binary):
    h , w = binary.shape
    # 垂直投影
    vprojection = np.zeros(binary.shape,dtype=np.uint8)
    #创建 w 长度都为0的数组
    w_w = [0]*w
    for i in range(w):
        for j in range(h):
            if binary[j,i] == 0:
                w_w[i] += 1

    for i in range(w):
        for j in range(w_w[i]):
            vprojection[j,i] = 255

    print('w_w is {}'.format(w_w))

    return w_w


def fix(x, y, x_plus_w, y_plus_h,x_max,y_max):
    x = 0 if x < 0 else x
    y = 0 if y < 0 else y
    x_plus_w = x_max if x_plus_w > x_max else x_plus_w
    y_plus_h = y_max if y_plus_h > y_max else y_plus_h
    return x, y, x_plus_w, y_plus_h


# 切割图片区域
def seg_one_img(aimPath,imageFileName, positions):
    img = cv2.imread(str(imageFileName))
    add = 2 #扩充2个像素

    h, w,d = img.shape

    cut_list = []
    for name, position in zip(imageFileName.name.split(".")[0],positions):
        per_dict = []

        x, y, x_plus_w, y_plus_h = fix(position[0]-add, position[1]-add, position[2]+add,position[3]+add,w,h)
        # print('x, y, x_plus_w, y_plus_h is {},{},{},{}'.format(x, y, x_plus_w, y_plus_h))
        try:
            hanzi_img = img[y:y_plus_h, x:x_plus_w]

            normal_img = cv2.resize(hanzi_img, (28, 28),
                                    interpolation=cv2.INTER_CUBIC)  # 将截取的图片规范化为65*65*3
            path = os.path.join(aimPath,'{}.png'.format(name))
            cv2.imwrite(path, normal_img)
            # print('hanzi_img is {}'.format(hanzi_img))

            cut_list.append(path)
        except:
            print('#' * 20)
            print('存在不规则的图片')
    return cut_list



if __name__ == '__main__':


    root = Path("original_captchas")
    aimPath = "cuts"

    for file in root.glob("*.png"):

        imageFileName = file
        cvImg = cv2.imread(str(imageFileName),1)

        img_gray = cv2.cvtColor(cvImg, cv2.COLOR_BGR2GRAY)  # 转换了灰度化

        #将灰度图像二值化，设定阈值是100
        ret,thresh = cv2.threshold(img_gray, 240, 255, cv2.THRESH_TOZERO)

        th = thresh
        h,w = th.shape
        w_w = vProject(th)

        start = 0
        w_start,w_end = [],[]
        positions = []
        #根据水平投影获取垂直分割
        for i in range(len(w_w)):
            if w_w[i] > 0 and start == 0:
                w_start.append(i)
                start = 1
            if w_w[i] == 0 and start == 1:
                w_end.append(i)
                start = 0

        print('h_start is {}'.format(w_start))
        print('h_end is {}'.format(w_end))

        for i in range(len(w_start)):

            if i == 0:
                pass
            cropImg = th[0:h,w_start[i]:w_end[i]]
            h_h = hProject(cropImg)
            # print('length h_h is {}'.format(len(h_h)))

            hstart , hend , h_start ,h_end = 0, 0, 0, 0
            for j in range(len(h_h)):
                if h_h[j] > 0 and hstart == 0:
                    h_start = j
                    hstart= 1
                    hend = 0
                if h_h[j] == 0 and hstart == 1:
                    h_end = j
                    hstart = 0
                    hend = 1
                #当确认了起点和终点以后保存坐标
                if hend == 1:
                    print('#'*30)
                    positions.append([w_start[i],h_start,w_end[i],h_end])
                    hend = 0

        print('position length is {}'.format(positions))

        # #确定分割位置
        # for p in positions:
        #     cv2.rectangle(cvImg,(p[0],p[1]),(p[2] ,p[3]),(0,255,255),2)
        # # cv2.imshow('over',cvImg)
        # cv2.waitKey()
        seg_one_img(aimPath,imageFileName,positions)




