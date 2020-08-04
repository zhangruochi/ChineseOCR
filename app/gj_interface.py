import cv2
import numpy as np
import os
import time

# 返回整数时间戳


def timestamp():
    return int(time.time() * 10000)

# 水平方向投影


def hProject(binary):
    h, w = binary.shape
    # 水平投影
    hprojection = np.zeros(binary.shape, dtype=np.uint8)
    # 创建h长度都为0的数组
    h_h = [0] * h
    for j in range(h):
        for i in range(w):
            if binary[j, i] == 0:
                h_h[j] += 1
    # 画出投影图
    for j in range(h):
        for i in range(h_h[j]):
            hprojection[j, i] = 255

    return h_h

# 垂直方向投影


def vProject(binary):
    h, w = binary.shape
    # 垂直投影
    vprojection = np.zeros(binary.shape, dtype=np.uint8)
    #创建 w 长度都为0的数组
    w_w = [0]*w
    for i in range(w):
        for j in range(h):
            if binary[j, i] == 0:
                w_w[i] += 1

    for i in range(w):
        for j in range(w_w[i]):
            vprojection[j, i] = 255

    return w_w


def fix(x, y, x_plus_w, y_plus_h, x_max, y_max):
    x = 0 if x < 0 else x
    y = 0 if y < 0 else y
    x_plus_w = x_max if x_plus_w > x_max else x_plus_w
    y_plus_h = y_max if y_plus_h > y_max else y_plus_h
    return x, y, x_plus_w, y_plus_h


# 切割图片区域
def seg_one_img(img, positions):
    add = 2  # 扩充2个像素
    h, w, d = img.shape
    cut_list = []
    for position in positions:
        x, y, x_plus_w, y_plus_h = fix(
            position[0]-add, position[1]-add, position[2]+add, position[3]+add, w, h)
        try:
            hanzi_img = img[y:y_plus_h, x:x_plus_w]
            normal_img = cv2.resize(hanzi_img, (28, 28),
                                    interpolation=cv2.INTER_CUBIC)  # 将截取的图片规范化为28*28*3
            cut_list.append(normal_img)
        except Exception as e:
            print(e)
    return cut_list

# 接收参数：image对象
# 返回参数：list[image]


def cut_single_txt(cvImg):
    img_gray = cv2.cvtColor(cvImg, cv2.COLOR_BGR2GRAY)  # 转换了灰度化
    # 将灰度图像二值化，设定阈值是100
    ret, thresh = cv2.threshold(img_gray, 240, 255, cv2.THRESH_TOZERO)

    th = thresh
    h, w = th.shape
    w_w = vProject(th)
    start = 0
    w_start, w_end = [], []
    positions = []
    # 根据水平投影获取垂直分割
    for i in range(len(w_w)):
        if w_w[i] > 0 and start == 0:
            w_start.append(i)
            start = 1
        if w_w[i] == 0 and start == 1:
            w_end.append(i)
            start = 0
    for i in range(len(w_start)):
        if i == 0:
            pass
        cropImg = th[0:h, w_start[i]:w_end[i]]
        h_h = hProject(cropImg)

        hstart, hend, h_start, h_end = 0, 0, 0, 0
        for j in range(len(h_h)):
            if h_h[j] > 0 and hstart == 0:
                h_start = j
                hstart = 1
                hend = 0
            if h_h[j] == 0 and hstart == 1:
                h_end = j
                hstart = 0
                hend = 1
            # 当确认了起点和终点以后保存坐标
            if hend == 1:
                positions.append([w_start[i], h_start, w_end[i], h_end])
                hend = 0
    cut_list = seg_one_img(cvImg, positions)
    return cut_list


if __name__ == '__main__':

    fileExt = '.png'
    filePath = r'E:\zhouqiong\ocr\dealqgxxgsxt'
    fileName = 'sample_txt'
    imageFileName = os.path.join(filePath, fileName+fileExt)
    cvImg = cv2.imread(imageFileName, 1)

    cut_list = cut_single_txt(cvImg)
    print(cut_list)