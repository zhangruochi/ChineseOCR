# import chart_flier_model as cfm

import cv2
from split_ocr_images import demo
import os



# 接收参数：原始图片的路径
# 返回参数：切割完成后图片的路径（有序的）和验证码文字的位置（有序的）
def split_img(sess,model,img):

    # img = cv2.imread(filepath)

    boxs = demo.predict_one(sess,model,img)

    # 将boxs进行排序
    n = len(boxs)
    k = n  # k为循环的范围，初始值n
    for i in range(n):
        flag = True
        for j in range(1, k):  # 只遍历到最后交换的位置即可
            if boxs[j - 1][:1] > boxs[j][:1]:
                boxs[j - 1], boxs[j] = boxs[j], boxs[j - 1]
                k = j  # 记录最后交换的位置
                flag = False
        if flag:
            break
    i = 0
    locations = []
    splited_images = []

    
    for box in boxs:
        img_x = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        if img_x.size != 0:
            pass
        else:
            continue
        x = (int(box[0]) + int(box[2])) // 2
        y = (int(box[1]) + int(box[3])) // 2
        locations.append([x, y])
        # cv2.imshow(str(i),img_x)
        i += 1
        # savename = os.path.join(split_save_path,  str(i) + '.png')
        # cv2.imwrite(savename, img_x)
        # split_save_paths.append(savename)
        splited_images.append(img_x)

    return splited_images,locations




# if __name__ == '__main__':
#     root_path = "test_images/b_15_1593312861484.png"

    
#     split_save_paths, locations = split_img(root_path)
#     print(split_save_paths)
#     print(locations)














































