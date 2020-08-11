#-------------------------------------#
#       对单张图片进行预测
#-------------------------------------#
from yolo import YOLO
from PIL import Image

yolo = YOLO()


img = "./img/hanzi_1.png"
try:
    image = Image.open(img)
except:
    print('Open Error! Try again!')
else:
    # r_image = yolo.detect_image(image)
    # r_image.show()
    boxes = yolo.predict_one(image)
    print(boxes)
