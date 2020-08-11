import base64
import requests
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

PyTorch_REST_API_URL = 'http://172.16.20.21:5000/'

def pretrained_predict(base64_data):
 
    # Submit the request.
    r = requests.post(PyTorch_REST_API_URL + "pretrained_predict", data = {'image': base64_data}).json() 
    return r

def tyc_predict(base64_data):
 
    # Submit the request.
    r = requests.post(PyTorch_REST_API_URL + "tyc_predict", data = {'image': base64_data}).json() 
    return r

def gj_predict(base64_data):
    r = requests.post(PyTorch_REST_API_URL + "gj_predict", data={'image': base64_data}).json()
    return r



def test_tyc_valid_images():
    root_path = Path("/home/ruochi/Documents/share/services/app/tyc_valid_img/pictures")
    for direc in root_path.glob("*"):

        print(direc)
        for file in direc.glob("*.png"):
            if file.name.startswith("b"):
                with open(file, "rb") as f:
                    base64_data = base64.b64encode(f.read())
                res = tyc_predict(base64_data)
                if res["success"]:
                    print([ item["label"] for item in res["prediction"]])
                    img = Image.open(str(file))
                    plt.imshow(img)
                else:
                    print("error!")
        

            if file.name.startswith("f"):
                with open(file, "rb") as f:
                    base64_data = base64.b64encode(f.read())
                res = tyc_predict(base64_data)
                if res["success"]:
                    print([item["label"] for item in res["prediction"]])
                    img = Image.open(str(file))
                    plt.imshow(img)
                else:
                    print("error!")
            
            plt.show()
        
            
        
        
if __name__ == '__main__':
    
    # with open("./yolov4_pytorch/img/hanzi_2.png","rb") as f: 
    #      base64_data = base64.b64encode(f.read())
    # print(tyc_predict(base64_data))
    # print(pretrained_predict(base64_data))

    # with open("/home/ruochi/Downloads/2.jpg", "rb") as f:
    #     base64_data = base64.b64encode(f.read())
    # print(gj_predict(base64_data))
    test_tyc_valid_images()








