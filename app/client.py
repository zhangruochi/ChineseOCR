import base64
import requests

PyTorch_REST_API_URL = 'http://127.0.0.1:5000/predict'


def predict(base64_data):
 
    # Submit the request.
    r = requests.post(PyTorch_REST_API_URL, data = {'image': base64_data}).json() 
    return r


if __name__ == '__main__':
    
    with open("test_images/f_15_1593312861484.png","rb") as f: 
        base64_data = base64.b64encode(f.read())

    print(predict(base64_data))






