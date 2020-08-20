import base64
import requests
from pathlib import Path

PyTorch_REST_API_URL = 'http://172.16.20.21:5002/'


def tyc_predict(base64_data):

    # Submit the request.
    r = requests.post(PyTorch_REST_API_URL + "tyc_single_predict",
                      data={'image': base64_data}).json()
    return r


def load_model(model_name):
    """
    model_name: 
        - "tian_yan_cha"
        - "pretrained_model"
        - "xin_yong_xin_xi_zhong_xin"
    """
    r = requests.post(PyTorch_REST_API_URL + "load_model",
                      data={'model_name': model_name}).json()

    return r


def unload_model(model_name):
    """
    model_name: 
        - "tian_yan_cha"
        - "pretrained_model"
        - "xin_yong_xin_xi_zhong_xin"
    """
    r = requests.post(PyTorch_REST_API_URL + "unload_model",
                      data={'model_name': model_name}).json()
    return r


def label_one(file):
    with open(file, "rb") as f:
        base64_data = base64.b64encode(f.read())
        res = tyc_predict(base64_data)
        print(file.parent / (res['prediction'] +
                             "_{}".format(file.name.split("_")[1])))
        file.rename(
            file.parent / (res['prediction'] + "_{}".format(file.name.split("_")[1])))


def label_all_data():
    root = Path("/home/ruochi/Documents/share/wd_test_patent_pic/services/data/pictures")
    for direc in root.glob("*"):
        source = direc / "0"
        target = direc / "1"

        for file in source.glob("*"):
            label_one(file)
        for file in target.glob("*"):
            label_one(file)


if __name__ == '__main__':
    load_model("tian_yan_cha")
    label_all_data()
