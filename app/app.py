import numpy as np
import flask
from flask import Flask,flash, request,redirect, url_for, jsonify, render_template, session

from ModelClass import DownstreamModel, Model, CharClassificationModel, GJImageModel

import cv2
import base64
from io import BytesIO
import argparse
import os
import pickle as pkl

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils as utils
import albumentations as A
from albumentations.pytorch import ToTensor
from torchvision import transforms as T

from PIL import Image

## api from yolov4
from interface import split_img
from yolov4_pytorch.yolo import YOLO

## api from jone
from gj_interface import cut_single_txt



app = Flask(__name__)
app.secret_key = 'dev'


split_model = None

tyc_model = None
tyc_index2char = None

pretrained_model = None
pretrained_index2char = None

gj_char_model = None
gj_char_index2char = None

gj_image_model = None
gj_image_index2class = None



with open("../OCR/models/tyc_model_checkpoint/cache.pkl", "rb") as f:
        tyc_index2char = pkl.load(f)["index2char"]

with open("../OCR/models/ocr_pretrained_model_checkpoint/chinese_labels", "rb") as f:
        pretrained_index2char = pkl.load(f)

with open("../OCR/models/gj_model_checkpoint/cache.pkl", "rb") as f:
        gj_char_index2char = pkl.load(f)["index2char"]

with open("../OCR/models/gj_image_model_checkpoint/cache.pkl", "rb") as f:
        gj_image_index2class = pkl.load(f)["index2char"]


use_gpu = False


def load_tyc_model():
    """Load the pre-trained model, you can use your model just as easily.

    """
    global tyc_model

    checkpoint = torch.load("../OCR/models/tyc_model_checkpoint/best_model.pt",map_location=torch.device('cpu'))

    tyc_model = DownstreamModel(len(tyc_index2char))
    tyc_model.load_state_dict(checkpoint["model_state_dict"])

    tyc_model.eval()
    if use_gpu:
        tyc_model.cuda() 

def load_pretrained_model():
    """Load the pre-trained model, you can use your model just as easily.

    """
    global pretrained_model

    checkpoint = torch.load("../OCR/models/ocr_pretrained_model_checkpoint/best_model.pt",map_location=torch.device('cpu'))

    pretrained_model = Model(len(pretrained_index2char))
    pretrained_model.load_state_dict(checkpoint["model_state_dict"])

    pretrained_model.eval()
    if use_gpu:
        pretrained_model.cuda() 

def load_gj_char_model():
    """Load the pre-trained model, you can use your model just as easily.

    """
    global gj_char_model

    checkpoint = torch.load("../OCR/models/gj_model_checkpoint/best_model.pt",map_location=torch.device('cpu'))

    gj_char_model = CharClassificationModel(len(gj_char_index2char))
    gj_char_model.load_state_dict(checkpoint["model_state_dict"])

    gj_char_model.eval()
    if use_gpu:
        gj_char_model.cuda() 

def load_gj_image_model():
    """Load the pre-trained model, you can use your model just as easily.

    """
    global gj_image_model

    checkpoint = torch.load("../OCR/models/gj_image_model_checkpoint/best_model.pt", map_location=torch.device('cpu'))

    gj_image_model = GJImageModel(len(gj_image_index2class))
    gj_image_model.load_state_dict(checkpoint["model_state_dict"])

    gj_image_model.eval()
    if use_gpu:
        gj_image_model.cuda() 


def load_split_model():
    global split_model

    split_model = YOLO()
    

def split_gj(raw_image, crop_area):
    """
    裁切出验证码文字部分和图像部分
    :param raw_image: Image对象或图像路径
    :param mode: 图中有几组验证码文字
    :return:
    """

    contents = [raw_image[region[1]:region[3], region[0]:region[2]]
                for region in crop_area]
    chars = contents[0]
    images = contents[1:]

    return chars, images


def prepare_image(image, test_transform):
    """Do image preprocessing before prediction on any data.

    :param image:       original image
    :param target_size: target image size
    :return:
                        preprocessed image
    """
    image = test_transform(image=image)["image"]

    # Add batch_size axis.
    image = torch.unsqueeze(image,0)

    if use_gpu:
        image = image.cuda()

    return image


def predict_one(model, image, test_transform, index2char):
    image = prepare_image(image, test_transform)
    logits, probas = model(image)
    _, pred_label = torch.max(probas, 1)    
    pred_char = index2char[pred_label.item()]

    return pred_char


@app.route("/pretrained_predict", methods=["POST"])
def pretrained_predict():
    # Initialize the data dictionary that will be returned from the view.
    data = {"success": False}

    test_transform = A.Compose([
        A.Resize(36, 36),
        A.CenterCrop(32, 32),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensor()
    ])

    # Ensure an image was properly uploaded to our endpoint.
    if request.method  == 'POST':

        ## base64 format
        byte_data = base64.b64decode(str(request.form['image']))
        image_data = BytesIO(byte_data)
        image = Image.open(image_data)
        splited_images, locations = split_img(split_model, image)

        res = []

        # Classify the input image and then initialize the list of predictions to return to the client.
        
        for i, (image, location) in enumerate(zip(splited_images, locations)):
            pred_char = predict_one(
                pretrained_model, image, test_transform, pretrained_index2char)
            res.append({"label": pred_char, "location": location})
        data['prediction'] = res
        # Indicate that the request was a success.
        data["success"] = True

    # Return the data dictionary as a JSON response.
    return flask.jsonify(data)



@app.route("/tyc_predict", methods=["POST"])
def tyc_predict():
    # Initialize the data dictionary that will be returned from the view.
    data = {"success": False}

    test_transform = A.Compose([
        A.Resize(36, 36),
        A.CenterCrop(32, 32),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensor()
    ])

    # Ensure an image was properly uploaded to our endpoint.
    if request.method  == 'POST':

        ## base64 format
        byte_data = base64.b64decode(str(request.form['image']))
        # image = np.fromstring(byte_data, np.uint8)
        # image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_data = BytesIO(byte_data)
        image = Image.open(image_data)
        splited_images,locations = split_img(split_model,image)

        res = []

        # Classify the input image and then initialize the list of predictions to return to the client.
        
        for i, (image, location) in enumerate(zip(splited_images, locations)):
            pred_char = predict_one(
                tyc_model, image, test_transform, tyc_index2char)
            res.append({"label": pred_char, "location": location})
        data['prediction'] = res
        # Indicate that the request was a success.
        data["success"] = True

    # Return the data dictionary as a JSON response.
    return flask.jsonify(data)


@app.route("/gj_predict", methods=["POST"])
def gj_predict():
    # Initialize the data dictionary that will be returned from the view.
    data = {"success": False}

    test_char_transform = A.Compose([
        A.Resize(32,32),
        A.ToGray(p=1.0),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensor()
    ])

    test_image_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensor()
    ])

    crop_area = [(0, 344, 116, 384), (0, 0, 111, 111), (116, 0, 227, 111), (232, 0, 343, 111), (0, 116, 111, 227),
                 (116, 116, 227, 227), (232, 116, 343, 227), (0, 232, 111, 343), (116, 232, 227, 343), (232, 232, 343, 343)]

    # Ensure an image was properly uploaded to our endpoint.
    if request.method == 'POST':

        ## base64 format
        image = base64.b64decode(str(request.form['image']))
        image = np.fromstring(image, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        chars, images = split_gj(image, crop_area)
        char_images = cut_single_txt(chars)

        res = []

        # Classify the input image and then initialize the list of predictions to return to the client.
        target = ""
        for i,char_img in enumerate(char_images):
            target += predict_one(gj_char_model, char_img,
                                  test_char_transform, gj_char_index2char)

        res.append({"target": target})

        for i, image in enumerate(images):
            pred_class = predict_one(
                gj_image_model, image, test_image_transform, gj_image_index2class)

            if pred_class == target:
                res.append({"index": i+1, "location": crop_area[1:][i]})
    
        data['prediction'] = res
        # Indicate that the request was a success.
        data["success"] = True

    # Return the data dictionary as a JSON response.
    return flask.jsonify(data)

def prepare_app():

    load_split_model()
    load_tyc_model()
    load_pretrained_model()
    load_gj_char_model()
    load_gj_image_model()


prepare_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port = 5000)

    # image = cv2.imread("test_images/1/猿_4.png")
    # pred_char = predict_one(image)
    # print(pred_char)
