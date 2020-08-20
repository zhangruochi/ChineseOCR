import base64
import requests
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# ||| 

PyTorch_REST_API_URL = 'http://172.16.20.21:5001/'

def pretrained_predict(doc):
 
    # Submit the request.
    r = requests.post(PyTorch_REST_API_URL + "bert_embedding", data = {'doc':  doc}).json() 
   
    return r



if __name__ == "__main__":
    with open("test_doc.txt","r") as f:
        all_docs = f.readlines()
    sentences = all_docs[0].split("；")

    ## predict single sentence

    res = pretrained_predict(sentences[0])
    print(res["embedding"])

    # ## predict  doc (concat a list of sentences to a string)
    # ## concatenating them with ||| (with whitespace before and after)
    doc = " ||| ".join(sentences)
    # print(doc)
    res = pretrained_predict(doc)
    print(res["embedding"])
