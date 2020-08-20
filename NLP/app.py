import numpy as np
import flask
from flask import Flask,flash, request,redirect, url_for, jsonify, render_template, session
import sys

from bert_serving.client import BertClient


app = Flask(__name__)
app.secret_key = 'dev'



BC = BertClient()


@app.route("/bert_embedding", methods=["POST"])
def bert_embedding():
    data = {"success": False}

    if request.method == 'POST':
        doc = request.form['doc']
        embedding = BC.encode([doc])
        data["embedding"] = embedding.tolist()
        data["success"] = True

    return flask.jsonify(data)



if __name__ == "__main__":
    app.run(host="0.0.0.0", port = 5001)
