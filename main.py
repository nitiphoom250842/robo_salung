import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='3'
from typing import Union
from fastapi import FastAPI
import cv2
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import json
import numpy as np


with open('./application/model_ai/class_name.json', 'r',encoding='utf8') as json_class:
    class_names = json.loads(json_class.read())


with open('./application/model_ai/architecture.json', 'r') as json_file:
    model_j = model_from_json(json_file.read())
    model_j.load_weights('./application/model_ai/model.h5')

app = FastAPI()


@app.get("/")
def read_root():
    return {"data": 1}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}