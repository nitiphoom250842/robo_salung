import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='3'
from typing import Union
from datetime import datetime
import pickle
import io
from fastapi import FastAPI,File,UploadFile
import cv2
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import json
import numpy as np
import pendulum


with open('./model_ai/class_name.json', 'r',encoding='utf8') as json_class:
    class_names = json.loads(json_class.read())


with open('./model_ai/architecture.json', 'r') as json_file:
    model_j = model_from_json(json_file.read())
    model_j.load_weights('./model_ai/model.h5')

app = FastAPI()
orb = cv2.ORB_create(nfeatures=100) 
bf = cv2.BFMatcher()


@app.get("/")
def read_root():
    start_time=datetime.now()
    data=ai()
    end_time=datetime.now()
    return {"result": data[0],
            "score":data[1],
            "time":end_time-start_time}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.post("/files/")
async def create_file(file: bytes = File(...)):
    name= str(pendulum.now("Asia/Bangkok")).split('.')[0]
    open("./images/"+name.replace(':','_')+".jpg","wb").write(file)



   


def ai():
    
    # ret,img = cap.read()
    img=cv2.imread('./images/2022-05-21 09_37_46.491272.jpg')
    img_height=224
    img_width=224
    img_resize =cv2.resize(img,(img_height,img_width))
    # cv2.imwrite('./image/test_.jpg',img)
    img_resize =cv2.resize(img,(img_height,img_width))
    img_array = tf.keras.utils.img_to_array(img_resize)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = model_j.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    print(class_names[np.argmax(score)],"ค่าความแม่นยำที่",100 * np.max(score))
    return class_names[np.argmax(score)],100 * np.max(score)