from pandas import concat
import tensorflow
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import base64
import io
from io import BytesIO
from PIL import Image
from flask import Response, jsonify
import time
import os
class Predict():
    def createPath(img):
        imgs = Image.open(io.BytesIO(base64.b64decode(img)))
        timestr = time.strftime("%Y%m%d-%H%M%S")
        imgs.save('app//imagetopredict//'+ timestr +'.png', 'PNG')
        path = 'app//imagetopredict//'+ timestr +'.png'
        return path
    def prediction(path):
        class_names = ['AbdomenCT', 'BreastMRI', 'CXR', 'ChestCT', 'Hand', 'HeadCT']
        model = load_model('app//modelvgg.h5',compile = False) 
        img_path = path
        img = image.load_img(img_path, target_size=(64, 64))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds=model.predict(x)
        maxElement = np.amax(preds)
        result = np.where(preds == maxElement)
        index = result[1]
        num = index[0]
        print(class_names[num])
        print(path)
        os.remove(path)
        return Response(class_names[num])
