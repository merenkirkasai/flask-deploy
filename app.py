from __future__ import division, print_function
#from crypt import methods
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Flatten
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from skimage import transform
import tensorflow

app = Flask(__name__)

model = load_model('f.h5',compile=False)

def model_predict(img_path,model):  

    img = image.load_img(img_path,target_size=(100,100))    
    img = np.array(img).astype('float32')/255
    img = transform.resize(img,(100,100,3))
    img = np.expand_dims(img,axis=0)
    
    #img = image.img_to_array(img)
    #img = np.expand_dims(img,axis=0)

    preds = model.predict(img)
    print("Tahmin Olasılıkları :",preds)
    print("-------------------------------")
    preds=np.argmax(preds[0])
    print("Tahmin :",preds)
    return preds

@app.route('/',methods=['GET'])
def index():
    return render_template("index.html")

@app.route('/hakkında')
def hakkında():
    return "Bu Site Mehmet Eren Kırkaş Tarafından Oluşturulmultur."

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        preds = model_predict(file_path,model)
        os.remove(file_path)

        isim1 = 'Ferrari'
        isim2 = 'Mclaren'
        isim3 = 'Mercedes'
        isim4 = 'Redbull'

        if preds == 0:
            return isim1
        elif preds == 1:
            return isim2
        elif preds == 2:
            return isim3
        else :
            return isim4
    return None

if __name__ == '__main__':
    app.run(debug= True) 