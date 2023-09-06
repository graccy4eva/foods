from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
# Keras
#from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
#from keras.preprocessing import image
from keras.utils import load_img, img_to_array
#import cv2
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from PIL import Image
import base64
from io import BytesIO
import io
#from tf.keras.applications.mobilenet import preprocess_input
#tf.keras.applications.mobilenet.preprocess_input
# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
#MODEL_PATH = 'models/model_resnet.h5'
MODEL_PATH = 'models/sequential_model3.h5'
#MODEL_PATH = ''
# Load your trained model
model = load_model(MODEL_PATH)
model.make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')


def preprocess_img(img_path):
    op_img = Image.open(img_path)
    op_img =  op_img.convert('RGB')
    img_resize = op_img.resize((150, 150))
    img2arr = img_to_array(img_resize)
    img2arr = np.expand_dims(img2arr, axis=0)
    img2arr /= 255.
    img_reshape = img2arr.reshape(1, 150, 150, 3)
    return img_reshape

# Predicting function
def predict_result(predict):
    pred = model.predict(predict)
    return np.argmax(pred[0], axis=-1)

def load_image(img_path, show=False):

    img = load_img(img_path, target_size=(150, 150))
    img_tensor = img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    return img_tensor

def model_predict(img_path, model):
    img = load_img(img_path, target_size=(150, 150))

    # Preprocessing the image
    x = img_to_array(img)
    # x = np.true_divide(x, 255)
    #x = x.reshape(1, 150, 150, 3)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #x = preprocess_input(x)
    x /= 255.

    preds = model.predict(x)
    #return 
    #target_names = ['egusi', 'ekwang', 'eru', 'jollof-ghana', 'jollof-nigeria', 'moi-moi', 'ndole', 'palm-nut-soup', 'waakye']
    return np.argmax(preds[0], axis=-1)


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        img = preprocess_img(request.files['file'].stream)
        preds = predict_result(img)
        # image_string = base64.b64encode(f.read())
        # image = base64_to_np(image_string)
        # img = preprocess_image(image)
        # preds = predict_result(np.array([img]))

        # Save the file to ./uploads
        # basepath = os.path.dirname(__file__)
        # file_path = os.path.join(basepath, secure_filename(f.filename))
        # #file_path = os.path.join(
        #  #   basepath, 'uploads', secure_filename(f.filename))
        # f.save(file_path)

        # Make prediction
        #preds = model_predict(image_string, model)
        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=9)   # ImageNet Decode
        #predictions=str(pred)
        #result = str(pred_class[0][0][1])               # Convert to string
        result = str(preds)
        #information = ""
        if result == "0":
            result= "Egusi" + '\n' + "Nutritional Info: Egusi soup contains 4 main ingredients:" + '\n' +  "Vegetables, Meat or Fish, Melon seed, Red oil." + '\n' + "This gives vitamins, protein, fats and oil."  + '\n' + "Learn More:" + "https://demandafrica.com/food/recipes/nigerian-egusi-soup/"
        if result == "1":
            result = "Ekwang" + '\n' + "Nutritional Info: Ekwang soup contains 4 main ingredients:" + '\n' +  "Cocoyam tubers, Vegetables, Meat or Fish and Red oil." + '\n' + "This gives carbohydrate, vitamins, protein, fats and oil." + '\n' + "Learn More:" + "https://www.africanbites.com/ekwang-ekpang-nkukwo/" 
        if result == "2":
            result=="Eru"
            result = "Eru" + '\n' + "Nutritional Info: Eru soup contains 3 main ingredients:" + '\n' +  "Vegetables, Meat or Fish and Red oil." + '\n' + "This gives vitamins, protein, fats and oil." + '\n' + "Learn More:" + "https://afrogistmedia.com/an-ultimate-guide-on-how-to-prepare-eru-the-cameroonian-style" 
        if result == "3":
            result=="Ghana Jollof-Rice"
            result = "Ghana Jollof-Rice" + '\n' + "Nutritional Info: Ghana Jollof-Rice contains 5 main ingredients:" + '\n' +  "Rice, Salad/Vegetables, Tomatoes, Meat or Fish and Vegetable oil." + '\n' + "This gives Carbohydrate, vitamins, protein, fats and oil." + '\n' + "Learn More:" + "https://tasty.co/recipe/ghanaian-jollof-rice-as-made-by-tei-hammond" 
        if result == "4":
            result=="Nigeria Jollof-Rice"
            result = "Nigeria Jollof-Rice" + '\n' + "Nutritional Info: Nigeria Jollof-Rice contains 4 main ingredients:" + '\n' +  "Rice, Tomatoes, Meat or Fish and Vegetable oil." + '\n' + "This gives Carbohydrate, vitamins,  protein, fats and oil." + '\n' + "Learn More:" + "https://cheflolaskitchen.com/jollof-rice/"
        if result == "5":
            result=="Moi-Moi(Beans Pudding)"
            result = "Moi-Moi(Beans Pudding)" + '\n' + "Nutritional Info: Moi-Moi(Beans Pudding) contains 3 main ingredients:" + '\n' +  "Beans, Egg or Fish and Vegetable oil." + '\n' + "This gives protein, fats and oil." + '\n' + "Learn More:" + "https://allnigerianfoods.com/moi-moi/"  
        if result == "6":
            result=="Ndole"
            result = "Ndole" + '\n' + "Nutritional Info: Ndole contains 5 main ingredients:" + '\n' +  "Shrimp, Meat, Vegetables, Vegetable oil and Peanut." + '\n' + "This gives protein, vitamins, fats and oil." + '\n' + "Learn More:" + "https://ingmar.app/blog/recipe-the-national-dish-of-cameroon-ndole-with-plantains/" 
        if result == "7":
            result=="Palm-nut Soup"
            result = "Palm-nut Soup" + '\n' + "Nutritional Info: Palm-nut Soup contains 4 main ingredients:" + '\n' +  "Tomatoes, Fish or Meat and Palm-nut Extract." + '\n' + "This gives vitamins, protein, fats and oil." + '\n' + "Learn More:" + "https://myrecipejoint.com/prepare-palm-nut-soup/"  
        if result == "8":
            result=="Waakye"
            result = "Waakye" + '\n' + "Nutritional Info: Waakye contains 5 main ingredients:" + '\n' +  "Rice, Beans, Egg, Fish and Vegetables." + '\n' + "This gives carbohydrate, protein and vitamins." + '\n' + "Learn More:" + "https://travelfoodatlas.com/ghanian-waakye-recipe"    
        return result 
    return "File cannot be processed."

if __name__ == '__main__':
    app.run(debug=True)

