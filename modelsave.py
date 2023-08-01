from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import keras
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.python.keras.saving.saved_model import load as load_model
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from tensorflow.keras.applications.resnet50 import ResNet50
#from keras.applications.resnet50 import ResNet50
model = ResNet50(weights='imagenet')
model.save('models/saved_model.pb')
model = tf.saved_model.load('models/saved_model.pb')
keras_model = load_model(model)
keras_model.save('models/save_model.h5')