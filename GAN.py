#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 08:42:56 2020

@author: roger"""

from flask import Flask

from flask import render_template, request
#from keras.models import load_model
from tensorflow.keras.models import load_model
import tensorflow as tf
import keras
import IPython.display as display
from PIL import Image
import os 

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))+'/'

app = Flask(__name__)

generator = load_model(ROOT_DIR+'generator.h5')
num_img = 30
latent_dim = 128
def avatar_gen():
    random_latent_vectors = tf.random.normal(shape=(num_img, latent_dim))
    generated_images = generator(random_latent_vectors)
    generated_images *= 255
    generated_images.numpy()
    for i in range(num_img):
        img = keras.preprocessing.image.array_to_img(generated_images[i])
        img.save(ROOT_DIR+"static/generated_img_%d.png" % (i))

    
@app.route('/')
def index():  
    avatar_gen()
    return render_template('layout.html')

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5001)