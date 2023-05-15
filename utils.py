import os
import typing
import urllib
from urllib import request
import io
import numpy as np
import scipy as sp
import cv2
import tensorflow as tf

def get_imagenet_classes() -> typing.List[str]:
    filepath = "imagenet2012.txt"
    with open(filepath) as f:
        classes = [l.strip() for l in f.readlines()]
    return classes

def read_image(input_image,size=384):
    if isinstance(input_image, str):  
        if input_image.startswith('http'):  
            resp = urllib.request.urlopen(input_image)
            image_array = np.asarray(bytearray(resp.read()), dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        else:  
            image = cv2.imread(input_image)
    elif isinstance(input_image, np.ndarray): 
        image = input_image
    else:
        print("Error: Invalid input image format.")
    return cv2.resize(image, (size, size))

def preprocess_inputs(X):
    return tf.keras.applications.imagenet_utils.preprocess_input(X, data_format=None, mode="tf")


