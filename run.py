import numpy as np
import tensorflow as tf
from vit import ViT
from utils import get_imagenet_classes,read_image,preprocess_inputs
import argparse

image_size = 384
classes = get_imagenet_classes()
vit = ViT()

parser = argparse.ArgumentParser(description="Process input image")
parser.add_argument('--image', type=str, default='sample.jpg')
args = parser.parse_args()

image = read_image(args.image,image_size)

X = preprocess_inputs(image).reshape(1, image_size, image_size, 3)
y = vit(X)

print(classes[tf.argmax(y[0][0])])