import tensorflow as tf
import numpy as np

class Linear(tf.keras.layers.Layer):
    def __init__(self,weight,bias,layer_name):
        super().__init__()
        self.num_filters = bias.shape[-1]
        self.weight = weight
        self.bias = bias
        self.layer_name = layer_name

    def call(self,inputs):
        matmul = tf.keras.layers.Conv2D(filters=self.num_filters, kernel_size=1,
                                         strides=1, name=self.layer_name)
        tmp = matmul(tf.zeros(shape=inputs.shape))
        matmul.set_weights([self.weight,self.bias])
        return matmul(inputs)



