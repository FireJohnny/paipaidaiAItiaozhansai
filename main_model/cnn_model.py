#!/usr/bin/env python
# encoding: utf-8

"""
__author__ = 'FireJohnny'
@license: Apache Licence 
@file: cnn_model.py
@time: 2018/6/3 19:12
"""

import tensorflow as tf
from tensorflow.contrib import  slim
from utils.utils import *

class cnn_model():
    def __init__(self, name = "cnn_model", out_size=64,pad = "VALID"):
        self.name = name
        self.out_size = out_size
        self.pad = pad
        pass
    def __call__(self, x, kernel_size=None, reuse=False, ):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            net = slim.conv2d(x, self.out_size, kernel_size,padding= self.pad,weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.02))
            # w_pool(variable_scope="w_pool",x,attention=)
        return net

if __name__ == '__main__':
    pass


