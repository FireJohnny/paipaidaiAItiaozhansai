#!/usr/bin/env python
# encoding: utf-8

"""
__author__ = 'FireJohnny'
@license: Apache Licence 
@file: fc_model.py
@time: 2018/6/4 10:38
"""
import tensorflow as tf
from tensorflow.contrib import slim

class FC_model():
    def __init__(self,name = "fc_model"):
        self.name = name
    def __call__(self, x,reuse=False,layer_size= 100):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            net = slim.fully_connected(x,layer_size)
            net = slim.fully_connected(net,2,activation_fn=None)
        return net
        pass


if __name__ == '__main__':
    pass


