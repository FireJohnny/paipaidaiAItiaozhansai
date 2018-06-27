#!/usr/bin/env python
# encoding: utf-8

"""
__author__ = 'FireJohnny'
@license: Apache Licence 
@file: rnn_model.py
@time: 2018/6/3 19:20
"""

import tensorflow as tf
from tensorflow.contrib import  slim

class rnn_model():
    def __init__(self,name = "rnn_model",cell_size =256):
        self.name = name
        self.cell_size = cell_size
    def __call__(self, x, reuse = False, num_layer = 1):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.cell_size ,)
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layer)
            outputs,states = tf.nn.static_rnn(cell,x)
        return outputs


class birnn_model():
    def __init__(self, name = "birnn_model", cell_size=256):
        self.name = name
        self.cell_size = cell_size
    def __call__(self, x, reuse = False, num_layer=1):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            fw_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.cell_size)
            bw_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.cell_size)
            fw_cell = tf.nn.rnn_cell.MultiRNNCell([fw_cell]*num_layer)
            bw_cell = tf.nn.rnn_cell.MultiRNNCell([bw_cell]*num_layer)
            outputs, output_state_fw, output_state_bw = tf.nn.static_bidirectional_rnn(fw_cell, bw_cell, x, dtype=tf.float32)

        return outputs
        pass

if __name__ == '__main__':
    pass


