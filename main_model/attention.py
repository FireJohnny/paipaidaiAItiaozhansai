#!/usr/bin/env python
# encoding: utf-8

"""
__author__ = 'FireJohnny'
@license: Apache Licence 
@file: attention.py
@time: 2018/6/3 21:04
"""
import tensorflow as tf
from tensorflow.contrib import  slim

def attention(inputs, attention_size ):
    """
    Attention mechanism layer.
    :param inputs: outputs of RNN/Bi-RNN layer (not final state)
    :param attention_size: linear size of attention weights
    :return: outputs of the passed RNN/Bi-RNN reduced with attention vector
    """
    # In case of Bi-RNN input we need to concatenate outputs of its forward and backward parts
    if isinstance(inputs, tuple):
        inputs = tf.concat(2, inputs)

    sequence_length = inputs.get_shape()[1].value  # the length of sequences processed in the antecedent RNN layer
    hidden_size = inputs.get_shape()[2].value  # hidden size of the RNN layer

    # Attention mechanism
    with tf.variable_scope("weight_and_bias",reuse=False):
        W_omega = tf.get_variable(name = "W_omega", initializer=tf.random_normal([hidden_size, attention_size], stddev=0.1))
        b_omega = tf.get_variable(name = "b_omega", initializer=tf.random_normal([attention_size], stddev=0.1))
        u_omega = tf.get_variable(name = "u_omega", initializer=tf.random_normal([attention_size], stddev=0.1))

    v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))
    vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
    exps = tf.reshape(tf.exp(vu), [-1, sequence_length])
    alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])

    # Output of Bi-RNN is reduced with attention vector
    output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1)
    #if l2_reg_lambda > 0:
    #    l2_loss += tf.nn.l2_loss(W_omega)
    #    l2_loss += tf.nn.l2_loss(b_omega)
    #    l2_loss += tf.nn.l2_loss(u_omega)
    #    tf.add_to_collection('losses', l2_loss)

    return output


class Attention_1():
    def __init__(self,name = "Attention_1",size = 64):
        self.name = name
        self.size = size
    def __call__(self,x,reuse=False,):
        with tf.variable_scope(self.name ) as scope:
            if reuse:
                scope.reuse_variables()
            outputs = attention(x,self.size)
        return outputs


class Attention_2():
    def __init__(self, name = "Attention_2"):
        self.name = name
    def __call__(self, x_1,x_2, reuse=False,):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            euclidean = tf.sqrt(tf.reduce_sum(tf.square(x_1 - tf.matrix_transpose(x_2)), axis=1))
            return 1/(1+euclidean)


        

if __name__ == '__main__':
    pass


