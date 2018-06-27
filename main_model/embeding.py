#!/usr/bin/env python
# encoding: utf-8

"""
__author__ = 'FireJohnny'
@license: Apache Licence 
@file: embeding.py
@time: 2018/6/3 19:35
"""
import tensorflow as tf


class embed():
    def __init__(self, name = "embed",vocab_size=None,embed_size=300):
        self.name = name
        self.vocab_size = vocab_size
        self.embed_size = embed_size
    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            with tf.device("/CPU:0"):
                embed_w = tf.Variable(
                    tf.random_uniform([self.vocab_size, self.embed_size], -1.0, 1.0),
                    name="embed_W")
                vec = tf.nn.embedding_lookup(embed_w, x)
                vec = tf.transpose(vec,[0,2,1])
                expend_vec = tf.expand_dims(vec,-1)
        return expend_vec
        pass

class pre_embed():
    def __init__(self, name  = "embed"):
        self.name = name
    def __call__(self, x):
        pass


if __name__ == '__main__':
    pass


