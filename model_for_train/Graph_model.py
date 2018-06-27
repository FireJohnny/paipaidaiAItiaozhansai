#!/usr/bin/env python
# encoding: utf-8

"""
__author__ = 'FireJohnny'
@license: Apache Licence 
@file: Graph_model.py
@time: 2018/6/14 22:24
"""
from main_model.attention import *
from main_model.cnn_model import *
from main_model.embeding import *
from main_model.rnn_model import *
from main_model.fc_model import *

import tensorflow as tf




class Graph_model(object):
    def __init__(self, embed_size, conv_out, rnn_cell, length,vocab_len ,embed_w,filter_size):
        self.embed_size = embed_size
        self.conv_out = conv_out
        self.rnn_cell = rnn_cell
        self.length = length
        self.vocab = vocab_len
        self.emb_w = embed_w
        self.filter_size = filter_size
        pass

    def _placeholder(self):
        self.q1 = tf.placeholder(dtype=tf.int32, shape=[None, self.length])
        self.labels = tf.placeholder(dtype=tf.int32, shape=[None, 2])
        # self.emb_w = tf.placeholder(dtype=tf.float32,shape=[None,self.vocab,self.embed_size])
        self.keep_prob = tf.placeholder(dtype=tf.float32)
    def creat_model(self):
        self.cnn = cnn_model(pad="SAME")
        self.rnn = birnn_model(cell_size=self.rnn_cell)
        self.att = Attention_1()
        self.fc = FC_model()

    def embed(self,x):
        embed_w = tf.Variable(tf.constant(self.emb_w,
                                          shape=[self.vocab,self.embed_size]),
                                          trainable = True,
                                          name = "embed_w")
        embed_vec = tf.nn.embedding_lookup(embed_w,x)
        return embed_vec
    def build(self):
        self.creat_model()
        self._placeholder()

        input = self.embed(self.q1)
        input = tf.cast(tf.expand_dims(input, -1),dtype=tf.float32,)

        with tf.name_scope("CNN_layer"):
            net = self.cnn(input,kernel_size=[self.filter_size,self.embed_size])
            net = tf.nn.avg_pool(net,ksize=[1,1,self.embed_size,1],strides=[1, 1, 1, 1],padding="VALID")
        net = self.squeeze_data(net)
        with tf.name_scope("LSTM_layer"):
            net = self.rnn(net)
        net = tf.transpose(net, [1,0,2])
        with tf.name_scope("Attention_layer"):
            net = self.att(net)
        with tf.name_scope("FC_layer"):
            self.final_out = self.fc(net)

        self.predict()
        self.loss()
        self.optimzer()
        self.acc()
        self.Summary()


    def squeeze_data(self,x):
        # import pdb
        # pdb.set_trace()
        _input = tf.reshape(x,[-1, self.length,self.conv_out])
        input_x = [tf.squeeze(input, [1]) for input in tf.split(_input, self.length, 1)]
        return input_x

    def predict(self):
        self.pre = tf.argmax(self.final_out,1)

    def acc(self):
        self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.final_out,1),tf.argmax(self.labels,1)),tf.float32))
    def loss(self):
        self.losses = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=self.labels, logits=self.final_out))

    def optimzer(self):
        self.opt  = tf.train.AdamOptimizer().minimize(self.losses)

    def Summary(self):
        tf.summary.scalar("loss",self.losses)
        tf.summary.scalar("acc",self.acc)
        self.summary = tf.summary.merge_all()
if __name__ == '__main__':
    pass


