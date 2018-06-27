#!/usr/bin/env python
# encoding: utf-8

"""
__author__ = 'FireJohnny'
@license: Apache Licence 
@file: data_read.py
@time: 2018/6/10 13:52
"""


import codecs
import csv
import random
import numpy as np
import pandas as pd

def load_data(question_file,train_file,test_file):
    all_data = {}
    question_text = pd.read_csv(question_file)
    train_file = pd.read_csv(train_file)
    test_file = pd.read_csv(test_file)

    all_words = question_text["words"]
    q1id = train_file["q1"]
    q2id = train_file["q2"]
    labels = train_file["label"]
    test_q1 = test_file["q1"]
    test_q2 = test_file["q2"]
    q1id = get_ids(q1id)
    q2id = get_ids(q2id)

    #load word vector

    train_text = []
    test_text = []
    for id in zip(q1id,q2id):
        train_text.append(all_words[id[0]] + ' ' + all_words[id[1]])


    for id in zip(test_q1,test_q2):
        test_text.append(all_words[id[0]]+" "+all_words[id[1]])
    pass

def get_texts(file_path, question_path):
    qes = pd.read_csv(question_path)
    file = pd.read_csv(file_path)
    q1id, q2id = file['q1'], file['q2']
    id1s, id2s = get_ids(q1id), get_ids(q2id)
    all_words = qes['words']
    all_chars = qes["chars"]
    words_texts = []
    for t_ in zip(id1s, id2s):
        words_texts.append(all_words[t_[0]] + ' ' + all_words[t_[1]])
    chars_texts = []
    for _t in zip(id1s,id2s):
        chars_texts.append(all_chars[_t[0]]+" "+all_chars[_t[1]])

    return words_texts,chars_texts


def load_w2v(file_dir):

    vocab = []
    vector = []
    with codecs.open(file_dir,encoding="utf8") as char_e:
        t = char_e.readlines()
        for _t in t:
            t1 = _t.strip().split(" ")[0]
            temp = _t.strip().split(" ")[1:]
            t2 = np.array(list(map(float,temp)))
            vocab.append(t1)
            vector.append(t2)
    return vocab,np.array(vector)
    pass

def get_vector(text,vec):
    pass

def get_ids(qids):
    ids = []
    for t_ in qids:
        ids.append(int(t_[1:]))
    return np.asarray(ids)


def balance_data(t1,label_1,use_ = True):
    label_1_ind = []
    label_0_ind = []
    text1 = []

    # label = label.tolist()
    for i in range(len(label_1)):
        if int(label_1[i]) == 0:
            label_0_ind.append(i)
        else:
            label_1_ind.append(i)

    for i in label_1_ind:
        text1.append(t1[i])
    if use_:
        label_0_ind_new = random.sample(label_0_ind,25000)
        for i in label_0_ind_new:
            text1.append(t1[i])
    else:
        label_0_ind_new = label_0_ind

    label1 = [[0, 1] for _ in range(len(label_1_ind)) ]
    label0 = [[1, 0] for _ in range(len(label_0_ind_new))]
    label_a = label1 + label0
    return text1,label_a

def creat_batch(text1,labels,batch_size = 64,random_data = True,):
    data_len  = len(text1)
    num_batch_per_epoch = int((data_len-1)/batch_size)+1
    if random_data:
        shuffle_indices = np.random.permutation(np.arange(data_len))
        shuffle_text1 = np.array(text1)[shuffle_indices]

        shuffle_lablels = labels[shuffle_indices]
    for batch in range(num_batch_per_epoch):
        start_index = batch*batch_size
        end_index = min((batch+1)*batch_size,data_len)
        yield shuffle_text1[start_index:end_index],shuffle_lablels[start_index:end_index]
        pass
    pass

if __name__ == '__main__':
    load_data("../data_set/question.csv","../data_set/train.csv","../data_set/test.csv")
    pass


