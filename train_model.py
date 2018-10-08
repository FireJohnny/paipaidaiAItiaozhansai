#!/usr/bin/env python
# encoding: utf-8

"""
__author__ = 'FireJohnny'
@license: Apache Licence 
@file: model_for_train.py
@time: 2018/6/10 13:44
"""

from data_process.data_read import *
import tensorflow as tf
import model_for_train.Graph_model as m
import time
import os
from tensorflow.contrib import learn
from collections import Counter
import gensim

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,"
def train():
    char_dir = "./data_set/char_embed.txt"

    word_dir = "./data_set/word_embed.txt"
    question_file = "./data_set/question.csv"
    train_file = "./data_set/train.csv"
    test_file = "./data_set/test.csv"



    # char_vector = load_w2v(char_dir)
    vocab,vector = load_w2v(word_dir)
    # print(vector.shape)
    train_words, train_chars = get_texts(train_file, question_file)
    test_words, test_chars = get_texts(test_file, question_file)
    t = [len(l) for l in train_words+test_words ]
    t_1 = [len(l)for l in train_chars+test_chars]
    x = Counter(t)
    x_1 = Counter(t_1)
    labels_1 = pd.read_csv(train_file)["label"]
    _labels = []
    for i in labels_1:
        if i ==0:
            _labels.append([1,0])
        else:
            _labels.append([0,1])


    max_len = 210


    vocab_process = learn.preprocessing.VocabularyProcessor(max_len)
    vocab_process.fit(vocab)
    t_1 = list(vocab_process.transform(train_words))
    #balance data

    t_1, labels = balance_data(t_1, labels_1, use_=False)
    vocab_len = len(vocab_process.vocabulary_)
    vocab_process.save("vocab")

    dev_sample_index = -1 * int(0.0005 * float(len(labels)))

    train_1,dev_1 = t_1[:dev_sample_index],t_1[dev_sample_index:]
    train_labels,dev_labels = np.array(_labels)[:dev_sample_index],np.array(_labels)[dev_sample_index:]

    epoches = 500

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            model = m.Graph_model(embed_size=300, length=max_len,  filter_size=4, conv_out=64, rnn_cell=64,embed_w=vector,vocab_len =vocab_len )
            model.build()
            sess.run(tf.global_variables_initializer())
            # Saver = tf.
            timenow = str(int(time.time()))
            if not os.path.exists("./log/"+timenow):
                os.mkdir("./log/"+timenow)
            if not os.path.exists("./log/"+timenow+"/train"):
                os.mkdir("./log/"+timenow+"/train")
            if not os.path.exists("./log/"+timenow+"/dev"):
                os.mkdir("./log/"+timenow+"/dev")
            trainWriter=tf.summary.FileWriter("./log/"+timenow+"/train",sess.graph)
            devWriter=tf.summary.FileWriter("./log/"+timenow+"/dev",sess.graph)
            saver = tf.train.Saver(tf.global_variables(),max_to_keep=0)
            def train(train_1,_labels,step):

                for b_1,labels in creat_batch(train_1,_labels,batch_size=128):
                    feed_dict = {
                        model.q1:b_1,

                        model.labels:labels,
                        model.keep_prob:0.5
                    }

                    _,acc,summary = sess.run([model.opt,model.acc,model.summary],feed_dict=feed_dict)
                    trainWriter.add_summary(summary,global_step=step)
                    print("step:{}, acc:{}".format(step,acc))
                    if step >9999 and step %1000==0:
                        for dev_t ,devlabels in creat_batch(dev_1,dev_labels,batch_size=128):

                            dev(dev_t,devlabels,step)

                        if not os.path.exists("./log/"+timenow+"/model"):
                            os.mkdir("./log/"+timenow+"/model")
                        saver.save(sess,"./log/"+timenow+"/model/model",global_step=step)
                    step+=1
                return step

            def dev(test_1,labels,step):
                feed_dict = {
                        model.q1:test_1,
                        model.labels:labels,
                        model.keep_prob:1.0
                }
                acc,summary = sess.run([model.acc,model.summary],feed_dict=feed_dict)
                devWriter.add_summary(summary,global_step=step)

                print("step:{}, acc:{}".format(step,acc))

            step =1
            for epoch in range(epoches):
                for t,label in creat_batch(train_1,train_labels):
                    step = train(t,label,step)
                print("\n\tepoch: {}".format(epoch+1))
def out(input_path,out_path):
    test_file = "./data_set/test.csv"
    max_len = 20
    test_words, test_chars = get_texts(input_path)
    vocab_process = learn.preprocessing.VocabularyProcessor.restore("./vocab")
    # vocab_process.restore("./vocab")
    t_1 = list(vocab_process.transform(test_words))

    vocab = len(vocab_process.vocabulary_)
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            model = m.Graph_model(emb_dim=100, length=max_len, vocab_size=vocab, filter_size=2, conv_out=64, lstm_cell=32)
            model.build()
            # sess.run(tf.global_variables_initializer())
            # ckpt = tf.train.load_checkpoint("./log/model/model-4000")#_checkpoint("./log/model/model-4000")
            saver = tf.train.Saver()
            saver.restore(sess,"./log/model/model-10000")
            def final(out_t1,out_t2, index):
                feed_dict={
                    model.x_1:out_t1,
                    model.x_2:out_t2,
                    model.keep_prob:1.0
                }
                pre = sess.run([model.pre], feed_dict=feed_dict)
                with codecs.open(out_path, mode="a", encoding="utf-8") as out:
                    for label in pre[0]:
                        w_str = now_index+"\t"+str(label)+"\n"
                        out.writelines(w_str)
                        index +=1
                        pass
                return index

            len_t = len(t_1)

            i = 1000
            count_index = 0
            for x in range(0,len_t,i):

                _t1, _t2 = t_1[x:i+x]
                count_index = final(_t1, _t2, count_index)

if __name__ == '__main__':
    train()
    pass


