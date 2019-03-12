#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
from build_nnmodel import *
from data_processing import *
from sklearn import metrics
import numpy as np
import tensorflow as tf
import time
from datetime import timedelta


base_dir = '/opt/gongxf/python3_pj/nlp_practice/5_context_classification/dnn_model/data/'
train_dir = os.path.join(base_dir, 'train.txt')
test_dir = os.path.join(base_dir, 'aaa.txt')
val_dir = os.path.join(base_dir, 'test.txt')
vocab_dir = os.path.join(base_dir, 'vocab_word2vec.txt')

save_dir = '/opt/gongxf/python3_pj/nlp_practice/5_context_classification/dnn_model/result_conv2d/'
save_path_cnn = os.path.join(save_dir, 'cnn/best_validation')   # 最佳验证结果保存路径
save_path_rnn = os.path.join(save_dir, 'rnn/best_validation')   # 最佳验证结果保存路径

ff_pool=open("/opt/gongxf/python3_pj/nlp_practice/5_context_classification/dnn_model/data/test_pool.txt",'w',encoding='utf-8')


class CNN_Predict(object):
    def __init__(self):
        self.categories, self.cat_to_id = read_category()
        self.id_to_cate = dict(zip(self.cat_to_id.values(), self.cat_to_id.keys()))
        self.words, self.word_to_id = read_vocab(vocab_dir)
        self.config = TCNNConfig()
        self.config.vocab_size = len(self.words)
        self.model = TextCNN(self.config)
        variable_name = [v.name for v in tf.trainable_variables()]
        print("11111",variable_name)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        variable_name = [v.name for v in tf.trainable_variables()]
        print("22222",variable_name)
        self.saver = tf.train.Saver()
        self.saver.restore(sess=self.session, save_path=save_path_cnn)  # 读取保存的模型
        variable_name = [v.name for v in tf.trainable_variables()]
        print("33333", variable_name)

    def predict(self,text):
        x_test = process_file_predict(text, self.word_to_id,self.config.seq_length)
        print("x_test", x_test)
        feed_dict = {self.model.input_x: x_test,self.model.keep_prob: 1.0}
        logits,logits_softmax= self.session.run([self.model.logits,self.model.logits_softmax], feed_dict=feed_dict)
        maximum_probability = np.max(logits_softmax[0])
        index_max = np.where(logits_softmax[0] == maximum_probability)[0][0]
        pre_cate=self.id_to_cate[index_max]
        print("maximum_probability------index_max",maximum_probability,index_max,pre_cate)
        return pre_cate,maximum_probability


if __name__ == '__main__':
    cnn_predict=CNN_Predict()
    while True:
        text=input("请输入测试文本：")
        cnn_predict.predict(text)
    # cnn_predict.predict("我们是任性贷")
