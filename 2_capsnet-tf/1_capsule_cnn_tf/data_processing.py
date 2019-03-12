#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import Counter
import tensorflow.contrib.keras as kr
import numpy as np
import pickle
import jieba
import os
jieba.load_userdict("/opt/algor/gongxf/python3_pj/Robot/original_data/finWordDict.txt")
base_path='/opt/algor/gongxf/python3_pj/nlp_practice/19_capsule/2_capsnet-tf/1_capsule_cnn_tf'



def open_file(filename, mode='r'):
    """    Commonly used file reader, change this to switch between python2 and python3.    mode: 'r' or 'w' for read or write    """
    return open(filename, mode, encoding='utf-8', errors='ignore')
 #读取停止词
def get_stop_words():
    stop_words=[]
    with open('/opt/algor/gongxf/python3_pj/Robot/original_data/stop_words.txt','r',encoding="utf-8") as f:
        line=f.readline()
        while line:
            stop_words.append(line[:-1])
            line=f.readline()
    return stop_words

stopwords_list=get_stop_words()
def read_file(filename):
    """读取文件数据"""
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                content_cut = list(jieba.cut(content, cut_all=False))
                # 去停用词
                contents.append([word for word in content_cut if word not in stopwords_list])
                labels.append(label)
            except:
                pass
    return contents, labels

# def build_vocab(train_dir, vocab_dir, vocab_size=5000):
#     """根据训练集构建词汇表，存储"""
#     data_train, _ = read_file(train_dir)
#
#     all_data = []
#     for content in data_train:
#         all_data.extend(content)
#
#     counter = Counter(all_data)
#     count_pairs = counter.most_common(vocab_size - 1)
#     words, _ = list(zip(*count_pairs))
#     # 添加一个 <PAD> 来将所有文本pad为同一长度
#     words = ['<PAD>'] + list(words)
#
#     open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')

def read_vocab(vocab_dir):
    """读取词汇表"""
    words = open_file(vocab_dir).read().strip().split('\n')
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id

def read_category():
    """读取分类目录，固定"""
    pickle_file = open(base_path+'/data/categories.pkl', 'rb')
    categories = pickle.load(pickle_file)
    cat_to_id = dict(zip(categories, range(len(categories))))
    return categories, cat_to_id

def to_words(content, words):
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)

def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)
    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])
    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id)  # 将标签转换为one-hot表示
    return x_pad, y_pad

def process_file_predict(text, word_to_id, max_length=600):
    """将文件转换为id表示"""
    contents = []
    content_cut = list(jieba.cut(text, cut_all=False))
    # 去停用词
    contents.append([word for word in content_cut if word not in stopwords_list])
    print("contents",contents)
    data_id = []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    return x_pad

def process_file_poolout(filename, word_to_id, cat_to_id, max_length=600):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)
    # print("contents",contents,"labels",labels)
    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])
    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id)  # 将标签转换为one-hot表示
    return x_pad, y_pad,contents,labels


def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1
    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]