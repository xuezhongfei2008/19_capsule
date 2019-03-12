import tensorflow as tf

import tensorflow as tf
import numpy as np
import pickle

base_path='/opt/algor/gongxf/python3_pj/nlp_practice/19_capsule/2_capsnet-tf/1_capsule_cnn_tf'

def load_embedding():
    #读取word2vec词向量
    embedding_array = np.load(base_path+"/data/embedding_word2vec.npy")
    #获取词汇大小
    vocab_size=embedding_array.shape[0]
    #回去词向量 shape
    embedding_dim=embedding_array.shape[1]
    #返回labels的种类数
    pickle_file = open(base_path+'/data/categories.pkl', 'rb')
    categories = pickle.load(pickle_file)
    categories_len=len(categories)
    return embedding_array,vocab_size,embedding_dim,categories_len

class TCNNConfig(object):
    """CNN配置参数"""
    embedding_array=load_embedding()[0]         #初始化预训练的词向量
    embedding_dim = load_embedding()[2]         # 词向量维度,使用gensim 设置的值
    seq_length = 8                              # 序列长度
    num_classes = load_embedding()[3]           # 类别数,也作为胶囊网络的 胶囊个数
    vec_len=16                                  # 胶囊网络长度
    # num_filters = 32                          # 卷积核数目
    # kernel_size = 3                           # 卷积核尺寸
    vocab_size = load_embedding()[1]            # 词汇表达小 使用gensim 设置的值

    # hidden_dim = 180                          # 全连接层神经元

    dropout_keep_prob = 0.5                     # dropout保留比例
    learning_rate = 1e-3                        # 学习率

    batch_size = 128                            # 每批训练大小
    num_epochs = 2                             # 总迭代轮次

    print_per_batch = 10                        # 每多少轮输出一次结果
    save_per_batch = 100                        # 每多少轮存入tensorboard
    m_plus=0.9                                  #'the parameter of m plus')
    m_minus= 0.1                                #'the parameter of m minus')
    lambda_val=0.5                              #'down weight of the loss for absent digit classes')
    iter_routing=3                              #'number of iterations in routing algorithm')
    mask_with_y= True                          #use the true label to mask out target capsule or not')

    stddev=0.01                                 #, 'stddev for W initializer')
    regularization_scale=0.5                  #'regularization coefficient for reconstruction loss, default to 0.0005*784=0.392')
    is_training=True                           #'train or predict phase')
    num_threads=8                               # 'number of threads of enqueueing examples')
    logdir='./logdir'                              #', 'logs directory')
    train_sum_freq=100                          #, 'the frequency of saving train summary(step)')
    val_sum_freq=500                            #, 'the frequency of saving valuation summary(step)')
    save_freq=3                                 #, 'the frequency of saving model(epoch)')
    results='./results'                            #', 'path for saving results')

# from keras.layers import K
# K.conv1d
# tf.nn.conv1d