# -*- coding: utf-8 -*-

import tensorflow as tf
from capsLayer import CapsLayer
from config import TCNNConfig as cfg
from utils import reduce_sum
from utils import softmax

epsilon = 1e-9


class TextCNN(object):
    """文本分类，CNN模型"""

    def __init__(self, ):

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, cfg.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, cfg.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.cnn()

    def input_embedding(self):
        # 词向量的转换# 词向量映射
        with tf.device('/cpu:0'):
            # 当trainable 值设为True时，该模型就是non_static,当trainable值设为False时模型就是static的
            embedding = tf.get_variable('embedding', trainable=True, initializer=cfg.embedding_array)
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)
            embedding_inputs = tf.expand_dims(embedding_inputs, -1)
        return embedding_inputs

    def cnn(self):
        """CNN模型"""
        embedding_inputs = self.input_embedding()
        filter_sizes = [[1, 300], [2, 300], [3, 300], [5, 300]]
        global all_conv
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("cnn%s" % filter_size[0]):
                # filter_shape=[filter_size[0],cfg.embedding_dim,1,cfg.num_filters]
                filter_shape = [filter_size[0], cfg.embedding_dim, 1, filter_size[1]]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                conv = tf.nn.conv2d(
                    embedding_inputs,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                conv = tf.reshape(conv, shape=[-1, filter_size[1], conv.shape[1], 1])
                if i == 0:
                    all_conv = conv
                else:
                    all_conv = tf.concat([all_conv, conv], axis=2)

        digitCaps = CapsLayer(num_outputs=cfg.num_classes, vec_len=cfg.vec_len, with_routing=True, layer_type='FC')
        self.caps2 = digitCaps(all_conv)
        print("self.caps2",self.caps2)

        # self.cap_flatten=tf.reshape(self.caps2,[-1,cfg.num_classes*cfg.vec_len])    #映射成一个 num_filters_total 维的特征向量
        # print("self.cap_flatten", self.cap_flatten.shape)

        with tf.variable_scope('Masking'):
            # a). calc ||v_c||, then do softmax(||v_c||)
            # [batch_size, 10, 16, 1] => [batch_size, 10, 1, 1]
            self.v_length = tf.sqrt(reduce_sum(tf.square(self.caps2),
                                               axis=2, keepdims=True) + epsilon)
            # print("self.v_length",self.v_length)
            # 计算 v 向量的模
            self.softmax_v = softmax(self.v_length, axis=1)
            # print("self.softmax_v",self.softmax_v)
            # 对每个低层胶囊i而言，所有权重cij的总和等于1。
            # assert self.softmax_v.get_shape() == [cfg.batch_size, self.num_label, 1, 1]

            # b). pick out the index of max softmax val of the 10 caps
            # [batch_size, 10, 1, 1] => [batch_size] (index)
            self.argmax_idx = tf.to_int32(tf.argmax(self.softmax_v, axis=1))
            # print("self.argmax_idx",self.argmax_idx)
            # 获取最佳的预测id
            # assert self.argmax_idx.get_shape() == [cfg.batch_size, 1, 1]
            self.argmax_idx = tf.reshape(self.argmax_idx, shape=(cfg.batch_size,))
            # print("self.argmax_idx",self.argmax_idx)
            # Method 1.
            if not cfg.mask_with_y:
                self.masked_v=tf.reshape(self.caps2,(-1,cfg.num_classes,cfg.vec_len))
                # # c). indexing
                # # It's not easy to understand the indexing process with argmax_idx
                # # as we are 3-dim animal
                # masked_v = []
                # for batch_size in range(cfg.batch_size):
                #     v = self.caps2[batch_size][self.argmax_idx[batch_size], :]
                #     # print("v",v)
                #     masked_v.append(tf.reshape(v, shape=(1, 1, 16, 1)))
                #
                # self.masked_v = tf.concat(masked_v, axis=0)
                # # print("self.masked_v",self.masked_v )
                # assert self.masked_v.get_shape() == [cfg.batch_size, 1, 16, 1]
            # Method 2. masking with true label, default mode
            else:
                self.masked_v = tf.multiply(tf.squeeze(self.caps2), tf.reshape(self.input_y, (-1, cfg.num_classes, 1)))
                '''
                请注意，它在训练时仅使用正确的DigitCap向量，忽略不正确的DigitCap,取出正确的DigitCap向量
                '''
                self.v_length = tf.sqrt(reduce_sum(tf.square(self.caps2), axis=2, keepdims=True) + epsilon)
                print("self.masked_v2", self.masked_v)
                # print("self.v_length2",self.v_length)

        # 2. Reconstructe the MNIST images with 3 FC layers
        # [batch_size, 1, 16, 1] => [batch_size, 16] => [batch_size, 512]

        with tf.name_scope("score"):
            vector_j = tf.reshape(self.masked_v, shape=(cfg.batch_size, -1))
            self.logits = tf.layers.dense(vector_j, cfg.num_classes, name='fc2')
            # self.y_pred = tf.contrib.layers.fully_connected(vector_j,
            #                                                 num_outputs=cfg.num_classes,
            #                                                 activation_fn=tf.sigmoid)

            # 输出层,分类器
        # self.logits = tf.layers.dense(cur_layer, cfg.num_classes, name='fc2')
        self.logits_softmax = tf.nn.softmax(self.logits)
        # self.logits1 = tf.nn.local_response_normalization(self.logits,dim = 0)
        # print("self.logits", self.logits.shape)
        self.y_pred = tf.argmax(self.logits_softmax, 1)  # 预测类别
        # print("self.y_pred",self.y_pred.shape)

        with tf.name_scope("loss"):
            # 使用优化方式，损失函数，交叉熵
            # 1. The margin loss

            # [batch_size, 10, 1, 1]
            # max_l = max(0, m_plus-||v_c||)^2
            max_l = tf.square(tf.maximum(0., cfg.m_plus - self.v_length))
            # max_r = max(0, ||v_c||-m_minus)^2
            max_r = tf.square(tf.maximum(0., self.v_length - cfg.m_minus))
            '''
            当正确DigitCap预测正确标签的概率大于0.9时，损失函数为零，当概率小于0.9时，损失函数不为零。
            '''
            assert max_l.get_shape() == [cfg.batch_size, cfg.num_classes, 1, 1]

            # reshape: [batch_size, 10, 1, 1] => [batch_size, 10]
            max_l = tf.reshape(max_l, shape=(cfg.batch_size, -1))
            max_r = tf.reshape(max_r, shape=(cfg.batch_size, -1))

            # calc T_c: [batch_size, 10]
            # T_c = Y, is my understanding correct? Try it.
            T_c = self.input_y
            # [batch_size, 10], element-wise multiply
            L_c = T_c * max_l + cfg.lambda_val * (1 - T_c) * max_r

            self.margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))

            # 2. The reconstruction loss
            # print("self.input_y", self.input_y)
            # orgin = tf.reshape(self.input_y, shape=(cfg.batch_size, -1))
            # print("self.y_pred",self.y_pred)
            # print("orgin",orgin)
            squared = tf.square(self.logits_softmax - self.input_y)
            self.reconstruction_err = tf.reduce_mean(squared)

            # 3. Total loss
            # The paper uses sum of squared error as reconstruction error, but we
            # have used reduce_mean in `# 2 The reconstruction loss` to calculate
            # mean squared error. In order to keep in line with the paper,the
            # regularization scale should be 0.0005*10=0.005
            self.loss = self.margin_loss + cfg.regularization_scale * self.reconstruction_err
            # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            # self.loss = tf.reduce_mean(cross_entropy)
        with tf.name_scope("optimize"):
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=cfg.learning_rate).minimize(self.loss)
        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(self.y_pred, tf.argmax(self.input_y, 1))
            self.acc = tf.reduce_mean(tf.cast(correct_pred, "float"), name="accuracy")
