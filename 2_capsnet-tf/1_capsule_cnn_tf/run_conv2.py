#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
from build_nnmodel import *
from data_processing import *
from config import TCNNConfig as cfg
from sklearn import metrics
import numpy as np
import tensorflow as tf
import time
from datetime import timedelta
from sklearn import preprocessing


base_dir = '/opt/algor/gongxf/python3_pj/nlp_practice/19_capsule/2_capsnet-tf/1_capsule_cnn_tf/data/'
train_dir = os.path.join(base_dir, 'train.txt')
test_dir = os.path.join(base_dir, 'test.txt')
val_dir = os.path.join(base_dir, 'test.txt')
vocab_dir = os.path.join(base_dir, 'vocab_word2vec.txt')

save_dir = '/opt/algor/gongxf/python3_pj/nlp_practice/19_capsule/2_capsnet-tf/1_capsule_cnn_tf/result_conv2d/'
save_path_cnn = os.path.join(save_dir, 'cnn/best_validation')   # 最佳验证结果保存路径
save_path_rnn = os.path.join(save_dir, 'rnn/best_validation')   # 最佳验证结果保存路径


class Run(object):
    def __init__(self):
        self.cnn=True

    def get_time_dif(self,start_time):
        """获取已使用时间"""
        end_time = time.time()
        time_dif = end_time - start_time
        return timedelta(seconds=int(round(time_dif)))

    def feed_data(self,model,x_batch, y_batch, keep_prob):
        feed_dict = {
            model.input_x: x_batch,
            model.input_y: y_batch,
            model.keep_prob: keep_prob
        }
        return feed_dict

    def evaluate(self,model,sess, x_, y_):
        """评估在某一数据上的准确率和损失"""
        data_len = len(x_)
        # print("data_len",data_len)
        batch_eval = batch_iter(x_, y_, cfg.batch_size)
        total_loss = 0.0
        total_acc = 0.0
        for x_batch, y_batch in batch_eval:
            batch_len = len(x_batch)
            # print("batch_len3",batch_len)
            if batch_len==cfg.batch_size:
                feed_dict = self.feed_data(model,x_batch, y_batch, 1.0)
                loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
                total_loss += loss * batch_len
                total_acc += acc * batch_len
            else:
                pass

        return total_loss / data_len, total_acc / data_len

    def train(self):
        # 创建session
        with tf.Graph().as_default():
            session = tf.Session()
            with session.as_default():
                categories, cat_to_id = read_category()
                # print(categories,cat_to_id)
                words, word_to_id = read_vocab(vocab_dir)
                if self.cnn:
                    print('using CNN model...')
                    #cfg = TCNNConfig()
                    # cfg.vocab_size = len(words)
                    # cfg.mask_with_y=True
                    model = TextCNN()
                    tensorboard_dir = '../tensorboard/textcnn'
                else:
                    print('using RNN model...')
                    #cfg = TRNNConfig()
                    # cfg.vocab_size = len(words)
                    model = TextRNN()
                    tensorboard_dir = '../tensorboard/textrnn'
                #
                # print("Configuring TensorBoard and Saver...")
                # # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
                # tensorboard_dir = '/opt/gongxf/python3_pj/Robot/CNN_Classification/tensorboard/textcnn'
                if not os.path.exists(tensorboard_dir):
                    os.makedirs(tensorboard_dir)

                tf.summary.scalar("loss", model.loss)           #生成损失函数标量图
                tf.summary.scalar("accuracy", model.acc)        #生成准确率标量图
                merged_summary = tf.summary.merge_all()
                writer = tf.summary.FileWriter(tensorboard_dir)     #定义一个写入summary的目标文件，dir为写入文件地址
                # 配置 Saver
                # variable_name = [c.name for c in tf.trainable_variables()]
                # print("variable_name",variable_name)
                saver = tf.train.Saver()
                if not os.path.exists(save_dir):
                    print("save_dir",save_dir)
                    os.makedirs(save_dir)

                print("Loading training and validation data...")
                # 载入训练集与验证集
                start_time = time.time()
                #
                x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, cfg.seq_length)
                x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, cfg.seq_length)
                time_dif = self.get_time_dif(start_time)
                print("Time usage:", time_dif)

                session.run(tf.global_variables_initializer())
                writer.add_graph(session.graph)

                print('Training and evaluating...')
                start_time = time.time()
                total_batch = 0              # 总批次
                best_acc_val = 0.0           # 最佳验证集准确率
                last_improved = 0            # 记录上一次提升批次
                require_improvement = 1000   # 如果超过1000轮未提升，提前结束训练

                flag = False
                for epoch in range(cfg.num_epochs):
                    print('Epoch:', epoch + 1)
                    batch_train = batch_iter(x_train, y_train, cfg.batch_size)
                    # print("batch_train",batch_train)
                    #每一轮迭代的次数等于 len(batch_train)/batch_size
                    for x_batch, y_batch in batch_train:
                        # print("x_batch:",len(x_batch))
                        if len(x_batch)==cfg.batch_size:
                            #每次迭代喂入得数据
                            feed_dict = self.feed_data(model,x_batch, y_batch, cfg.dropout_keep_prob)

                            if total_batch % cfg.save_per_batch == 0:
                                # 每多少轮次将训练结果写入tensorboard scalar
                                s = session.run(merged_summary, feed_dict=feed_dict)
                                writer.add_summary(s, total_batch)

                            if total_batch % cfg.print_per_batch == 0:
                                # 每多少轮次输出在训练集和验证集上的性能
                                loss_train, acc_train= session.run([model.loss, model.acc], feed_dict=feed_dict)
                                # print("x_val",len(x_val))
                                loss_val, acc_val = self.evaluate(model,session, x_val, y_val)   # todo
                                if acc_val > best_acc_val:
                                    # 保存最好结果
                                    best_acc_val = acc_val
                                    last_improved = total_batch
                                    # saver.save(sess=session, save_path=save_path)
                                    improved_str = '*'
                                else:
                                    improved_str = ''

                                time_dif = self.get_time_dif(start_time)
                                # msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},'
                                # print(msg.format(total_batch, loss_train, acc_train))

                                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},'\
                                    + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

                            session.run(model.optim, feed_dict=feed_dict)  # 运行优化
                            total_batch += 1
                            # print("total_batch:last_improved",total_batch,last_improved)
                            if total_batch - last_improved > require_improvement:
                                # 验证集正确率长期不提升，提前结束训练
                                print("No optimization for a long time, auto-stopping...")
                                flag = True
                                break  # 跳出循环
                        else:
                            pass
                    #跳出Epoch 迭代训练
                    if flag:  # 同上
                        break
                # variable_name = [v.name for v in tf.trainable_variables()]
                # print(variable_name)
                if self.cnn:
                    saver.save(sess=session, save_path=save_path_cnn)
                else:
                    saver.save(sess=session, save_path=save_path_rnn)
                # session.close()


    def test(self):
        print("Loading test data...")
        start_time = time.time()
        categories, cat_to_id = read_category()
        # print(categories,cat_to_id)
        words, word_to_id = read_vocab(vocab_dir)
        # cfg = TCNNConfig()
        # cfg.vocab_size = len(words)
        if self.cnn:
            #cfg = TCNNConfig()
            # cfg.vocab_size = len(words)
            cfg.mask_with_y = False
            cfg.batch_size = 1
            model = TextCNN()
        else:
            #cfg = TRNNConfig()
            # cfg.vocab_size = len(words)
            model = TextRNN()

        x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, cfg.seq_length)
        print("x_test",len(x_test))
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        # variable_name = [c.name for c in tf.trainable_variables()]
        # print(variable_name)
        saver = tf.train.Saver()
        if self.cnn:
            saver.restore(sess=session, save_path=save_path_cnn)  # 读取保存的模型
        else:
            saver.restore(sess=session, save_path=save_path_rnn)  # 读取保存的模型
        print('Testing...')
        # loss_test, acc_test = self.evaluate(model,session, x_test, y_test)
        # msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
        # print("loss_test----acc_test",msg.format(loss_test, acc_test))

        # batch_size = 1
        data_len = len(x_test)
        num_batch = int((data_len - 1) / cfg.batch_size) + 1
        print("num_batch",num_batch)
        y_test = np.argmax(y_test, 1)
        y_pred = np.zeros(shape=len(x_test), dtype=np.int32) # 保存预测结果
        print("y_test_y_pred",y_test,y_pred)
        for i in range(num_batch):   # 逐批次处理
            start_id = i * cfg.batch_size
            end_id = min((i + 1) * cfg.batch_size, data_len)
            feed_dict = {
                model.input_x: x_test[start_id:end_id],
                model.keep_prob: 1.0
            }
            y_pred[start_id:end_id] = session.run(model.y_pred, feed_dict=feed_dict)
        # 评估
        print("Precision, Recall and F1-Score...")
        print(metrics.classification_report(y_test, y_pred, target_names=categories))
        time_dif = self.get_time_dif(start_time)
        print("Time usage:", time_dif)

    def prediction(self):

        print("Loading test data...")
        categories, cat_to_id = read_category()
        id_to_cate= dict(zip(cat_to_id.values(), cat_to_id.keys()))
        words, word_to_id = read_vocab(vocab_dir)
        if self.cnn:
            #cfg = TCNNConfig()
            # cfg.vocab_size = len(words)
            model = TextCNN()
        else:
            #cfg = TRNNConfig()
            # cfg.vocab_size = len(words)
            model = TextRNN()
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        if self.cnn:
            saver.restore(sess=session, save_path=save_path_cnn)  # 读取保存的模型
        else:
            saver.restore(sess=session, save_path=save_path_rnn)  # 读取保存的模型
        while True:
            text = input("请输入测试问题：")
            x_test = process_file_predict(text, word_to_id, cat_to_id, cfg.seq_length)
            print("x_test", x_test)
            feed_dict = {
                    model.input_x: x_test,
                    model.keep_prob: 1.0
                }
            logits,logits_softmax= session.run([model.logits,model.logits_softmax], feed_dict=feed_dict)
            maximum_probability = np.max(logits_softmax[0])
            index_max = np.where(logits_softmax[0] == maximum_probability)[0][0]
            pre_cate=id_to_cate[index_max]
            print("maximum_probability------index_max",maximum_probability,index_max,pre_cate)

        # logits_norml1=preprocessing.normalize(logits, norm='l1')
        # logits_norml2 = preprocessing.normalize(logits, norm='l2')
        return pre_cate,maximum_probability


if __name__ == '__main__':
    run=Run()
    run.cnn=True
    # run.train()
    run.test()
    # run.prediction()
