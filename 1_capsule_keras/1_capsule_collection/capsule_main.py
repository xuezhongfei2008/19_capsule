#encoding:utf-8
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from keras.layers import Dense,Input,LSTM,Bidirectional,Activation,Conv1D,GRU
from keras.callbacks import Callback
from keras.layers import Dropout,Embedding,GlobalMaxPooling1D, MaxPooling1D, Add, Flatten
from keras.preprocessing import text, sequence
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from keras import utils
from keras.layers import *
from keras.layers import *
import re
from capsLayer import *


########################################## 1-数据预处理###################################
embedding_file = './input/dialogue_vectors.txt'
train= pd.read_table('./input/train1600.txt',names=['text','class'])
test = pd.read_table('./input/test1600.txt',names=['text','class'])
max_features=50000
maxlen=20                       #句子最大长度
embed_size=300                  #embedding维度


def del_label(doc):
    return re.findall("__label__(.*)",doc)[0]

train['class']=train['class'].apply(del_label)
test['class']=test['class'].apply(del_label)
X_train = train["text"]
Y_train = train["class"]
X_test = test["text"]
Y_test=test["class"]

'''x word_to_id'''
tok=text.Tokenizer(num_words=max_features,lower=True)
tok.fit_on_texts(list(X_train)+list(X_test))
XX_train=tok.texts_to_sequences(X_train)
XX_test=tok.texts_to_sequences(X_test)
x_train=sequence.pad_sequences(XX_train,maxlen=maxlen)
x_test=sequence.pad_sequences(XX_test,maxlen=maxlen)


tok_y=text.Tokenizer(num_words=100)
tok_y.fit_on_texts(list(Y_train)+list(Y_test))

YY_train=tok_y.texts_to_sequences(Y_train)
YY_train=np.reshape(np.array(YY_train),np.array(YY_train).shape[0])-1

YY_test=tok_y.texts_to_sequences(Y_test)
YY_test=np.reshape(np.array(YY_test),np.array(YY_test).shape[0])-1

num_class=YY_train.max()-YY_train.min()+1
y_train = utils.to_categorical(YY_train, num_class)
y_test = utils.to_categorical(YY_test, num_class)

embeddings_index = {}
with open(embedding_file,encoding='utf8') as f:
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs


word_index = tok.word_index
#prepare embedding matrix
num_words = min(max_features, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embed_size))
for word, i in word_index.items():
    if i >= max_features:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector



##################################### 3 -构建网络 ###############################
from keras.layers import K, Activation
from keras.engine import Layer
from keras.layers import Dense, Input, Embedding, Dropout, Bidirectional, GRU, Flatten, SpatialDropout1D

gru_len = 128
Routings = 3
Num_capsule = 21
Dim_capsule = 16
# dropout_p = 0.25
dropout_p = 0.5
rate_drop_dense = 0.28

def get_model():
    input1 = Input(shape=(maxlen,))
    embed_layer = Embedding(len(word_index) + 1,
                            embed_size,
                            input_length=maxlen,
                            weights=[embedding_matrix],
                            trainable=False)(input1)
#     embed_layer = SpatialDropout1D(rate_drop_dense)(embed_layer)

    x = Bidirectional(
        GRU(gru_len, activation='relu', dropout=dropout_p, recurrent_dropout=dropout_p, return_sequences=True))(
        embed_layer)
    capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings,
                      share_weights=True)(x)
#     output = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)))(capsule)
    capsule = Flatten()(capsule)
#     capsule = Dropout(dropout_p)(capsule)
    dense1 = Dense(128, activation='relu')(capsule)
    output = Dense(21, activation='sigmoid')(dense1)
    model = Model(inputs=input1, outputs=output)
    model.compile(
#         loss=lambda y_true,y_pred: y_true*K.relu(0.9-y_pred)**2 + 0.5*(1-y_true)*K.relu(y_pred-0.1)**2,
#         loss='binary_crossentropy',
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    model.summary()
    return model

model = get_model()

batch_size = 128
X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train, train_size=0.9, random_state=233)
RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)

hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=500, validation_data=(X_val, y_val),verbose=1)


y_pred = model.predict(x_test, batch_size=1024, verbose=1)

print("测试数据准确率：",np.sum(YY_test==y_pred.argmax(axis=1))/len(YY_test)*100)













