# -*- coding: utf-8 -*-
# @Author  : lzh
# @FileName: model.py
# @Software: PyCharm


from keras.layers import Input, Embedding, Convolution1D, MaxPooling1D, Concatenate, Dropout,Conv1D,Add, Dense,TimeDistributed
from keras.layers import Flatten, Dense, Activation, BatchNormalization, CuDNNGRU, CuDNNLSTM, Lambda,GRU,Multiply,Activation
from keras.models import Model, Sequential
from keras import backend as K
from keras.regularizers import l2
from keras.layers.wrappers import Bidirectional
from keras.optimizers import Adam
from keras.layers import Reshape
import tensorflow as tf
# from keras.activations import gelu
def create_umap_model(model):
    umap_input = model.input
    umap_output = model.layers[12].output  # 选择要可视化的层，根据模型结构进行调整
    umap_model = Model(inputs=umap_input, outputs=umap_output)
    return umap_model

def base(length, out_length, para):

    ed = para['embedding_dimension']
    ps = para['pool_size']
    fd = para['fully_dimension']
    dp = para['drop_out']
    lr = para['learning_rate']
    l2value = 0.001

    main_input = Input(shape=(length,), dtype='int64', name='main_input')

    x = Embedding(output_dim=ed, input_dim=21, input_length=length)(main_input)

    a = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    apool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(a)

    b = Convolution1D(64, 3, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    bpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(b)

    c = Convolution1D(64, 8, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    cpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(c)


    merge = Concatenate(axis=-1)([apool, bpool, cpool])
    merge = Dropout(dp)(merge)

    x = Flatten()(merge)

    x = Dense(fd, activation='relu', name='FC1', W_regularizer=l2(l2value))(x)

    output = Dense(out_length, activation='sigmoid', name='output', W_regularizer=l2(l2value))(x)

    model = Model(inputs=main_input, output=output)
    adam = Adam(lr=lr)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    return model


def BiGRU_base(length, out_length, para):

    ed = para['embedding_dimension']
    ps = para['pool_size']
    fd = para['fully_dimension']
    dp = para['drop_out']
    lr = para['learning_rate']
    l2value = 0.001

    main_input = Input(shape=(length,), dtype='int64', name='main_input')

    x = Embedding(output_dim=ed, input_dim=27, input_length=length)(main_input)

    a = Convolution1D(64, 2, activation='relu', border_mode='same', name='Conv_a', W_regularizer=l2(l2value))(x)
    apool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(a)

    b = Convolution1D(64, 3, activation='relu', border_mode='same',name='Conv_b',W_regularizer=l2(l2value))(x)
    bpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(b)

    c = Convolution1D(64, 8, activation='relu', border_mode='same', name='Conv_c', W_regularizer=l2(l2value))(x)
    cpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(c)


    merge = Concatenate(axis=-1)([apool, bpool, cpool])
    merge = Dropout(dp)(merge)


    # x = Bidirectional(CuDNNGRU(50, return_sequences=True))(merge)
    x = Bidirectional(GRU(50, return_sequences=True), name='BiGRU')(merge)
    x = Flatten()(x)

    x = Dense(fd, activation='relu', name='FC1', W_regularizer=l2(l2value))(x)

    # output = Dense(out_length, activation='sigmoid', name='output')(x)
    output = Dense(out_length, activation='sigmoid', name='output', W_regularizer=l2(l2value))(x)

    model = Model(inputs=main_input, output=output)
    adam = Adam(lr=lr)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    return model






