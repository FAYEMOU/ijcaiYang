#
import os
import numpy as np
import random as rn 
import tensorflow as tf 
from keras import backend as K # dir(keras)

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPooling2D
from keras.layers import Reshape, Flatten, Concatenate, Dropout
from keras.layers import LSTM, GRU, Bidirectional

from keras.layers import Activation, ZeroPadding2D, BatchNormalization 
from keras.layers import AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPool2D

from keras.utils.visualize_util import plot
from keras.utils import layer_utils

import warnings
warnings.filterwarnings("ignore")

# no atten in keras that needed to defined
class Attention(Layer):
    def __init__(self, attention_size, **kwargs): # 初始化一些需要的參數
        self.attention_size = attention_size
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape): # 具體定義權重
        # W: (EMBED_SIZE, ATTENTION_SIZE)
        # b: (ATTENTION_SIZE, 1)
        # u: (ATTENTION_SIZE, 1)
        self.W = self.add_weight(name="W_{:s}".format(self.name),
                                 shape=(input_shape[-1], self.attention_size),
                                 initializer="glorot_normal",
                                 trainable=True)
        self.b = self.add_weight(name="b_{:s}".format(self.name),
                                 shape=(input_shape[1], 1),
                                 initializer="zeros",
                                 trainable=True)
        self.u = self.add_weight(name="u_{:s}".format(self.name),
                                 shape=(self.attention_size, 1),
                                 initializer="glorot_normal",
                                 trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x, mask=None): # 核心部分定義向量是如何進行運算的
        # input: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        # et: (BATCH_SIZE, MAX_TIMESTEPS, ATTENTION_SIZE)
        et = K.tanh(K.dot(x, self.W) + self.b)
        # at: (BATCH_SIZE, MAX_TIMESTEPS)
        at = K.softmax(K.squeeze(K.dot(et, self.u), axis=-1))
        if mask is not None:
            at *= K.cast(mask, K.floatx())
        # ot: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        atx = K.expand_dims(at, axis=-1)
        ot = atx * x
        # output: (BATCH_SIZE, EMBED_SIZE)
        output = K.sum(ot, axis=1)
        return output

    def compute_mask(self, input, input_mask=None):
        return None

    def compute_output_shape(self, input_shape): # 定義該層輸出的大小
        return (input_shape[0], input_shape[-1])


def CNN_Encoder(*args):

	fact_inputs = Input(shape=(args['num_fact'],), dtype='int32', name='fact_description') # 實例化keras張量
	
	# embedding layer，自帶學習權重，輸出為二維向量，每個單詞在輸入文本序列中嵌入一個
	X_embedding = Embedding(input_dim=args['vocab_size'],
					   output_dim=args['embedding_size'], #100, # args['embedding_size']輸出向量維度大小
					   input_length=args['num_fact'],
					   trainable=True)(fact_inputs) # the first layer of a model
	#X_embedding_expanded = K.expand_dims(X_embedding, axis=-1)# 增加維度
	X_embedding_expanded = Reshape((args['num_fact'], args['embedding_size'], 1))(X_embedding)

	# cnn + maxpooling
	C_cnn0 = Conv2D(filter=args['num_filters'], kernel_size=(args['filter_size'][0], args['embedding_size']), padding='valid', kernel_initializer='normal', activation='relu')(X_embedding_expanded)
	C_cnn1 = Conv2D(filter=args['num_filters'], kernel_size=(args['filter_size'][1], args['embedding_size']), padding='valid', kernel_initializer='normal', activation='relu')(X_embedding_expanded)
	C_cnn2 = Conv2D(filter=args['num_filters'], kernel_size=(args['filter_size'][2], args['embedding_size']), padding='valid', kernel_initializer='normal', activation='relu')(X_embedding_expanded)
	C_cnn3 = Conv2D(filter=args['num_filters'], kernel_size=(args['filter_size'][3], args['embedding_size']), padding='valid', kernel_initializer='normal', activation='relu')(X_embedding_expanded)

	C_maxpool0 = MaxPooling2D(pool_size=(args['num_fact']-args['filter_size'][0]+1, 1), strides=(1,1), padding='valid')(C_cnn0)
	C_maxpool1 = MaxPooling2D(pool_size=(args['num_fact']-args['filter_size'][1]+1, 1), strides=(1,1), padding='valid')(C_cnn1)
	C_maxpool2 = MaxPooling2D(pool_size=(args['num_fact']-args['filter_size'][2]+1, 1), strides=(1,1), padding='valid')(C_cnn2)
	C_maxpool3 = MaxPooling2D(pool_size=(args['num_fact']-args['filter_size'][2]+1, 1), strides=(1,1), padding='valid')(C_cnn3)

	concatenated_tensor = Concatenate(axis=1)([C_maxpool0,C_maxpool1,C_maxpool2,C_maxpool3])
	flatten = Flatten()(concatenated_tensor)
	dropout = Dropout(0.5)(flatten)

	fact_outputs = Dense(units=??, activation='softmax', name='fact_encoder')(dropout) #full-con
    #               y_labels的長度不知道。。
	model = Model(inputs=fact_inputs, outputs=fact_outputs)

	return model

def lstm_layer(*args):
	lstm_input = Input(shape=(args['num_fact'],), dtype=int32, name='collocation_attention')

	lstm_embedding = Embedding(input_dim=args['vocab_size'],
						output_dim=args['embedding_size'],
						input_length=args['num_fact'],
						trainable=True)(lstm_input)
	
	bi_lstm = Bidirectional(LSTM(args['lstm_hidden_layer_size'], return_sequences=True, return_state=True))(lstm_embedding)

	bi_lstm_output = Dense(units=??, activation='softmax', name='lstm_layer')(bi_lstm)

	model = Model(inputs=lstm_input, outputs=bi_lstm_output)

	return model

def lstm_att_layer(*args):
	res_input = Input(shape=(args['num_fact'],), dtype=int32, name='result_distribution')

    res_embedding = Embedding(input_dim=args['vocab_size'],
                        output_dim=args['embedding_size'],
                        input_length=args['num_fact'],
                        trainable=True)(res_input)

    res_lstm = Bidirectional(LSTM(args['lstm_hidden_layer_size'], return_sequences=True, return_state=True))(res_embedding)
    res_att = Attention()(res_lstm)
    res_output = Dense(units=??，activation='softmax', name='lstm_att_layer')(res_att)

    model = Model(inputs=res_input, outputs=res_output)

    return model

# visualization

