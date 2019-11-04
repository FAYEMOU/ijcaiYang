
import os
os.environ['PYTHONHASHSEED'] = '0' # 防止脚本以不同順序迭代dict内容，還有？？
import numpy as np
np.random.seed(12)
import random as rn 
rn.seed(233)
import tensorflow as tf 
tf.set_random_seed(2019)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
from keras import backend as K # dir(keras)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

from keras.models import Model
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPooling2D
from keras.layers import Reshape, Flatten, Concatenate, Dropout

from keras.layers import Activation, ZeroPadding2D, BatchNormalization 

from keras.callbacks import LambdaCallback, EarlyStopping
from keras.optimizers import SGD, Adam
from keras.preprocessing.sequence import pad_sequences

from keras.utils.visualize_util import plot
from keras.utils import layer_utils

from layer import CNN_Encoder

import warnings
warnings.filterwarnings("ignore")

class MPBFN(object):
    def __init__(self, *args):
        '''initinalize model'''
        
        self.cnn_encoder = CNN_Encoder()

    
